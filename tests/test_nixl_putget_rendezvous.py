"""Dependency-free regression tests for the standalone NIXL benchmark."""
import importlib.util
import subprocess
import sys
import unittest
import contextlib
import io
import json
import tempfile
import types
import threading
import queue
from unittest.mock import patch
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples/21_cuda_layer_sweeps/nixl_putget_bench.py"
spec = importlib.util.spec_from_file_location("nixl_putget_bench", SCRIPT)
assert spec is not None
bench = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(bench)


class FakeComm:
    def __init__(self, size=2, rank=0):
        self.size, self.rank = size, rank
        self.calls = []

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def sendrecv(self, value, dest, source):
        self.calls.append((dest, source))
        phase, payload = value
        return phase, b"peer-" + payload

    def allgather(self, value):
        return [value, (value[0], None)]


class RendezvousTests(unittest.TestCase):
    def test_mpi_exchanges_binary_metadata_without_blocking_send_pair(self):
        self.assertTrue(hasattr(bench, "MPIRendezvous"), "MPI rendezvous missing")
        comm = FakeComm()
        rv = bench.MPIRendezvous(comm)
        self.assertEqual(rv.exchange("meta.4KiB", b"\x00\xff"), b"peer-\x00\xff")
        self.assertEqual(comm.calls, [(1, 1)])
        rv.checkpoint("ready")


    def test_mpi_requires_exactly_two_ranks(self):
        for size in (1, 3, 4):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "exactly 2"):
                bench.MPIRendezvous(FakeComm(size=size))

    def test_peer_error_reaches_both_ranks(self):
        comm = FakeComm()
        comm.allgather = lambda value: [(value[0], None), (value[0], "bad registration")]
        with self.assertRaisesRegex(RuntimeError, "rank 1: bad registration"):
            bench.MPIRendezvous(comm).checkpoint("registered")

    def test_phase_mismatch_is_rejected(self):
        comm = FakeComm()
        comm.sendrecv = lambda *a, **kw: ("wrong-phase", b"data")
        with self.assertRaisesRegex(RuntimeError, "phase mismatch"):
            bench.MPIRendezvous(comm).exchange("metadata", b"data")


class PairedRendezvousTests(unittest.TestCase):
    def test_two_concurrent_ranks_exchange_large_binary_messages_and_errors(self):
        mailboxes = [queue.Queue(), queue.Queue()]
        errors = []
        completed = []
        class Comm:
            def __init__(self, rank):
                self.rank = rank
            def Get_size(self):
                return 2
            def Get_rank(self):
                return self.rank
            def sendrecv(self, value, dest, source):
                mailboxes[dest].put(value)
                return mailboxes[self.rank].get(timeout=2)
            def allgather(self, value):
                peer = self.sendrecv(value, 1 - self.rank, 1 - self.rank)
                return [value, peer] if self.rank == 0 else [peer, value]
        def worker(rank):
            try:
                rv = bench.MPIRendezvous(Comm(rank))
                payload = bytes([rank]) * (2 << 20)
                peer = rv.exchange("metadata", payload)
                self.assertEqual(peer, bytes([1 - rank]) * (2 << 20))
                try:
                    rv.checkpoint("complete", "transfer timeout" if rank == 0 else None)
                except RuntimeError as exc:
                    completed.append(str(exc))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertFalse(any(t.is_alive() for t in threads), "rendezvous deadlocked")
        self.assertEqual(errors, [])
        self.assertEqual(completed, ["rank 0: transfer timeout"] * 2)

    def test_file_binary_exchange_and_barrier_work_concurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = []
            errors = []
            def worker(rank):
                try:
                    bench.publish(tmp, f"meta.{rank}", bytes([rank, 0, 255]))
                    bench.barrier(tmp, rank, "metadata", timeout=2)
                    values.append(bench.collect(tmp, f"meta.{1-rank}", timeout=2))
                except Exception as exc:
                    errors.append(exc)
            threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
            self.assertEqual(errors, [])
            self.assertCountEqual(values, [b"\x00\x00\xff", b"\x01\x00\xff"])


    def test_file_waits_use_monotonic_deadlines(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(bench.time, "time", side_effect=AssertionError("wall clock used")):
            with self.assertRaises(TimeoutError):
                bench.collect(tmp, "missing", timeout=0.01)
            with self.assertRaises(TimeoutError):
                bench.barrier(tmp, 0, "missing", timeout=0.01)


class TransferTests(unittest.TestCase):
    def test_done_and_error_states(self):
        class Agent:
            def __init__(self, post, state):
                self.post, self.state = post, state
            def transfer(self, handle):
                return self.post
            def check_xfer_state(self, handle):
                return self.state
        self.assertIsNone(bench.one_transfer(Agent("PROC", "DONE"), "handle", 1))
        for post, state in (("ERR", "DONE"), ("PROC", "ERR")):
            with self.assertRaisesRegex(RuntimeError, "ERR"):
                bench.one_transfer(Agent(post, state), "handle", 1)

    def test_busy_transfer_has_bounded_deadline(self):
        self.assertTrue(hasattr(bench, "one_transfer"), "bounded polling missing")
        class Agent:
            def transfer(self, handle):
                return "PROC"
            def check_xfer_state(self, handle):
                return "PROC"
        with self.assertRaisesRegex(TimeoutError, "transfer.*0.01"):
            bench.one_transfer(Agent(), object(), 0.01)


class Scalar:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class Buffer:
    def __init__(self, size, data=None):
        self.data = bytearray(size) if data is None else data
    def fill_(self, value):
        self.data[:] = bytes([value]) * len(self.data)
    def pin_memory(self):
        return self
    def data_ptr(self):
        return id(self)
    def __getitem__(self, key):
        return Buffer(0, self.data[key])
    def __ne__(self, value):
        return ScalarSum(sum(b != value for b in self.data))


class ScalarSum(Scalar):
    def sum(self):
        return self


class IntegrationComm(FakeComm):
    def __init__(self):
        super().__init__()
        self.phases = []
        self.aborted = []
        self.checkpoints = []
        self.mismatch_config = False
        self.peer_gpu = 0
    def sendrecv(self, value, dest, source):
        phase, payload = value
        self.phases.append(phase)
        if phase == "config" and self.mismatch_config:
            return phase, {"op": "WRITE"}
        if phase == "hosts":
            return phase, {"host": "peer-host", "gpu": self.peer_gpu}
        if phase == "gpus":
            return phase, self.peer_gpu
        return value
    def allgather(self, value):
        self.checkpoints.append(value)
        return [value, (value[0], None)]
    def Abort(self, code):
        self.aborted.append(code)


class IntegratedTests(unittest.TestCase):
    def run_benchmark(self, corruption=False, mode="mpi", op="READ", mem="DRAM", busy=False,
                      unsupported=False, rank=0, extra=(), mismatch_config=False):
        comm = IntegrationComm()
        comm.rank = rank
        comm.mismatch_config = mismatch_config
        buffers = []
        def zeros(size, **kwargs):
            buf = Buffer(size)
            buffers.append(buf)
            return buf
        cuda = types.SimpleNamespace(device_count=lambda: 2, set_device=lambda n: None,
                 get_device_name=lambda n: "fake GPU", synchronize=lambda: None,
                 empty_cache=lambda: None)
        torch = types.SimpleNamespace(zeros=zeros, uint8="uint8", cuda=cuda)
        class Agent:
            nixl_mems = {"DRAM": 1, "VRAM": 2}
            def register_memory(self, buf):
                return "registration"
            def get_xfer_descs(self, *a, **kw):
                return "descs"
            def get_agent_metadata(self):
                return b"\x00metadata\xff"
            def get_serialized_descs(self, descs):
                return b"\x00descs\xff"
            def add_remote_agent(self, metadata):
                return "peer"
            def deserialize_descs(self, descs):
                return "remote descs"
            def initialize_xfer(self, *a):
                if unsupported:
                    raise RuntimeError("WRITE backend unsupported")
                return "handle"
            def transfer(self, handle):
                buffers[-1].fill_(0xAB)
                if corruption:
                    buffers[-1].data[-1] = 0
                return "PROC"
            def check_xfer_state(self, handle):
                return "PROC" if busy else "DONE"
            def release_xfer_handle(self, handle):
                pass
            def deregister_memory(self, reg):
                pass
        modules = {"torch": torch, "nixl": types.ModuleType("nixl"),
                   "nixl._api": types.SimpleNamespace(nixl_agent=lambda *a: Agent(),
                                   nixl_agent_config=lambda **kw: kw),
                   "mpi4py": types.SimpleNamespace(MPI=types.SimpleNamespace(COMM_WORLD=comm))}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [str(SCRIPT), "--sizes", "1048577", "--iters", "1", "--warmup", "0",
                    "--op", op, "--mem", mem, "--json-out", str(output),
                    "--transfer-timeout", "0.01", *extra]
            if mode == "mpi":
                argv += ["--rendezvous", "mpi"]
            else:
                argv += ["--sync-dir", tmp]
                bench.publish(tmp, "host.1", "peer-host")
                for tag in ("hosts", "meta.1MiB", "xfer.1MiB", "done.1MiB"):
                    bench.publish(tmp, "bar." + tag + ".1", b"1")
                bench.publish(tmp, "meta.1MiB.1", b"metadata")
                bench.publish(tmp, "descs.1MiB.1", b"descs")
                bench.publish(tmp, "check.1MiB.1", "true")
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                 patch.dict(bench.os.environ, {"RANK": str(rank), "PALS_RANKID": str(rank)}), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                try:
                    code = bench.main()
                except SystemExit as exc:
                    code = exc.code
            result = json.loads(output.read_text()) if output.exists() else None
            return code, result, stdout.getvalue(), comm

    def test_mpi_main_uses_control_channel_without_sync_directory(self):
        try:
            code, result, stdout, comm = self.run_benchmark()
        except FileNotFoundError as exc:
            self.fail(f"MPI path still depends on sync-dir: {exc}")
        self.assertIn(code, (0, None), stdout)
        self.assertIsNotNone(result)
        self.assertTrue(result["rows"][0]["byte_exact"])
        self.assertIn("hosts", comm.phases)
        self.assertTrue(any("meta" in phase for phase in comm.phases))


    def test_corruption_after_first_mib_is_fatal(self):
        code, result, stdout, comm = self.run_benchmark(corruption=True)
        self.assertNotIn(code, (0, None), "tail corruption was accepted")
        self.assertIn("FAIL", stdout)


    def test_main_uses_bounded_polling_for_all_samples(self):
        with patch.object(bench, "one_transfer", wraps=bench.one_transfer) as polling:
            code, result, stdout, comm = self.run_benchmark()
        self.assertEqual(polling.call_count, 1, "main bypasses bounded transfer helper")

    def test_unexpected_mpi_failure_aborts_peer_instead_of_hanging(self):
        try:
            code, result, stdout, comm = self.run_benchmark(op="WRITE", unsupported=True)
        except RuntimeError as exc:
            self.fail(f"uncoordinated MPI failure: {exc}")
        self.assertNotIn(code, (0, None))
        self.assertIn("unsupported", stdout)
        self.assertEqual(comm.aborted, [1])


    def test_file_remains_default_and_preserves_metadata_filenames(self):
        code, result, stdout, comm = self.run_benchmark(mode="file")
        self.assertEqual(code, 0, stdout)
        self.assertTrue(result["rows"][0]["byte_exact"])
        self.assertEqual(comm.phases, [])

    def test_vram_whole_buffer_correctness(self):
        code, result, stdout, comm = self.run_benchmark(mem="VRAM", corruption=True)
        self.assertEqual(code, 1, stdout)
        self.assertFalse(result["rows"][0]["byte_exact"])

    def test_transfer_timeout_is_fatal_in_main(self):
        code, result, stdout, comm = self.run_benchmark(busy=True)
        self.assertEqual(code, 1, stdout)
        self.assertIn("within 0.01s", stdout)
        self.assertEqual(comm.aborted, [1])


    def test_mpi_configuration_mismatch_fails_before_transfer(self):
        code, result, stdout, comm = self.run_benchmark(mismatch_config=True)
        self.assertEqual(code, 1, "different peer operation was accepted")
        self.assertIn("configuration mismatch", stdout)
        self.assertIsNone(result)
        self.assertEqual(comm.aborted, [1])

    def test_transfer_timeout_is_exchanged_at_completion_checkpoint(self):
        code, result, stdout, comm = self.run_benchmark(busy=True)
        self.assertTrue(any(phase.startswith("xfer.") and error and "0.01s" in error
                            for phase, error in comm.checkpoints),
                        "timeout was not coordinated with target")


    def test_mpi_gpu_placement_checks_actual_peer_device(self):
        original = IntegrationComm.sendrecv
        def exchange(comm, value, dest, source):
            if value[0] == "hosts":
                return value[0], {"host": bench.socket.gethostname().split(".")[0]}
            return original(comm, value, dest, source)
        with patch.object(IntegrationComm, "sendrecv", exchange):
            code, result, stdout, comm = self.run_benchmark()
        self.assertEqual(code, 1, "same physical GPU was accepted")
        self.assertIn("same GPU", stdout)


    def test_write_initiator_reports_target_corruption(self):
        original = IntegrationComm.sendrecv
        def exchange(comm, value, dest, source):
            if value[0].startswith("check."):
                return value[0], False
            return original(comm, value, dest, source)
        with patch.object(IntegrationComm, "sendrecv", exchange):
            code, result, stdout, comm = self.run_benchmark(op="WRITE")
        self.assertEqual(code, 1)
        self.assertFalse(result["rows"][0]["byte_exact"])


    def test_file_corruption_is_fatal_and_reported(self):
        code, result, stdout, comm = self.run_benchmark(mode="file", corruption=True)
        self.assertEqual(code, 1, stdout)
        self.assertFalse(result["rows"][0]["byte_exact"])

    def test_write_target_checks_its_receiving_buffer(self):
        code, result, stdout, comm = self.run_benchmark(op="WRITE", rank=1)
        self.assertEqual(code, 1, "untouched WRITE target buffer was accepted")
        self.assertIn("byte-exact check: FAIL", stdout)

    def test_read_target_does_not_check_a_receiving_buffer(self):
        code, result, stdout, comm = self.run_benchmark(op="READ", rank=1)
        self.assertEqual(code, 0, stdout)
        self.assertNotIn("byte-exact check:", stdout)


class CommandLineTests(unittest.TestCase):
    def test_file_mode_requires_sync_directory_before_importing_torch(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--sync-dir is required", result.stderr)

    def test_invalid_numeric_arguments_are_rejected(self):
        for option, value in (("--transfer-timeout", "nan"), ("--transfer-timeout", "0"),
                ("--rendezvous-timeout", "inf"), ("--iters", "0"), ("--warmup", "-1"),
                ("--sizes", ""), ("--sizes", "0"), ("--sizes", "broken")):
            with self.subTest(option=option, value=value):
                result = subprocess.run([sys.executable, str(SCRIPT), "--sync-dir", "/unused",
                                        option, value], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)

    def test_help_exposes_optional_mpi_and_timeouts(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--rendezvous {file,mpi}", "--transfer-timeout", "--rendezvous-timeout"):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
