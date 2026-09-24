#!/usr/bin/env python3
"""Offline validation of the CUDA layer sweep.

PBS is down, so nothing here touches a cluster. These tests execute
sweep_layers_cuda.sh for real against a fake bin/ directory: stub mpiexec,
stub OSU/nccl binaries, stub python benchmarks. That exercises the actual
argument wiring, the CSV accounting and the status logic, which is where the
bugs in this harness have actually been -- a hardcoded launcher path, a
reduction op passed as an env var instead of -o, and a status function that
reported ok while every stage exited 127.

Run:  python3 -m pytest test_sweep_layers.py -v
"""
import csv
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The driver sits at examples/19_cuda_layer_sweeps/ in the repo and at
# code/19_cuda_layer_sweeps/ in the packaged deliverable. Look in both, and
# allow an explicit override, so the suite runs from either location instead
# of silently failing all 29 tests on a path that does not exist here.
_CANDIDATES = [
    Path(os.environ["DLCOMM_SWEEP"]) if os.environ.get("DLCOMM_SWEEP") else None,
    HERE.parent / "examples" / "19_cuda_layer_sweeps" / "sweep_layers_cuda.sh",
    HERE.parent / "code" / "19_cuda_layer_sweeps" / "sweep_layers_cuda.sh",
]
SWEEP = next((p for p in _CANDIDATES if p and p.is_file()), _CANDIDATES[1])


def sh(script: str) -> str:
    return textwrap.dedent(script).lstrip()


class SweepHarness(unittest.TestCase):
    """Builds a throwaway world where every external command is a stub."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sweeptest-"))
        self.w = self.tmp / "w"
        (self.w / "bin").mkdir(parents=True)
        self.run = self.tmp / "run"
        self.calls = self.tmp / "calls.log"

        # stub mpiexec: log the argv, then exec the trailing command so the
        # stub benchmarks below actually run.
        self._bin("mpiexec", f"""
            #!/bin/bash
            echo "MPIEXEC $*" >> {self.calls}
            args=("$@")
            for ((i=0; i<${{#args[@]}}; i++)); do
              case "${{args[i]}}" in
                -n|-ppn|--hostfile) ((i++)) ;;
                *) exec "${{args[@]:i}}" ;;
              esac
            done
        """)

        # OSU stub: emits a plausible two-column table.
        osu = self.w / "bench_install_gtl/libexec/osu-micro-benchmarks/mpi/collective"
        osu.mkdir(parents=True)
        for c in ("allreduce allgather alltoall reduce bcast reduce_scatter "
                  "barrier gather scatter").split():
            self._exe(osu / f"osu_{c}", f"""
                #!/bin/bash
                echo "OSU {c} $*" >> {self.calls}
                echo "# OSU MPI {c}"
                echo "4096   12.34"
                echo "65536  56.78"
            """)

        # nccl-tests stub: nccl-tests' own row format (leading spaces).
        nt = self.w / "bench_src/nccl-tests/build"
        nt.mkdir(parents=True)
        for b in ("all_reduce all_gather alltoall broadcast reduce_scatter "
                  "reduce sendrecv scatter gather").split():
            self._exe(nt / f"{b}_perf", f"""
                #!/bin/bash
                echo "NCCL {b} $*" >> {self.calls}
                echo "#  size  count  type  redop  time  algbw  busbw"
                echo "     4096   1024  float    sum  10.1   1.23   2.34"
            """)

        # ldd stub: the sweep gates OSU on "is the GTL linked?" and refuses to
        # run device buffers when it is not. Report a GTL line so the tests
        # reach the actual sweep logic; TestOsuGtlGate overrides this to assert
        # the refusal still works.
        self._bin("ldd", """
            #!/bin/bash
            echo "libmpi_gtl_cuda.so.0 => /opt/cray/pe/mpich/9.1.0/gtl/lib/libmpi_gtl_cuda.so.0"
        """)
        self._exe(self.w / "pals_osu_env.sh",   '#!/bin/bash\nexec "$@"\n')
        self._exe(self.w / "pals_nccl_env.sh",  '#!/bin/bash\nexec "$@"\n')
        self._exe(self.w / "pals_torch_env.sh", '#!/bin/bash\nexec "$@"\n')
        self._exe(self.w / "pals_nixl_env.sh",  '#!/bin/bash\nexec "$@"\n')

        self._exe(self.w / "python_stub", f"""
            #!/bin/bash
            echo "PY $*" >> {self.calls}
            echo "        4096      20     0.1234        1.00        2.00"
            exit 0
        """)
        self._exe(self.w / "torch_dist_bench.py", "#!/bin/bash\nexit 0\n")
        self._exe(self.w / "nixl_putget_bench.py", "#!/bin/bash\nexit 0\n")

        fib = self.w / "shs-libfabric-install/bin"
        fib.mkdir(parents=True)
        self._exe(fib / "fi_info", """
            #!/bin/bash
            echo "provider: cxi"
            echo "    domain: cxi0"
            echo "provider: cxi"
            echo "    domain: cxi1"
        """)
        self._exe(fib / "fi_pingpong", """
            #!/bin/bash
            echo "bytes #sent total time MB/sec"
            echo "65536 100 6M 0.1 23436"
        """)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _exe(self, path: Path, body: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sh(body))
        path.chmod(0o755)

    def _bin(self, name: str, body: str):
        self._exe(self.w / "bin" / name, body)

    def sweep(self, layers: str, nodes: str = "1", extra_env=None):
        env = dict(os.environ)
        env.update(
            PATH=f"{self.w/'bin'}:{env['PATH']}",
            DLCOMM_WORKDIR=str(self.w),
            DLCOMM_NNODES=nodes,
            DLCOMM_GPUS_PER_NODE="4",
            DLCOMM_RUN_DIR=str(self.run),
            DLCOMM_LAYERS=layers,
            DLCOMM_QUICK="1",
            DLCOMM_PYTHON=str(self.w / "python_stub"),
        )
        env.pop("PBS_NODEFILE", None)
        if extra_env:
            env.update(extra_env)
        p = subprocess.run(["bash", str(SWEEP)], capture_output=True,
                           text=True, env=env, timeout=300)
        rows = []
        csvf = self.run / "sweep_results.csv"
        if csvf.exists():
            with open(csvf) as fh:
                rows = list(csv.DictReader(fh))
        return p, rows

    def log(self) -> str:
        return self.calls.read_text() if self.calls.exists() else ""


class TestLauncherResolution(SweepHarness):
    def test_resolves_mpiexec_from_path(self):
        """A hardcoded /opt/cray/pe/pals path made every cell exit 127 on a
        machine whose PALS lives at /opt/cray/pals."""
        p, rows = self.sweep("fi")
        self.assertNotIn("no usable mpiexec", p.stdout + p.stderr)
        self.assertTrue(rows)

    def test_fails_loudly_when_no_launcher(self):
        # Keep a usable PATH (bash itself must still resolve) and block only
        # the launcher: DLCOMM_MPIEXEC takes precedence over discovery, so a
        # bad value here is exactly the "no launcher" condition.
        env = dict(os.environ)
        env.update(PATH="/usr/bin:/bin", DLCOMM_WORKDIR=str(self.w),
                   DLCOMM_NNODES="1", DLCOMM_RUN_DIR=str(self.run),
                   DLCOMM_LAYERS="osu", DLCOMM_MPIEXEC="/nonexistent/mpiexec")
        p = subprocess.run(["bash", str(SWEEP)], capture_output=True,
                           text=True, env=env, timeout=120)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("no usable mpiexec", p.stdout + p.stderr)


class TestStatusHonesty(SweepHarness):
    def test_zero_rows_is_never_ok(self):
        """The first harness printed STATUS=ok for five layers while every
        stage exited 127. A cell with no rows must never be ok."""
        for c in (self.w / "bench_install_gtl/libexec/osu-micro-benchmarks/mpi/collective").iterdir():
            self._exe(c, "#!/bin/bash\nexit 127\n")
        _, rows = self.sweep("osu")
        self.assertTrue(rows)
        for r in rows:
            if r["status"] == "ok":
                self.fail(f"cell {r['cell']} reported ok with rows={r['rows']}")

    def test_usage_output_is_failure_not_success(self):
        """An invalid OSU argument form prints usage and exits 0 -- in a batch
        log that is indistinguishable from a real run."""
        tgt = self.w / "bench_install_gtl/libexec/osu-micro-benchmarks/mpi/collective/osu_allreduce"
        self._exe(tgt, '#!/bin/bash\necho "Usage: osu_allreduce [-d TYPE]"\nexit 0\n')
        _, rows = self.sweep("osu")
        cells = [r for r in rows if r["collective"] == "allreduce"]
        self.assertTrue(cells)
        for r in cells:
            self.assertEqual(r["status"], "failed", f"usage text accepted: {r}")
            self.assertEqual(r["detail"], "usage_printed_invalid_args")

    def test_good_run_reports_ok(self):
        _, rows = self.sweep("osu")
        self.assertTrue(any(r["status"] == "ok" for r in rows))


class TestOsuAxes(SweepHarness):
    def test_sweeps_device_and_host_for_every_collective(self):
        _, rows = self.sweep("osu")
        colls = {r["collective"] for r in rows if r["layer"] == "osu"}
        for expect in ("allreduce", "allgather", "alltoall", "reduce",
                       "bcast", "reduce_scatter", "barrier", "gather", "scatter"):
            self.assertIn(expect, colls)
        for c in colls:
            bufs = {r["value"] for r in rows if r["collective"] == c}
            self.assertEqual(bufs, {"device", "host"}, f"{c} missing a buffer type")

    def test_device_runs_enable_gpu_support_host_runs_do_not(self):
        """MPICH_GPU_SUPPORT_ENABLED must be per-cell: leaking it into a
        non-GTL step aborts that step and looks like a second bug."""
        self._bin("mpiexec", f"""
            #!/bin/bash
            echo "GPUFLAG=${{MPICH_GPU_SUPPORT_ENABLED:-unset}} ARGS=$*" >> {self.calls}
            args=("$@")
            for ((i=0; i<${{#args[@]}}; i++)); do
              case "${{args[i]}}" in
                -n|-ppn|--hostfile) ((i++)) ;;
                *) exec "${{args[@]:i}}" ;;
              esac
            done
        """)
        self.sweep("osu")
        for line in self.log().splitlines():
            if "GPUFLAG=" not in line:
                continue
            if "-d cuda" in line:
                self.assertIn("GPUFLAG=1", line)
            elif "osu_" in line:
                self.assertIn("GPUFLAG=unset", line)


class TestOsuGtlGate(SweepHarness):
    def test_refuses_osu_when_gtl_absent(self):
        """A non-GTL OSU build segfaults the instant it touches device
        buffers. The sweep must refuse it with a reason rather than run it."""
        self._bin("ldd", '#!/bin/bash\necho "libc.so.6 => /lib/libc.so.6"\n')
        _, rows = self.sweep("osu")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["status"], "unavailable")
        self.assertIn("GTL", rows[0]["detail"])


class TestIndentedRows(SweepHarness):
    def test_indented_barrier_value_counts_as_a_row(self):
        """osu_barrier indents its single latency value. Counting only
        lines starting with a digit in column 1 scored a real 5.60 us
        measurement as rows=0 -> failed."""
        tgt = (self.w / "bench_install_gtl/libexec/osu-micro-benchmarks"
                        "/mpi/collective/osu_barrier")
        self._exe(tgt, """
            #!/bin/bash
            echo "# OSU MPI-CUDA Barrier Latency Test v7.5"
            echo "# Avg Latency(us)"
            echo "             5.60"
        """)
        _, rows = self.sweep("osu")
        bar = [r for r in rows if r["collective"] == "barrier"]
        self.assertTrue(bar, "no barrier cells emitted")
        for r in bar:
            self.assertEqual(r["rows"], "1", f"indented value not counted: {r}")
            self.assertEqual(r["status"], "ok", f"real measurement marked {r['status']}")

    def test_indented_pingpong_rows_count(self):
        """Same habit in libfabric output."""
        fib = self.w / "shs-libfabric-install/bin"
        self._exe(fib / "fi_pingpong", """
            #!/bin/bash
            echo "bytes   #sent   #ack   total   time   MB/sec   usec/xfer"
            echo "  65536   100    100    12M    0.10   23436.0      2.70"
        """)
        _, rows = self.sweep("fi", nodes="2")
        pp = [r for r in rows if r["cell"] == "pingpong"]
        self.assertTrue(pp)
        for r in pp:
            self.assertEqual(r["rows"], "1", f"indented pingpong row missed: {r}")


class TestNcclAxes(SweepHarness):
    def test_all_axes_present(self):
        _, rows = self.sweep("nccl")
        axes = {r["axis"] for r in rows if r["layer"] == "nccl"}
        self.assertEqual(axes, {"collective", "dtype", "redop", "algo", "proto"})

    def test_redop_uses_dash_o_flag(self):
        """nccl-tests takes -o <op>. Passing it as NCCL_TESTS_OP silently ran
        every redop cell as the default sum."""
        self.sweep("nccl")
        log = self.log()
        for op in ("max", "min", "prod"):
            self.assertRegex(log, rf"NCCL all_reduce .*-o {op}",
                             f"redop {op} not passed as -o")

    def test_redop_not_passed_to_nonreducing_collectives(self):
        self.sweep("nccl")
        for line in self.log().splitlines():
            if line.startswith("NCCL all_gather") or line.startswith("NCCL alltoall"):
                self.assertNotIn(" -o ", line)

    def test_algo_and_proto_exported(self):
        self._bin("mpiexec", f"""
            #!/bin/bash
            echo "ALGO=${{NCCL_ALGO:-unset}} PROTO=${{NCCL_PROTO:-unset}}" >> {self.calls}
            args=("$@")
            for ((i=0; i<${{#args[@]}}; i++)); do
              case "${{args[i]}}" in
                -n|-ppn|--hostfile) ((i++)) ;;
                *) exec "${{args[@]:i}}" ;;
              esac
            done
        """)
        self.sweep("nccl")
        log = self.log()
        self.assertIn("ALGO=Ring", log)
        self.assertIn("ALGO=Tree", log)
        self.assertIn("PROTO=LL128", log)

    def test_dtype_reaches_the_binary(self):
        self.sweep("nccl")
        self.assertIn("-d bfloat16", self.log())


class TestNixlAxes(SweepHarness):
    def test_op_by_memory_matrix(self):
        _, rows = self.sweep("nixl")
        cells = {r["cell"] for r in rows if r["layer"] == "nixl"}
        self.assertEqual(cells, {"READ_VRAM", "READ_DRAM",
                                 "WRITE_VRAM", "WRITE_DRAM"})

    def test_cxi_write_rejection_is_unsupported_not_failed(self):
        """WRITE returns -260 Flags not supported on this CXI stack. That is a
        platform limit, not a broken benchmark, and must not read as a bug."""
        self._exe(self.w / "python_stub", """
            #!/bin/bash
            echo "ERROR: nixl transfer failed: -260 Flags not supported"
            exit 1
        """)
        _, rows = self.sweep("nixl")
        w = [r for r in rows if r["cell"].startswith("WRITE")]
        self.assertTrue(w)
        for r in w:
            self.assertEqual(r["status"], "unsupported")

    def test_write_backend_error_is_unsupported(self):
        """CXI cannot do RMA WRITE: postXferReq raises NIXL_ERR_BACKEND."""
        self._exe(self.w / "python_stub", """
            #!/bin/bash
            case "$*" in
              *"--op WRITE"*)
                 echo "nixl_cu12._bindings.nixlBackendError: NIXL_ERR_BACKEND"
                 exit 1 ;;
              *) echo "[nixl_putget] byte-exact check: PASS"; exit 0 ;;
            esac
        """)
        _, rows = self.sweep("nixl", nodes="2")
        w = [r for r in rows if r["cell"].startswith("WRITE")]
        r = [r for r in rows if r["cell"].startswith("READ")]
        self.assertTrue(w and r)
        for row in w:
            self.assertEqual(row["status"], "unsupported", f"{row}")
        for row in r:
            self.assertEqual(row["status"], "ok", f"{row}")

    def test_read_backend_error_is_still_a_failure(self):
        """The same error on READ is a real fault -- it must not be excused."""
        self._exe(self.w / "python_stub", """
            #!/bin/bash
            case "$*" in
              *"--op READ"*)
                 echo "nixl_cu12._bindings.nixlBackendError: NIXL_ERR_BACKEND"
                 exit 1 ;;
              *) echo "[nixl_putget] byte-exact check: PASS"; exit 0 ;;
            esac
        """)
        _, rows = self.sweep("nixl", nodes="2")
        r = [x for x in rows if x["cell"].startswith("READ")]
        self.assertTrue(r)
        for row in r:
            self.assertEqual(row["status"], "failed",
                             f"READ backend error was excused as {row['status']}")

    def test_one_node_is_unsupported_not_attempted(self):
        """A second LIBFABRIC agent cannot construct on one node (CXI has no
        loopback), so 1-node NIXL must be reported unsupported with a reason
        rather than run and recorded as a failure."""
        _, rows = self.sweep("nixl", nodes="1")
        nixl = [r for r in rows if r["layer"] == "nixl"]
        self.assertEqual(len(nixl), 4, "expected READ/WRITE x VRAM/DRAM")
        for r in nixl:
            self.assertEqual(r["status"], "unsupported", f"{r['cell']}: {r}")
            self.assertIn("needs_2_nodes", r["detail"])
        self.assertNotIn("--expect same-node", self.log(),
                         "1-node NIXL should not be launched at all")

    def test_two_node_expects_cross_node_placement(self):
        self.sweep("nixl", nodes="2")
        self.assertIn("--expect cross-node", self.log())


class TestFiLayer(SweepHarness):
    def test_one_node_records_loopback_reason_not_a_zero(self):
        """CXI has no loopback (fi_domain() returns -38). A skipped pingpong
        must carry its reason, never a 0 that looks like a measurement."""
        _, rows = self.sweep("fi", nodes="1")
        pp = [r for r in rows if r["cell"] == "pingpong"]
        self.assertTrue(pp)
        self.assertEqual(pp[0]["status"], "unavailable")
        self.assertIn("loopback", pp[0]["detail"])

    def test_two_nodes_sweep_all_four_rails(self):
        _, rows = self.sweep("fi", nodes="2")
        rails = {r["value"] for r in rows if r["cell"] == "pingpong"}
        self.assertEqual(rails, {"cxi0", "cxi1", "cxi2", "cxi3"})

    def test_provider_count_is_recorded(self):
        _, rows = self.sweep("fi")
        inv = [r for r in rows if r["cell"] == "inventory"][0]
        self.assertEqual(inv["status"], "ok")
        self.assertEqual(inv["rows"], "2")


class TestLoaderErrorsAreNotMeasurements(SweepHarness):
    def test_fi_info_127_is_failed_not_zero_providers(self):
        """fi_info exiting 127 on a missing .so was recorded as providers=0,
        which reads as 'no fabric on this node'. It must read as failed, with
        the missing library named."""
        fib = self.w / "shs-libfabric-install/bin"
        self._exe(fib / "fi_info", """
            #!/bin/bash
            echo "fi_info: error while loading shared libraries: libcudart.so.12: cannot open shared object file: No such file or directory" >&2
            exit 127
        """)
        _, rows = self.sweep("fi")
        inv = [r for r in rows if r["cell"] == "inventory"]
        self.assertTrue(inv)
        r = inv[0]
        self.assertEqual(r["status"], "failed",
                         "a 127 was reported as a provider measurement")
        self.assertEqual(r["exit"], "127")
        self.assertIn("libcudart.so.12", r["detail"],
                      f"missing library not named in detail: {r['detail']}")

    def test_genuine_zero_providers_still_reports_failed(self):
        """A clean exit that genuinely finds nothing is also failed -- but for
        a different, honest reason."""
        fib = self.w / "shs-libfabric-install/bin"
        self._exe(fib / "fi_info", '#!/bin/bash\nexit 0\n')
        _, rows = self.sweep("fi")
        inv = [r for r in rows if r["cell"] == "inventory"][0]
        self.assertEqual(inv["status"], "failed")
        self.assertEqual(inv["exit"], "0")


class TestScaleAndSelection(SweepHarness):
    def test_rank_count_follows_node_count(self):
        _, r1 = self.sweep("osu", nodes="1")
        self.assertTrue(all(r["ranks"] == "4" for r in r1))
        shutil.rmtree(self.run, ignore_errors=True)
        _, r2 = self.sweep("osu", nodes="2")
        self.assertTrue(all(r["ranks"] == "8" for r in r2))

    def test_layer_selection_excludes_others(self):
        _, rows = self.sweep("fi")
        self.assertEqual({r["layer"] for r in rows}, {"fi"})

    def test_missing_binaries_are_unavailable_with_reason(self):
        shutil.rmtree(self.w / "bench_src/nccl-tests/build")
        _, rows = self.sweep("nccl")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["status"], "unavailable")
        self.assertTrue(rows[0]["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
