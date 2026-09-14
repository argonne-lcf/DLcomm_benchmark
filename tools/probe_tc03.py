"""Capability matrix for torchcomms 0.3.0 on XPU.

Job 8824688 probed the module's bundled torchcomms 0.1.0 and found 13 of 14
ops raising "XCCL <op> is not supported now and will be added later". This
re-probes the locally built 0.3.0 against the same op list.

Every op is CALLED. hasattr is useless here: it returns True for ops whose
body is a throw, which is exactly how 0.1.0 failed.
"""

import faulthandler
import os
import time
import sys

# No mpi4py. Probe 8825011 showed the Abort(16) came from pals_env.sh
# clobbering PMI_RANK/PMI_SIZE, which corrupts MPICH's own bootstrap -- not
# from MPI being uninitialized. The working Aurora reference
# (pshukla/torchcomms_custom_torch/.../aurora_xccl_lifecycle_rank.py) reads
# the launcher variables directly and never imports mpi4py.

# Dump a C-level traceback on signal 11. Job 8824800 reported only
# "rank 9 died from signal 11" with no Python output at all, which gives no
# indication of which call faulted.
faulthandler.enable(all_threads=True)
import traceback

import torch
import torchcomms as tc


def env_int(*names, default=0):
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


rank = env_int("PALS_RANKID", "PMI_RANK", "RANK")
world = env_int("PMI_SIZE", "WORLD_SIZE", "PALS_SIZE", default=1)

# Fail loudly rather than segfaulting inside XCCL bootstrap. PALS sets
# PALS_RANKID but no global size variable at all (proved by job 8824786), so
# world stays 1 unless the pals_env.sh wrapper derives WORLD_SIZE from
# PALS_LOCAL_SIZE x node count. Without it, ranks segfaulted (8824754,
# 8824771) far from the actual cause.
if world < 2:
    raise SystemExit(
        f"rank {rank}: world size {world} -- no global size in the "
        "environment. Launch through pals_env.sh, which derives WORLD_SIZE "
        "from PALS_LOCAL_SIZE and DLCOMM_NNODES."
    )

os.environ.setdefault("RANK", str(rank))
os.environ.setdefault("WORLD_SIZE", str(world))
# No localhost default: a per-node MASTER_ADDR gives each node its own
# rendezvous and hangs instead of failing.
if "MASTER_ADDR" not in os.environ:
    raise SystemExit("MASTER_ADDR must be set by the caller for all ranks")
os.environ.setdefault("MASTER_PORT", "29517")

# Device selection under ZE_AFFINITY_MASK.
#
# The launcher masks each rank down to a single visible XPU and then exports
# LOCAL_RANK=0, so the correct device here is always xpu:0 and device_count()
# is 1. Assert that rather than assume it: if the mask is ever dropped, this
# fails loudly instead of silently co-locating every rank on device 0, which
# is the failure that invalidated five earlier jobs.
local_rank = int(os.environ.get("LOCAL_RANK",
                                os.environ.get("PALS_LOCAL_RANKID", "0")))
_ndev = torch.xpu.device_count()
_mask = os.environ.get("ZE_AFFINITY_MASK")
if _ndev == 0:
    raise SystemExit(f"rank {rank}: no XPU devices visible")
if _mask is not None and _ndev != 1:
    raise SystemExit(
        f"rank {rank}: ZE_AFFINITY_MASK={_mask} but device_count={_ndev}; "
        "expected exactly one visible device under the mask")
if not 0 <= local_rank < _ndev:
    raise SystemExit(
        f"rank {rank}: local_rank={local_rank} outside device_count={_ndev}")
dev = torch.device("xpu", local_rank)
torch.xpu.set_device(dev)
print(f"rank {rank}/{world}: local_rank={local_rank} mask={_mask} "
      f"visible={_ndev} device set to {dev}", flush=True)

from torch.distributed import TCPStore  # noqa: E402

# torchcomms 0.3.0 discovers rank/size itself via query_ranksize(), which
# reads PMI_RANK/PMI_SIZE under mpiexec on Aurora -- no store is required and
# every upstream example omits it:
#     new_comm("rcclx", device, name="main_comm")
# Passing an externally built TCPStore segfaulted all 12 ranks during
# bootstrap (job 8824754). Let the library create its own.
_use_store = os.environ.get("DLCOMM_TC_STORE", "0") == "1"
if _use_store:
    store = TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                     world, rank == 0)
    comm = tc.new_comm("xccl", dev, "dlcomm-probe", store=store)
else:
    print(f"rank {rank}: calling new_comm(xccl, {dev}) "
          f"visible_xpus={torch.xpu.device_count()} "
          f"mask={os.environ.get('ZE_AFFINITY_MASK', 'unset')} ...", flush=True)
    comm = tc.new_comm(backend="xccl", device=dev, name="dlcomm-probe")
print(f"rank {rank}: new_comm returned", flush=True)

N = 1024
results = {}


def probe(name, fn):
    """Call the op. Record the outcome, never the presence of an attribute."""
    try:
        fn()
        torch.xpu.synchronize()
        results[name] = ("yes", "")
        print(f"PROBE {name} yes", flush=True)
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:90]
        results[name] = ("NO ", msg)
        print(f"PROBE {name} NO {results[name][1]}", flush=True)


t = lambda: torch.ones(N, device=dev)          # noqa: E731
tbig = lambda: torch.ones(N * world, device=dev)  # noqa: E731

probe("all_reduce", lambda: comm.all_reduce(t(), tc.ReduceOp.SUM, False))
probe("barrier", lambda: comm.barrier(False))
probe("broadcast", lambda: comm.broadcast(t(), 0, False))
probe("reduce", lambda: comm.reduce(t(), 0, tc.ReduceOp.SUM, False))
probe("all_gather", lambda: comm.all_gather([t() for _ in range(world)], t(), False))
probe("all_gather_single", lambda: comm.all_gather_single(tbig(), t(), False))
probe("reduce_scatter_single",
      lambda: comm.reduce_scatter_single(t(), tbig(), tc.ReduceOp.SUM, False))
# all_to_all_single requires dim 0 divisible by world size. N=1024 leaves
# a remainder of 4 at 12 ranks and 16 at 24, so trim to a multiple.
_n_a2a = (N // world) * world
def _t_a2a():
    return torch.ones(_n_a2a, dtype=torch.float32, device=dev)
probe("all_to_all_single", lambda: comm.all_to_all_single(_t_a2a(), _t_a2a(), False))
probe("scatter",
      lambda: comm.scatter(t(), [t() for _ in range(world)] if rank == 0 else [], 0, False))
probe("gather",
      lambda: comm.gather([t() for _ in range(world)] if rank == 0 else [], t(), 0, False))


def _sendrecv():
    if world < 2:
        raise RuntimeError("needs >= 2 ranks")
    if rank == 0:
        comm.send(t(), 1, False)
        comm.recv(t(), 1, False)
    elif rank == 1:
        comm.recv(t(), 0, False)
        comm.send(t(), 0, False)


probe("send_recv", _sendrecv)


def _sendrecv_async():
    if world < 2:
        raise RuntimeError("needs >= 2 ranks")
    if rank == 0:
        w1 = comm.send(t(), 1, True)
        w2 = comm.recv(t(), 1, True)
    elif rank == 1:
        w2 = comm.recv(t(), 0, True)
        w1 = comm.send(t(), 0, True)
    else:
        return
    w1.wait()
    w2.wait()


probe("send_recv_async", _sendrecv_async)
if os.environ.get("DLCOMM_TC_SPLIT", "0") == "1":
    probe("split", lambda: comm.split([0], "probe-split"))

if rank == 0:
    import importlib.metadata as md
    print(f"TC_VERSION={md.version('torchcomms')}")
    print(f"TC_TORCH={torch.__version__}")
    print(f"TC_WORLD={world}")
    print("MATRIX")
    yes = 0
    for name, (state, msg) in results.items():
        print(f"[{state}] {name:<24} {msg}")
        if state == "yes":
            yes += 1
    print(f"TC_SUPPORTED={yes}/{len(results)}")
    stubs = [n for n, (s, m) in results.items()
             if "not supported now" in m]
    print(f"TC_STUB_OPS={len(stubs)} {stubs}")


# ---------------------------------------------------------------------------
# Timed sweep. Emits the record shape used by every other layer so the results
# are comparable. Only ops the matrix marked "yes" are measured.
# ---------------------------------------------------------------------------
import statistics  # noqa: E402

WARMUP, ITERS = 5, 20


def _busbw_factor(op, n):
    if n < 2:
        return 1.0
    if op == "all_reduce":
        return 2.0 * (n - 1) / n
    if op in ("all_gather_single", "reduce_scatter_single", "all_to_all_single"):
        return (n - 1) / n
    # broadcast and reduce are root-bottlenecked: nccl-tests uses 1.
    return 1.0


def _bench(op, call, nbytes, moved_multiplier=1):
    for _ in range(WARMUP):
        call()
    torch.xpu.synchronize()
    ts = []
    for _ in range(ITERS):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        call()
        torch.xpu.synchronize()
        ts.append(time.perf_counter() - t0)
    if rank == 0:
        t = statistics.median(ts)
        algbw = (nbytes * moved_multiplier) / t
        print(f"LAYER=torchcomms BACKEND=xccl OP={op} BYTES={nbytes} "
              f"RANKS={world} T_MED={t:.6g} ALGBW={algbw:.6g} "
              f"BUSBW={algbw * _busbw_factor(op, world):.6g}", flush=True)


if os.environ.get("DLCOMM_TC_BENCH", "1") == "1":
    for _mib in (1, 2, 4):
        _nb = _mib * 1024 * 1024
        _ne = _nb // 4
        _ne_div = (_ne // world) * world
        _x = torch.ones(_ne, dtype=torch.float32, device=dev)
        _xd = torch.ones(_ne_div, dtype=torch.float32, device=dev)
        _big = torch.ones(_ne * world, dtype=torch.float32, device=dev)
        if results.get("all_reduce", ("NO",))[0] == "yes":
            _bench("all_reduce", lambda: comm.all_reduce(_x, tc.ReduceOp.SUM, False), _nb)
        if results.get("all_gather_single", ("NO",))[0] == "yes":
            _bench("all_gather_single", lambda: comm.all_gather_single(_big, _x, False),
                   _nb, world)
        if results.get("reduce_scatter_single", ("NO",))[0] == "yes":
            _bench("reduce_scatter_single",
                   lambda: comm.reduce_scatter_single(_x, _big, tc.ReduceOp.SUM, False),
                   _nb, world)
        if results.get("all_to_all_single", ("NO",))[0] == "yes":
            _bench("all_to_all_single",
                   lambda: comm.all_to_all_single(_xd, _xd, False), _ne_div * 4)
        if results.get("broadcast", ("NO",))[0] == "yes":
            _bench("broadcast", lambda: comm.broadcast(_x, 0, False), _nb)
        if results.get("reduce", ("NO",))[0] == "yes":
            _bench("reduce", lambda: comm.reduce(_x, 0, tc.ReduceOp.SUM, False), _nb)

try:
    fin = getattr(comm, "finalize", None)
    if fin is not None:
        fin()
except Exception as e:  # noqa: BLE001
    print(f"rank {rank}: finalize failed: {e}", flush=True)

sys.stdout.flush()
