"""Run real ``torch.distributed`` collectives over gloo in spawned processes.

This harness is what makes the correctness tests meaningful: the collectives
genuinely execute across separate processes, so a verification routine that
cannot distinguish a working collective from a broken one is exposed.

A "broken" mode is supported: the worker skips the collective entirely (or
corrupts one rank's contribution) and the test asserts that verification
*fails*. A verifier that passes both the healthy and the broken run is vacuous.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _worker(rank, world_size, port, collective, op_name, num_elems, mode,
            dtype_name, return_queue):
    """One process of the gloo group."""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        from dl_comm.verify import build_payload, failures, scatter_source, choose_moduli
        from dl_comm.analysis.correctness import check_collective_correctness

        failures.reset()

        dtype = getattr(torch, dtype_name)
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "prod": dist.ReduceOp.PRODUCT,
        }
        op = op_map.get(op_name)

        x = build_payload(torch, num_elems, dtype, rank, world_size, op_name)

        if mode == "broken_noop":
            # The collective is simply not performed. Every rank keeps its own
            # input. This is the exact failure the all-ones payload could not
            # detect for 12 of 15 configurations.
            result = _shape_only_result(collective, x, world_size)
        elif mode == "broken_corrupt" and rank == world_size - 1:
            # Perform the collective, then corrupt the result on the last rank.
            result = _run(collective, x, op, world_size)
            x.add_(7.0)
            if torch.is_tensor(result):
                result.add_(7.0)
            elif isinstance(result, list):
                for t in result:
                    t.add_(7.0)
        else:
            result = _run(collective, x, op, world_size)

        context = {"mpi_rank": rank, "cfg": _Cfg(), "log": _Log(), "iteration": 0}
        check_collective_correctness(context, x, collective, op=op, group=None,
                                     result_data=result, group_type="Flatview",
                                     group_id="All")

        snap = failures.snapshot()
        return_queue.put({"rank": rank, **snap})
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        return_queue.put({"rank": rank, "error": traceback.format_exc()})


def _shape_only_result(collective, x, world_size):
    """Result object of the right shape for a collective that never ran."""
    if collective in ("allgather", "alltoall", "gather"):
        return [x.clone() for _ in range(world_size)]
    if collective in ("reducescatter",):
        return x[: x.numel() // world_size].clone()
    if collective in ("alltoallsingle",):
        return x.clone()
    return None


def _run(collective, x, op, world_size):
    """Execute the real collective via the shipped implementations."""
    from dl_comm.comm.collectives import COLLECTIVES, init_framework_constants

    init_framework_constants("pytorch")
    fn = COLLECTIVES[collective]
    return fn(x, op, group=None, dist=dist, framework="pytorch")


class _Cfg:
    framework = "pytorch"


class _Log:
    def output(self, msg=""):
        pass

    info = warning = error = output


def run_gloo(collective, op_name=None, world_size=4, num_elems=16,
             mode="healthy", dtype_name="float32", timeout=120):
    """Spawn ``world_size`` processes and return each rank's verification tally.

    Returns a list of dicts with keys ``rank``, ``checks``, ``failures``,
    ``skipped``, ``details`` -- or ``error`` if that rank raised.
    """
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker,
                        args=(rank, world_size, port, collective, op_name,
                              num_elems, mode, dtype_name, queue))
        p.start()
        procs.append(p)

    results = []
    for _ in range(world_size):
        try:
            results.append(queue.get(timeout=timeout))
        except Exception:
            break

    for p in procs:
        p.join(timeout=20)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    return results


def total_failures(results):
    return sum(r.get("failures", 0) for r in results)


def total_checks(results):
    return sum(r.get("checks", 0) for r in results)


def errors(results):
    return [r["error"] for r in results if "error" in r]
