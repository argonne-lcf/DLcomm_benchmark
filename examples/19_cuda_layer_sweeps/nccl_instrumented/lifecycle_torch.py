#!/usr/bin/env python3
"""Lifecycle cost of a torch.distributed collective.

Phases, separated because they have different causes:

    import      `import torch` -- process startup, not communication, but
                it is real wall-clock cost every job pays
    init        dist.init_process_group: TCP rendezvous + NCCL bootstrap
    first_op    first all_reduce: lazy comm construction inside torch
    steady      median of N subsequent identical all_reduces
    destroy     dist.destroy_process_group

Device buffers, nccl-tests busbw formula, analytic correctness check --
matched to lifecycle_nccl.cu so the layers are directly comparable.
"""
import argparse
import json
import os
import sys
import time

T_PROC_START = time.monotonic()

import torch                                            # noqa: E402
import torch.distributed as dist                        # noqa: E402

T_IMPORT = (time.monotonic() - T_PROC_START) * 1e6


def env_first(*names, default=0):
    for n in names:
        if os.environ.get(n) not in (None, ""):
            return int(os.environ[n])
    return int(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=16777216)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    # PALS first: it is the launcher. PMI_*/RANK are fallbacks.
    rank = env_first("PALS_RANKID", "PMI_RANK", "RANK")
    world = env_first("PALS_NRANKS", "PMI_SIZE", "WORLD_SIZE", default=1)
    local = env_first("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "LOCAL_RANK")

    if world < 2:
        print("RESULT lib=torch error=world_size_1_rank_env_missing")
        return 2

    torch.cuda.set_device(local)

    dev = torch.device(f"cuda:{local}")
    t0 = time.monotonic()
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=dev)
    t_init = (time.monotonic() - t0) * 1e6

    n = a.bytes // 4
    y = torch.full((n,), float(rank + 1), dtype=torch.float32, device="cuda")
    expected = float(world * (world + 1) // 2)
    torch.cuda.synchronize()

    # First op: includes whatever torch builds lazily on first use.
    dist.barrier()
    torch.cuda.synchronize()
    t1 = time.monotonic()
    dist.all_reduce(y)
    torch.cuda.synchronize()
    t_first = (time.monotonic() - t1) * 1e6

    # Steady state. One buffer, allocated once -- no clone inside the timed
    # region, which would charge torch for a D2D copy nccl-tests never does.
    samples = []
    for _ in range(a.iters):
        dist.barrier()
        torch.cuda.synchronize()
        s = time.monotonic()
        dist.all_reduce(y)
        torch.cuda.synchronize()
        samples.append((time.monotonic() - s) * 1e6)
    samples.sort()
    steady = samples[len(samples) // 2]
    p90 = samples[int(len(samples) * 0.9)]

    # Validate a FRESH reduction: after N in-place allreduces the running
    # value is no longer the analytic constant.
    y.fill_(float(rank + 1))
    dist.all_reduce(y)
    torch.cuda.synchronize()
    bad = int((y != expected).sum().item())
    flags = torch.tensor([bad], device="cuda")
    dist.all_reduce(flags)
    bad_total = int(flags.item())

    t2 = time.monotonic()
    dist.destroy_process_group()
    t_destroy = (time.monotonic() - t2) * 1e6

    if rank == 0:
        if bad_total:
            print(f"RESULT lib=torch world={world} bytes={a.bytes} "
                  f"VALIDATION_FAILED wrong={bad_total}")
            return 3
        busbw = (2.0 * (world - 1) / world) * a.bytes / (steady * 1e-6) / 1e9
        print(f"RESULT lib=torch world={world} bytes={a.bytes} "
              f"import_us={T_IMPORT:.1f} init_us={t_init:.1f} "
              f"first_us={t_first:.1f} steady_us={steady:.1f} "
              f"p90_us={p90:.1f} destroy_us={t_destroy:.1f} gbps={busbw:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
