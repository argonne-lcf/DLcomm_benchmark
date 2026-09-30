#!/usr/bin/env python3
"""torch.distributed allreduce over whatever transport NCCL selects.

Prints one machine-readable line the harness parses:

    RESULT usec=<mean> gbps=<busbw> transport=<see NCCL log>

Bus bandwidth uses the same formula as nccl-tests so the layers are
directly comparable:  busbw = (2 * (n-1) / n) * bytes / time.

Validation: every rank checks the reduced tensor against the analytic
expected value. A wrong answer must never be reported as a fast one --
NCCL allreduce on this machine is known to return silently wrong data at
1 MiB over CXI, so the check is not optional.
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    def env_first(*names, default=0):
        for n in names:
            if os.environ.get(n) not in (None, ""):
                return int(os.environ[n])
        return int(default)

    # PALS first: it is the launcher in use. PMI_*/RANK are fallbacks for
    # other launchers. Getting this order wrong makes every rank think it is
    # rank 0, which does not error -- it silently produces a wrong answer.
    rank = env_first("PALS_RANKID", "PMI_RANK", "RANK")
    world = env_first("PALS_NRANKS", "PMI_SIZE", "WORLD_SIZE", default=1)
    local = env_first("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "LOCAL_RANK")

    if world < 2:
        print(f"RESULT usec=0 gbps=INVALID  # world={world}, rank env missing",
              flush=True)
        sys.exit(2)

    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    n = a.bytes // 4
    # Each rank contributes (rank+1); the sum is a known constant, so a
    # corrupted reduction is detectable rather than merely suspicious.
    x = torch.full((n,), float(rank + 1), dtype=torch.float32, device="cuda")
    expected = float(world * (world + 1) // 2)

    # One buffer, allocated once -- same as nccl-tests, which reuses its
    # send/recv buffers across iterations.
    y = x.clone()

    for _ in range(a.warmup):
        dist.all_reduce(y)
    torch.cuda.synchronize()

    # Time ONLY the collectives. Per-iteration events so a single stall is
    # visible instead of being averaged away.
    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True)) for _ in range(a.iters)]
    for st, en in evs:
        st.record()
        dist.all_reduce(y)
        en.record()
    torch.cuda.synchronize()

    times = [st.elapsed_time(en) * 1000.0 for st, en in evs]
    times.sort()
    usec = sum(times) / len(times)
    med = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]

    # Re-seed for the correctness check: after N in-place allreduces the
    # content is world^N-ish, so validate a FRESH single reduction instead.
    y.fill_(float(rank + 1))
    dist.all_reduce(y)

    bad = int((y != expected).sum().item())
    worst = float((y - expected).abs().max().item())
    flags = torch.tensor([bad], device="cuda")
    dist.all_reduce(flags)
    bad_total = int(flags.item())

    if rank == 0:
        busbw = (2.0 * (world - 1) / world) * a.bytes / (usec * 1e-6) / 1e9
        if bad_total:
            print(f"VALIDATION_FAILED wrong={bad_total} worst_abs_err={worst}")
            print(f"RESULT usec={usec:.2f} gbps=INVALID")
            dist.destroy_process_group()
            sys.exit(3)
        print(f"RESULT usec={usec:.2f} gbps={busbw:.2f} "
              f"median_usec={med:.2f} p90_usec={p90:.2f}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
