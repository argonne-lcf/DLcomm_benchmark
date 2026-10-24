#!/usr/bin/env python3
"""Verify DLcomm's torchcomms backend moves real bytes on NVIDIA.

Prints, per rank::

    TORCHCOMMS_MAP <host> rank <r> dev <d> transport <t>
    TORCHCOMMS_VERIFIED rank <r> allreduce_sum_ok=<bool> latency_us=<f>

"It ran" is not evidence a collective worked -- a no-op transport also exits 0.
Each rank contributes its own rank number and the allreduce SUM must equal
world*(world-1)/2 in every element, which only holds if every rank's data
actually arrived.

Transport is NOT hardcoded: it comes from DLcomm's own ``resolve_transport``,
which reads the registered entry points. ``--transport`` overrides it.
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transport", default=None,
                    help="torchcomms transport (default: resolve from device)")
    ap.add_argument("--elements", type=int, default=1 << 20)
    ap.add_argument("--iterations", type=int, default=10)
    args = ap.parse_args(argv)

    import torch
    import torchcomms

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()

    # torchcomms reads these, exactly as torchrun would set them.
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID",
                     os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1))))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))

    if not torch.cuda.is_available():
        print("TORCHCOMMS_SKIP no CUDA device", flush=True)
        return 1

    dev_idx = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(dev_idx)
    device = torch.device("cuda", dev_idx)

    from dl_comm.comm import torchcomms_backend as tcb
    transport = args.transport or tcb.resolve_transport("gpu")

    print(f"TORCHCOMMS_MAP {socket.gethostname()} rank {rank} "
          f"dev {dev_idx} transport {transport}", flush=True)

    tc = torchcomms.new_comm(
        transport, device=device, name="dlcomm_check",
        timeout=datetime.timedelta(seconds=600),
    )

    # Every rank contributes its rank number; SUM must be world*(world-1)/2.
    expected = world * (world - 1) // 2
    t = torch.full((args.elements,), float(rank), device=device)
    # async_op is a REQUIRED positional arg in torchcomms, unlike torch.distributed.
    tc.all_reduce(t, torchcomms.ReduceOp.SUM, False)
    torch.cuda.synchronize()
    ok = bool(torch.all(t == float(expected)).item())

    lat = []
    for _ in range(args.iterations):
        t2 = torch.full((args.elements,), float(rank), device=device)
        tc.barrier(False)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        tc.all_reduce(t2, torchcomms.ReduceOp.SUM, False)
        torch.cuda.synchronize()
        lat.append((time.monotonic() - t0) * 1e6)

    print(f"TORCHCOMMS_VERIFIED rank {rank} allreduce_sum_ok={ok} "
          f"latency_us={min(lat):.2f}", flush=True)

    # Without explicit teardown the NCCL watchdog outlives the CUDA context and
    # every rank aborts with signal 6 AFTER printing correct results (exit 134).
    tc.barrier(False)
    torch.cuda.synchronize()
    try:
        tc.finalize()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
