#!/usr/bin/env python3
"""torch.distributed collectives over NCCL, one collective at a time.

Reports algorithm bandwidth and bus bandwidth separately, because for most
collectives they differ by a factor that depends on the collective and the
rank count -- quoting algbw for an allreduce understates the wire traffic by
nearly 2x at large scale.

Bus-bandwidth factors (NCCL's own definitions, so the numbers are comparable
to nccl-tests rather than merely internally consistent):

    allreduce      2(n-1)/n     each byte is sent and received ~twice
    allgather      (n-1)/n      each rank contributes 1/n of the result
    reducescatter  (n-1)/n
    alltoall       (n-1)/n
    broadcast      1            one sender, tree/ring fanout
    reduce         1

Timing uses CUDA events around a loop, with the stream synchronised before the
first event and after the last, so host-side launch overhead is not counted as
wire time.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist


def bus_factor(coll: str, n: int) -> float:
    if n <= 1:
        return 1.0
    if coll == "allreduce":
        return 2.0 * (n - 1) / n
    if coll in ("allgather", "reducescatter", "alltoall"):
        return (n - 1) / n
    return 1.0


DTYPES = {
    "float32": (torch.float32, 4),
    "float16": (torch.float16, 2),
    "bfloat16": (torch.bfloat16, 2),
}


def make_buffers(coll: str, nbytes: int, n: int, dev, dtype_name="float32"):
    """Allocate per-collective input/output. Sizes follow nccl-tests semantics:
    nbytes is the SIZE OF THE RESULT the collective produces per rank, so
    allgather/reducescatter inputs are 1/n of it."""
    el, esz = DTYPES[dtype_name]
    if coll == "barrier":
        return None, None
    if coll == "send_recv":
        cnt = max(nbytes // esz, 1)
        return torch.ones(cnt, dtype=el, device=dev), torch.empty(cnt, dtype=el, device=dev)
    if coll in ("allreduce", "broadcast", "reduce"):
        cnt = max(nbytes // esz, 1)
        buf = torch.ones(cnt, dtype=el, device=dev)
        return buf, None
    if coll == "allgather":
        cnt = max(nbytes // esz // n, 1)
        inp = torch.ones(cnt, dtype=el, device=dev)
        out = torch.empty(cnt * n, dtype=el, device=dev)
        return inp, out
    if coll == "reducescatter":
        cnt = max(nbytes // esz // n, 1)
        inp = torch.ones(cnt * n, dtype=el, device=dev)
        out = torch.empty(cnt, dtype=el, device=dev)
        return inp, out
    if coll == "alltoall":
        cnt = max(nbytes // esz // n, 1)
        inp = torch.ones(cnt * n, dtype=el, device=dev)
        out = torch.empty(cnt * n, dtype=el, device=dev)
        return inp, out
    raise ValueError(coll)


def run_one(coll, inp, out, n, async_op=False, rank=0):
    """Issue one collective. With async_op=True the call returns a handle that
    is waited on immediately -- that measures the async submission path, which
    is what frameworks overlap compute against, not a different algorithm."""
    if coll == "barrier":
        dist.barrier()
        return
    if coll == "send_recv":
        # Ring pairing: even ranks send then receive, odd ranks the reverse,
        # so no pair deadlocks on a matched blocking send.
        peer = (rank + 1) % n
        prev = (rank - 1) % n
        if rank % 2 == 0:
            dist.send(inp, dst=peer); dist.recv(out, src=prev)
        else:
            dist.recv(out, src=prev); dist.send(inp, dst=peer)
        return
    if coll == "allreduce":
        h = dist.all_reduce(inp, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op:
            h.wait()
        return
    elif coll == "broadcast":
        dist.broadcast(inp, src=0)
    elif coll == "reduce":
        dist.reduce(inp, dst=0, op=dist.ReduceOp.SUM)
    elif coll == "allgather":
        dist.all_gather_into_tensor(out, inp)
    elif coll == "reducescatter":
        dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    elif coll == "alltoall":
        dist.all_to_all_single(out, inp)
    else:
        raise ValueError(coll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collective", required=True)
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--async-op", action="store_true",
                    help="issue the collective with async_op=True and wait on "
                         "the handle (the overlap path frameworks use)")
    ap.add_argument("--min-bytes", type=int, default=1 << 20)
    ap.add_argument("--max-bytes", type=int, default=1 << 28)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"[torch_dist] collective={args.collective} dtype={args.dtype} "
              f"async_op={args.async_op} world={world} "
              f"backend=nccl torch={torch.__version__} "
              f"nccl={'.'.join(map(str, torch.cuda.nccl.version()))}", flush=True)
        print(f"{'bytes':>12} {'iters':>6} {'ms/iter':>10} "
              f"{'algbw_GB/s':>11} {'busbw_GB/s':>11}", flush=True)

    factor = bus_factor(args.collective, world)
    rows = []
    # barrier moves no data. Sweeping message sizes for it would print a
    # bandwidth column that means nothing, so it gets one latency row.
    if args.collective == "barrier":
        for _ in range(args.warmup):
            dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            dist.barrier()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iters
        if rank == 0:
            print(f"{0:>12} {args.iters:>6} {ms:>10.4f} {0.0:>11.2f} {0.0:>11.2f}",
                  flush=True)
            rows.append(dict(bytes=0, ms=ms, algbw_gbps=0.0, busbw_gbps=0.0))
        if rank == 0 and args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(dict(collective="barrier", dtype=args.dtype,
                               world=world, rows=rows), fh, indent=2)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    nbytes = args.min_bytes
    while nbytes <= args.max_bytes:
        try:
            inp, out = make_buffers(args.collective, nbytes, world, dev,
                                    args.dtype)
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"{nbytes:>12} OOM -- stopping size sweep", flush=True)
            break

        for _ in range(args.warmup):
            run_one(args.collective, inp, out, world, args.async_op, rank)
        torch.cuda.synchronize()
        dist.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            run_one(args.collective, inp, out, world, args.async_op, rank)
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / args.iters
        algbw = nbytes / (ms / 1e3) / 1e9
        busbw = algbw * factor

        # every rank must agree the collective actually completed
        dist.barrier()

        if rank == 0:
            print(f"{nbytes:>12} {args.iters:>6} {ms:>10.4f} "
                  f"{algbw:>11.2f} {busbw:>11.2f}", flush=True)
            rows.append(dict(bytes=nbytes, ms=ms, algbw_gbps=algbw,
                             busbw_gbps=busbw))

        del inp, out
        torch.cuda.empty_cache()
        nbytes *= 4

    if rank == 0 and args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(dict(layer="torch_dist", collective=args.collective,
                           world=world, bus_factor=factor, rows=rows), fh, indent=1)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"[torch_dist] {args.collective} OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
