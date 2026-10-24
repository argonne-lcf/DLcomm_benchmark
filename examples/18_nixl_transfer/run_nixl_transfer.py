#!/usr/bin/env python3
"""Run the NIXL transfer layer (h2d / d2h / d2d) under MPI.

One rank per device. Each rank measures the three directions through NIXL and
prints machine-readable lines the jobscript verifies:

    NIXL_MAP <host> rank <global_rank> dev <device_index>
    NIXL_RESULT <direction> <backend> <bytes> <best_s> <gbytes_per_s>

Backend selection is optional everywhere: with no --backend the module
resolves one from the plugins the local NIXL build registered. That is
deliberate -- hardcoding a vendor backend string is what made the torchcomms
layer unrunnable outside the machine it was written on.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bytes", type=int, default=1 << 26,
                   help="buffer size per rank per copy (default 64 MiB)")
    p.add_argument("--iterations", type=int, default=10,
                   help="timed iterations (best-of is reported)")
    p.add_argument("--warmup", type=int, default=3,
                   help="untimed warmup iterations")
    p.add_argument("--directions", default="h2d,d2h,d2d",
                   help="comma-separated subset of h2d,d2h,d2d")
    p.add_argument("--backend", default=None,
                   help="NIXL backend (default: auto-resolve from registered "
                        "plugins, preferring UCX then GDS then POSIX)")
    p.add_argument("--backend-preference", default="UCX,GDS,POSIX",
                   help="auto-resolution order when --backend is omitted")
    p.add_argument("--backend-env", default="", metavar="K=V,K=V",
                   help="env vars applied before agent creation, e.g. "
                        "UCX_IB_GDA_RETAIN_INACTIVE_CTX=yes")
    p.add_argument("--no-barrier", action="store_true",
                   help="skip the pre-iteration barrier (single-rank runs)")
    return p.parse_args(argv)


def _kv(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if "=" not in item:
            raise SystemExit(f"--backend-env entry {item!r} is not K=V")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    import torch

    # Rank discovery: MPI first, then torchrun-style env, then single rank.
    dist = None
    rank, world = 0, 1
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, world = comm.Get_rank(), comm.Get_size()

        class _MPIDist:
            def barrier(self) -> None:
                comm.Barrier()

        dist = None if args.no_barrier else _MPIDist()
    except Exception:
        rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", 0)))
        world = int(os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))

    if not torch.cuda.is_available():
        print("NIXL_SKIP no CUDA device visible", flush=True)
        return 1

    local_rank = int(os.environ.get("PALS_LOCAL_RANKID",
                     os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    dev_idx = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(dev_idx)
    device = torch.device("cuda", dev_idx)

    # Printed before any measurement so a device collapse is visible even if
    # the run later fails.
    print(f"NIXL_MAP {socket.gethostname()} rank {rank} dev {dev_idx}", flush=True)

    # Touch the device so a primary CUDA context exists before NIXL builds its
    # backend; UCX probes for one at agent creation.
    torch.zeros(1, device=device)
    torch.cuda.synchronize()

    from dl_comm.transfer import nixl_transfer as nt

    if not nt.available():
        print("NIXL_SKIP nixl package not importable", flush=True)
        return 1

    directions = tuple(d.strip() for d in args.directions.split(",") if d.strip())
    preference = tuple(p.strip() for p in args.backend_preference.split(",") if p.strip())

    try:
        results = nt.measure_nixl(
            torch,
            device,
            nbytes=args.bytes,
            iterations=args.iterations,
            warmup=args.warmup,
            dist=dist,
            world_size=world,
            directions=directions,
            backend=args.backend,
            backend_preference=preference,
            backend_env=_kv(args.backend_env) or None,
        )
    except nt.NixlUnavailable as exc:
        print(f"NIXL_SKIP {exc}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    for r in results:
        print(
            f"NIXL_RESULT {r.direction} {r.backend} {r.nbytes} "
            f"{r.best_s:.9f} {r.gbytes_per_s:.4f}",
            flush=True,
        )

    if rank == 0:
        print(nt.format_nixl_table(results), flush=True)

    # NIXL 1.5.0 segfaults when its agent is destroyed during interpreter
    # shutdown. Reproduced with no torch and no CUDA in the process:
    #   python -c "from nixl._api import nixl_agent, nixl_agent_config; \
    #              a=nixl_agent('x', nixl_agent_config(backends=['UCX'])); del a"
    #   -> exit 139, after the script body completed.
    # Every measurement above is already printed and flushed, so the crash
    # corrupts nothing -- but it turns a successful run into exit 139 and under
    # mpiexec it takes the whole job down with "died from signal 11". Leaving
    # the process via os._exit skips the C++ static destructors that fault.
    # Remove once the upstream teardown bug is fixed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
