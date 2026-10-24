#!/usr/bin/env python3
"""Run the cross-node NIXL RDMA transfer layer under MPI.

Example 18 measures NIXL h2d/d2h/d2d between buffers on one node. This one
measures a rank writing directly into a *remote* rank's device memory -- the
transfer a disaggregated serving deployment actually performs when a prefill
node hands KV cache to a decode node. The interesting number is not the same
as 18's, and neither replaces the other.

Machine-readable output the jobscript verifies:

    NIXL_RMAP <host> rank <r> dev <d> peer <p> cross <bool>
    NIXL_REMOTE_RESULT <bytes> <backend> <best_s> <gbps> <med_gbps> <cross> <verified>

``cross`` and ``verified`` are in the result line rather than inferred,
because a same-node pairing and an unverified transfer both produce
plausible bandwidth numbers that must not be read as a fabric measurement.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bytes", type=int, default=1 << 26,
                   help="buffer size per rank (default 64 MiB)")
    p.add_argument("--sizes", default="",
                   help="comma-separated sizes to sweep; overrides --bytes. "
                        "A sweep is how you tell latency-bound from "
                        "bandwidth-bound: a flat GB/s curve is not bandwidth.")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--pairing", default="split", choices=("split", "ring"))
    p.add_argument("--ranks-per-node", type=int, default=None,
                   help="required for --pairing ring")
    # READ by default: on Slingshot/CXI, NIXL's LIBFABRIC WRITE path calls
    # fi_writedata(), which the cxi provider only implements when the
    # endpoint has the FI_RMA_EVENT capability. NIXL does not request it, so
    # WRITE fails with a bare NIXL_ERR_BACKEND. READ (fi_read) has no such
    # requirement. Both move the same bytes across the same fabric.
    p.add_argument("--op", choices=("READ", "WRITE"), default="READ",
                   help="READ = initiator pulls from peer (works on CXI); "
                        "WRITE = initiator pushes (needs FI_RMA_EVENT on CXI)")
    p.add_argument("--backend", default=None,
                   help="NIXL backend (default: resolve from registered plugins)")
    # LIBFABRIC first: on a Slingshot/CXI fabric it is the only registered
    # backend that can reach the NIC. UCX stays in the list as the fallback
    # for IB/RoCE sites, where it is the right choice. Ordering this
    # UCX-first (the earlier default) silently selects a TCP-over-IP path on
    # Slingshot and costs ~50x bandwidth while still passing every
    # correctness check -- a wrong answer that looks healthy.
    p.add_argument("--backend-preference", default="LIBFABRIC,UCX,GDS,POSIX")
    p.add_argument("--backend-env", default="", metavar="K=V,K=V",
                   help="env applied before agent creation; UCX reads its "
                        "config once at that point and ignores later changes")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the byte-level check (not recommended: a WRITE "
                        "to a wrong-but-valid address still reports DONE)")
    p.add_argument("--timeout", type=float, default=120.0)
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

    try:
        from mpi4py import MPI
    except Exception as exc:
        print(f"NIXL_REMOTE_SKIP mpi4py required for peer exchange: {exc}",
              flush=True)
        return 1

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    if world < 2:
        print("NIXL_REMOTE_SKIP cross-node transfer needs >= 2 ranks", flush=True)
        return 1
    if world % 2 != 0:
        print(f"NIXL_REMOTE_SKIP world_size {world} is odd; every rank needs "
              "a partner", flush=True)
        return 1

    if not torch.cuda.is_available():
        print("NIXL_REMOTE_SKIP no CUDA device visible", flush=True)
        return 1

    local_rank = int(os.environ.get(
        "PALS_LOCAL_RANKID",
        os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    dev_idx = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(dev_idx)
    device = torch.device("cuda", dev_idx)

    # Establish a primary CUDA context before NIXL builds its backend; UCX
    # probes for one at agent creation and logs "no active primary CUDA
    # context" if it is missing.
    torch.zeros(1, device=device)
    torch.cuda.synchronize()

    from dl_comm.transfer import nixl_remote as nr
    from dl_comm.transfer.nixl_remote import pair_ranks

    host = socket.gethostname()
    try:
        peer = pair_ranks(rank, world, pairing=args.pairing,
                          ranks_per_node=args.ranks_per_node)
    except ValueError as exc:
        print(f"NIXL_REMOTE_SKIP {exc}", flush=True)
        return 1

    hosts = comm.allgather(host)
    cross = hosts[rank] != hosts[peer]
    print(f"NIXL_RMAP {host} rank {rank} dev {dev_idx} peer {peer} cross {cross}",
          flush=True)

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = [args.bytes]

    results = []
    for nbytes in sizes:
        try:
            res = nr.measure_remote_nixl(
                torch,
                device,
                rank=rank,
                world_size=world,
                allgather=comm.allgather,
                barrier=comm.Barrier,
                hostname=host,
                nbytes=nbytes,
                iterations=args.iterations,
                warmup=args.warmup,
                pairing=args.pairing,
                ranks_per_node=args.ranks_per_node,
                backend=args.backend,
                op=args.op,
                backend_preference=tuple(
                    p.strip() for p in args.backend_preference.split(",") if p.strip()
                ),
                backend_env=_kv(args.backend_env) or None,
                timeout_s=args.timeout,
                verify=not args.no_verify,
            )
        except nr.NixlVerificationError as exc:
            # A delivered-wrong-bytes result is a hard failure, never a
            # slow-but-usable number.
            print(f"NIXL_REMOTE_CORRUPT {exc}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(2)
        except nr.NixlRemoteUnavailable as exc:
            print(f"NIXL_REMOTE_SKIP {exc}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

        results.append(res)
        print(
            f"NIXL_REMOTE_RESULT {res.nbytes} {res.backend} "
            f"{res.best_s:.9f} {res.gbytes_per_s:.4f} "
            f"{res.median_gbytes_per_s:.4f} {res.cross_node} {res.verified}",
            flush=True,
        )

    if rank == 0:
        print(nr.format_remote_table(results), flush=True)

    # NIXL 1.5.0 segfaults destroying its agent during interpreter shutdown
    # (reproduced with no torch and no CUDA in the process). Every result is
    # already printed and flushed; leaving via os._exit skips the C++ static
    # destructors that fault and prevents a successful run reporting exit 139.
    # Remove once the upstream teardown bug is fixed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
