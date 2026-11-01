#!/usr/bin/env python3
"""NIXL remote PUT/GET benchmark exercising the full memory-registration
lifecycle, GPU to GPU, intra-node and inter-node.

This is the NIXL-native layer requested as a feature: rather than treating
NIXL as a black-box bandwidth number, it walks the actual API sequence a KV
transfer performs and times each phase separately.

    registerMem      agent.register_memory(buf)
    describe         agent.get_xfer_descs(...)        local and remote
    exchange meta    agent.get_agent_metadata() / add_remote_agent()
    create xfer      agent.initialize_xfer(op, ...)
    post xfer        agent.transfer(handle)
    wait completion  agent.check_xfer_state(handle) until DONE
    deregisterMem    agent.release_xfer_handle() / deregister_memory()

Both directions are covered: READ (initiator pulls from the target = RDMA
read) and WRITE (initiator pushes to the target = RDMA write).

Why phase timing matters: registration is a one-off cost that pins pages and
exchanges rkeys, and at small buffer sizes it dominates the transfer itself by
orders of magnitude. A benchmark that folds registration into the transfer
time reports a "slow fabric" when what it actually measured was page pinning.
Real serving systems register once and transfer many times, so registration is
reported separately rather than amortised silently.

Placement is explicit: rank 0 and rank 1 each bind to a distinct GPU. On one
node that is GPU0 -> GPU1 on the same host (traffic stays on NVLink/PCIe and
never reaches a NIC). On two nodes it is GPU0@hostA -> GPU0@hostB (traffic
crosses Slingshot). The two cases are labelled so they are never conflated.

Peer discovery uses a file rendezvous under --sync-dir, so this is immune to
the PALS rank-ordering trap (rank 0 is not assumed to be on any particular
host).
"""

import argparse
import json
import os
import socket
import sys
import time


def log(msg):
    print(f"[nixl_putget] {msg}", flush=True)


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.0f}{unit}"
        n /= 1024.0
    return f"{n}B"


def size_label(nbytes):
    if nbytes >= 1 << 30:
        return f"{nbytes >> 30}GiB"
    if nbytes >= 1 << 20:
        return f"{nbytes >> 20}MiB"
    if nbytes >= 1 << 10:
        return f"{nbytes >> 10}KiB"
    return f"{nbytes}B"


def publish(sync_dir, name, data):
    tmp = os.path.join(sync_dir, f".{name}.tmp")
    final = os.path.join(sync_dir, name)
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(tmp, mode) as fh:
        fh.write(data)
    os.replace(tmp, final)


def collect(sync_dir, name, timeout=180):
    path = os.path.join(sync_dir, name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return fh.read()
        time.sleep(0.05)
    raise TimeoutError(f"peer file {name} did not appear within {timeout}s")


def barrier(sync_dir, rank, tag, peers=2, timeout=180):
    publish(sync_dir, f"bar.{tag}.{rank}", b"1")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(os.path.exists(os.path.join(sync_dir, f"bar.{tag}.{r}"))
               for r in range(peers)):
            return
        time.sleep(0.05)
    raise TimeoutError(f"barrier {tag} timed out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="LIBFABRIC")
    ap.add_argument("--mem", choices=("VRAM", "DRAM"), default="VRAM",
                    help="VRAM = GPU buffers (cuda), DRAM = pinned host "
                         "buffers. DRAM isolates the NIC path from GPU "
                         "registration: if VRAM fails and DRAM passes, the "
                         "fault is in GPU memory registration (dmabuf/IOMMU), "
                         "not the fabric.")
    ap.add_argument("--op", choices=("READ", "WRITE"), default="READ",
                    help="READ = initiator pulls (RDMA read); "
                         "WRITE = initiator pushes (RDMA write)")
    ap.add_argument("--sizes", default="4096,65536,1048576,268435456",
                    help="comma-separated byte sizes; defaults span a small "
                         "control buffer to a large bulk buffer")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sync-dir", required=True)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--expect", choices=("same-node", "cross-node", "any"),
                    default="any",
                    help="assert the placement actually achieved, so an "
                         "intra-node run cannot be reported as cross-node")
    args = ap.parse_args()

    rank = int(os.environ.get("PALS_RANKID", os.environ.get("RANK", "0")))
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID",
                                    os.environ.get("LOCAL_RANK", "0")))
    host = socket.gethostname().split(".")[0]
    is_initiator = (rank == 0)
    role = "initiator" if is_initiator else "target"

    os.makedirs(args.sync_dir, exist_ok=True)

    # --- placement: publish, then verify it is what was asked for ----------
    publish(args.sync_dir, f"host.{rank}", host)
    barrier(args.sync_dir, rank, "hosts")
    peer_rank = 1 - rank
    peer_host = collect(args.sync_dir, f"host.{peer_rank}").decode().strip()
    same_node = (peer_host == host)
    placement = "same-node" if same_node else "cross-node"

    import torch
    dev_count = torch.cuda.device_count()
    # One node: put the two ranks on DIFFERENT GPUs so this is a real
    # GPU-to-GPU transfer rather than a device talking to itself.
    gpu = local_rank % max(dev_count, 1)
    torch.cuda.set_device(gpu)
    gpu_name = torch.cuda.get_device_name(gpu)

    log(f"{role} rank={rank} host={host} gpu={gpu} ({gpu_name}) "
        f"peer_host={peer_host} placement={placement}")

    if args.expect != "any" and args.expect != placement:
        sys.exit(f"ERROR: expected {args.expect} placement but got {placement} "
                 f"(host={host} peer={peer_host}). Refusing to report a "
                 f"{placement} result as {args.expect}.")

    if same_node and gpu == (peer_rank % max(dev_count, 1)):
        sys.exit("ERROR: both ranks landed on the same GPU; that measures a "
                 "device copying to itself, not a GPU-to-GPU transfer.")

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except Exception as exc:
        sys.exit(f"ERROR: cannot import nixl ({exc})")

    agent = nixl_agent(f"{role}-r{rank}", nixl_agent_config(backends=[args.backend]))
    log(f"agent up, backend={args.backend} mem={args.mem}")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    results = []

    for nbytes in sizes:
        tag = size_label(nbytes)
        log(f"---- size {tag} ({nbytes} bytes) op={args.op} ----")

        # ---------------- registerMem ----------------
        if args.mem == "VRAM":
            buf = torch.zeros(nbytes, dtype=torch.uint8, device=f"cuda:{gpu}")
        else:
            # Pinned host memory: page-locked so the NIC can DMA it directly,
            # which is the comparable host-side path to a registered GPU buffer.
            buf = torch.zeros(nbytes, dtype=torch.uint8).pin_memory()
        if is_initiator and args.op == "WRITE":
            buf.fill_(0xAB)
        if (not is_initiator) and args.op == "READ":
            buf.fill_(0xAB)
        if args.mem == "VRAM":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        reg = agent.register_memory(buf)
        if args.mem == "VRAM":
            torch.cuda.synchronize()
        t_register = time.perf_counter() - t0
        if not reg:
            sys.exit("ERROR: register_memory failed")

        # ---------------- describe ----------------
        # get_xfer_descs takes 3-tuples (ptr, len, dev_id). The 4-tuple form
        # with a trailing metadata string belongs to get_reg_descs; passing it
        # here fails with "3-tuple list needed for transfer" (job 6954).
        #
        # mem_type spelling differs across builds: upstream examples use
        # "cuda"/"cpu", this build's agent.nixl_mems also accepts VRAM/DRAM.
        # Consult the agent rather than guessing.
        t0 = time.perf_counter()
        accepted = getattr(agent, "nixl_mems", None) or {}
        if args.mem == "VRAM":
            tuples = [(buf.data_ptr(), nbytes, gpu)]
            wanted = ("VRAM", "cuda")
        else:
            # host memory is not GPU-resident, so device id 0
            tuples = [(buf.data_ptr(), nbytes, 0)]
            wanted = ("DRAM", "cpu")
        mem_type = next((m for m in wanted if m in accepted), wanted[0])
        local_descs = agent.get_xfer_descs(tuples, mem_type=mem_type)
        t_describe = time.perf_counter() - t0
        if not local_descs:
            sys.exit(f"ERROR: get_xfer_descs returned nothing "
                     f"(mem_type={mem_type}, accepted={sorted(accepted)})")

        # ---------------- exchange metadata ----------------
        t0 = time.perf_counter()
        publish(args.sync_dir, f"meta.{tag}.{rank}", agent.get_agent_metadata())
        publish(args.sync_dir, f"descs.{tag}.{rank}", agent.get_serialized_descs(local_descs))
        barrier(args.sync_dir, rank, f"meta.{tag}")
        peer_meta = collect(args.sync_dir, f"meta.{tag}.{peer_rank}")
        peer_name = agent.add_remote_agent(peer_meta)
        remote_descs = agent.deserialize_descs(
            collect(args.sync_dir, f"descs.{tag}.{peer_rank}"))
        t_exchange = time.perf_counter() - t0
        log(f"metadata exchanged with {peer_name}, mem_type={mem_type}")

        row = dict(bytes=nbytes, label=tag, op=args.op, placement=placement,
                   register_s=t_register, describe_s=t_describe,
                   exchange_s=t_exchange)

        if is_initiator:
            # ---------------- create transfer ----------------
            t0 = time.perf_counter()
            handle = agent.initialize_xfer(args.op, local_descs, remote_descs,
                                           peer_name, b"done")
            t_create = time.perf_counter() - t0
            if not handle:
                sys.exit("ERROR: initialize_xfer returned no handle")

            def one_transfer():
                if agent.transfer(handle) == "ERR":
                    sys.exit("ERROR: transfer() returned ERR")
                while True:
                    st = agent.check_xfer_state(handle)
                    if st == "DONE":
                        return
                    if st == "ERR":
                        sys.exit("ERROR: transfer entered ERR state")

            for _ in range(args.warmup):
                one_transfer()

            # ---------------- post + wait, timed ----------------
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                one_transfer()
                samples.append(time.perf_counter() - t0)

            best = min(samples)
            mean = sum(samples) / len(samples)
            row.update(create_s=t_create,
                       best_s=best, mean_s=mean,
                       best_gbps=nbytes / best / 1e9,
                       mean_gbps=nbytes / mean / 1e9,
                       best_us=best * 1e6, mean_us=mean * 1e6)

            log(f"register {t_register*1e3:8.3f} ms | describe {t_describe*1e3:7.3f} ms | "
                f"exchange {t_exchange*1e3:7.3f} ms | create {t_create*1e3:7.3f} ms")
            log(f"transfer best {best*1e6:10.1f} us -> {nbytes/best/1e9:7.3f} GB/s | "
                f"mean {mean*1e6:10.1f} us -> {nbytes/mean/1e9:7.3f} GB/s")

            agent.release_xfer_handle(handle)

        barrier(args.sync_dir, rank, f"xfer.{tag}")

        # ---------------- correctness ----------------
        # The side that RECEIVES data checks it. READ: initiator pulled from
        # target, so initiator checks. WRITE: initiator pushed, target checks.
        checker = is_initiator if args.op == "READ" else (not is_initiator)
        if checker:
            if args.mem == "VRAM":
                torch.cuda.synchronize()
            _dev = f"cuda:{gpu}" if args.mem == "VRAM" else "cpu"
            expected = torch.full((min(nbytes, 1 << 20),), 0xAB,
                                  dtype=torch.uint8, device=_dev)
            got = buf[:min(nbytes, 1 << 20)]
            ok = bool(torch.equal(got, expected))
            row["byte_exact"] = ok
            log(f"byte-exact check: {'PASS' if ok else 'FAIL'}")
            if not ok:
                nz = int((got != 0xAB).sum().item())
                log(f"  mismatching bytes in first MiB: {nz}")

        # ---------------- deregisterMem ----------------
        t0 = time.perf_counter()
        agent.deregister_memory(reg)
        t_dereg = time.perf_counter() - t0
        row["deregister_s"] = t_dereg
        log(f"deregistered in {t_dereg*1e3:.3f} ms")

        results.append(row)
        del buf
        torch.cuda.empty_cache()
        barrier(args.sync_dir, rank, f"done.{tag}")

    if is_initiator and args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(dict(layer="nixl_putget", op=args.op,
                           placement=placement, backend=args.backend,
                           host=host, peer_host=peer_host, rows=results),
                      fh, indent=1)

    log(f"{role} complete ({placement}, op={args.op})")


if __name__ == "__main__":
    sys.exit(main())
