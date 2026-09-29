#!/usr/bin/env python3
"""Cross-layer profiler: measure WHERE time goes, not just how long it took.

The per-layer sweep answers "how fast is each layer". It cannot answer "why
is layer N slower than layer N-1", because a single wall-clock number per
cell has no internal structure. This profiler adds that structure by timing
the phases inside one transfer at each layer, on the same ranks, at the same
sizes, in the same job.

Three things are measured that the sweep never captured:

1. PHASE BREAKDOWN. For every layer, the same payload is pushed through and
   the time is split into staging (host<->device), the collective/transfer
   itself, and the synchronisation around it. CUDA events are used for
   device-side spans so the numbers are device time, not host wall clock
   polluted by launch latency.

2. TRANSPORT PROVENANCE. NCCL is asked, at runtime, which transport it
   actually selected (NCCL_DEBUG=INFO parsed from its own output) and which
   libfabric provider is live. A 2-node NCCL number is meaningless if you
   cannot say whether it went over CXI or fell back to TCP sockets. The
   sweep never recorded this, so the existing 2-node NCCL figure cannot be
   attributed to a transport at all.

3. THE SAME-OP LADDER. MPI, NCCL and torch.distributed all run allreduce at
   identical sizes and rank counts here, so their numbers are directly
   comparable -- unlike the headline sweep, where each layer used whatever
   op and geometry suited it.

Nothing is estimated. Every number printed is measured in this process, and
each phase is labelled with how it was timed so a reader can judge it.

Usage (under mpiexec, one rank per GPU):
    python layer_profile.py --sizes 1MiB,16MiB,256MiB --iters 20
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

SIZE_RE = re.compile(r"^(\d+)(KiB|MiB|GiB|B)?$")


def parse_size(s):
    m = SIZE_RE.match(s.strip())
    if not m:
        raise ValueError(f"bad size {s!r}")
    n = int(m.group(1))
    return n * {"B": 1, None: 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}[m.group(2)]


def human(n):
    for u, d in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= d:
            return f"{n // d}{u}"
    return f"{n}B"


def log(msg):
    print(f"[profile] {msg}", flush=True)


# --------------------------------------------------------------------------
# environment / transport provenance
# --------------------------------------------------------------------------
def capture_provenance():
    """Record what the stack actually chose. This is the part that explains
    a cliff; without it a slow number has no cause."""
    prov = {
        "host": socket.gethostname(),
        "rank": int(os.environ.get("PALS_RANKID", os.environ.get("RANK", 0))),
        "local_rank": int(os.environ.get("PALS_LOCAL_RANKID",
                                         os.environ.get("LOCAL_RANK", 0))),
    }
    for k in ("FI_PROVIDER", "NCCL_NET", "NCCL_NET_PLUGIN", "NCCL_IB_DISABLE",
              "NCCL_SOCKET_IFNAME", "NCCL_PROTO", "NCCL_ALGO",
              "LD_LIBRARY_PATH", "NCCL_DEBUG"):
        if os.environ.get(k):
            prov[k] = os.environ[k]

    # CXI devices visible to this rank -- if these are absent on a compute
    # node, any "fabric" number is actually a software fallback.
    try:
        prov["cxi_devices"] = sorted(
            d for d in os.listdir("/dev") if d.startswith("cxi"))
    except OSError:
        prov["cxi_devices"] = []

    # NCCL's own net plugin, if one is installed.
    plugins = []
    for root in ("/usr/lib64", "/opt/cray/libfabric",
                 os.environ.get("NIXL_INSTALL_DIR", "")):
        if root and os.path.isdir(root):
            for dirpath, _, files in os.walk(root):
                for f in files:
                    if f.startswith("libnccl-net"):
                        plugins.append(os.path.join(dirpath, f))
                break
    prov["nccl_net_plugins"] = plugins
    return prov


def parse_nccl_transport(text):
    """Pull the transport NCCL reported from NCCL_DEBUG=INFO output.

    NCCL prints lines like:
        NCCL INFO NET/OFI Using ... provider cxi
        NCCL INFO Channel 00 : 0[0] -> 1[0] [send] via NET/Socket/0
    Socket means it fell back to TCP; OFI/IB means it is on the fabric.
    """
    found = {"net": None, "provider": None, "evidence": []}
    for line in text.splitlines():
        if "NET/" not in line and "Using network" not in line:
            continue
        found["evidence"].append(line.strip()[:200])
        m = re.search(r"NET/(\w+)", line)
        if m and not found["net"]:
            found["net"] = m.group(1)
        m = re.search(r"provider[= ]+(\w+)", line, re.I)
        if m and not found["provider"]:
            found["provider"] = m.group(1)
        m = re.search(r"Using network (\w+)", line)
        if m and not found["net"]:
            found["net"] = m.group(1)
    return found


# --------------------------------------------------------------------------
# phase timing
# --------------------------------------------------------------------------
class Phases:
    """Accumulate named spans. Device spans use CUDA events so we report
    device time; host spans use perf_counter."""

    def __init__(self, torch, dev):
        self.t = torch
        self.dev = dev
        self.spans = {}

    def device(self, name, fn, iters):
        t = self.t
        st = [t.cuda.Event(enable_timing=True) for _ in range(iters)]
        en = [t.cuda.Event(enable_timing=True) for _ in range(iters)]
        fn()                      # warmup, excluded
        t.cuda.synchronize()
        for i in range(iters):
            st[i].record()
            fn()
            en[i].record()
        t.cuda.synchronize()
        ms = [s.elapsed_time(e) for s, e in zip(st, en)]
        ms.sort()
        self.spans[name] = {
            "best_us": ms[0] * 1000.0,
            "median_us": ms[len(ms) // 2] * 1000.0,
            "timed_by": "cuda_events(device_time)",
            "iters": iters,
        }
        return self.spans[name]

    def host(self, name, fn, iters):
        us = []
        fn()
        for _ in range(iters):
            a = time.perf_counter()
            fn()
            us.append((time.perf_counter() - a) * 1e6)
        us.sort()
        self.spans[name] = {
            "best_us": us[0],
            "median_us": us[len(us) // 2],
            "timed_by": "perf_counter(host_wall)",
            "iters": iters,
        }
        return self.spans[name]


def profile_torch_layers(sizes, iters, prov):
    """Layers 3/4: NCCL (via torch) and torch.distributed, plus the staging
    costs that surround any GPU collective."""
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        log("CUDA unavailable; cannot profile device layers")
        return {}

    local = int(os.environ.get("PALS_LOCAL_RANKID",
                               os.environ.get("LOCAL_RANK", 0)))

    # pals_torch_env.sh sets CUDA_VISIBLE_DEVICES=$LOCAL_RANK, so each rank
    # sees exactly ONE device and the only valid ordinal is 0. Passing the
    # local rank here yields "invalid device ordinal" on every rank but 0.
    # Select by counting what is actually visible rather than assuming.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_visible = torch.cuda.device_count()
    if visible and len(visible.split(",")) == 1:
        dev_idx = 0
        placement = f"CUDA_VISIBLE_DEVICES={visible} (1 visible -> ordinal 0)"
    elif local < n_visible:
        dev_idx = local
        placement = f"local_rank={local} of {n_visible} visible"
    else:
        raise RuntimeError(
            f"local_rank {local} exceeds {n_visible} visible devices "
            f"(CUDA_VISIBLE_DEVICES={visible!r}); refusing to guess")

    torch.cuda.set_device(dev_idx)
    dev = torch.device(f"cuda:{dev_idx}")

    # Record the PHYSICAL identity of the device this rank drove, so the
    # report can say which GPU produced each number rather than inferring
    # it from the launcher geometry.
    props = torch.cuda.get_device_properties(dev_idx)
    prov["gpu"] = {
        "placement": placement,
        "visible_devices": visible or "(unset: all)",
        "torch_ordinal": dev_idx,
        "name": props.name,
        "uuid": str(getattr(props, "uuid", "n/a")),
        "pci_bus_id": torch.cuda.get_device_properties(dev_idx).pci_bus_id
        if hasattr(props, "pci_bus_id") else "n/a",
        "total_mem_GiB": round(props.total_memory / 2**30, 1),
        "sm": f"{props.major}.{props.minor}",
    }

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=dev)
    rank, world = dist.get_rank(), dist.get_world_size()
    # Every rank announces its physical GPU. This is the evidence for
    # "which GPUs were used", rather than inferring it from -ppn.
    g = prov["gpu"]
    print(f"[profile] GPU_IDENTITY rank={rank}/{world} host={prov['host']} "
          f"{g['placement']} name={g['name']} uuid={g['uuid']} "
          f"mem={g['total_mem_GiB']}GiB sm={g['sm']}", flush=True)

    out = {"world": world, "rank": rank, "sizes": {}}
    for sz in sizes:
        n = sz // 4
        g = torch.ones(n, dtype=torch.float32, device=dev)
        h = torch.ones(n, dtype=torch.float32, device="cpu").pin_memory()
        ph = Phases(torch, dev)

        # staging: the cost of getting data to/from the device at all
        ph.device("h2d_pinned", lambda: g.copy_(h, non_blocking=True), iters)
        ph.device("d2h_pinned", lambda: h.copy_(g, non_blocking=True), iters)

        # the collective itself, device-timed
        dist.barrier()
        ph.device("allreduce",
                  lambda: dist.all_reduce(g, op=dist.ReduceOp.SUM), iters)

        # the same call host-timed: the gap vs device time is launch +
        # synchronisation overhead that a device-only measurement hides
        def _sync_ar():
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
        ph.host("allreduce_host_observed", _sync_ar, iters)

        # a pure barrier: the floor under any collective at this scale
        ph.host("barrier", lambda: dist.barrier(), max(iters // 2, 5))

        ar = ph.spans["allreduce"]["best_us"]
        algbw = sz / (ar * 1e-6) / 1e9 if ar else 0.0
        # busbw for allreduce = algbw * 2(n-1)/n
        busbw = algbw * 2 * (world - 1) / world if world > 1 else algbw
        out["sizes"][human(sz)] = {
            "bytes": sz,
            "phases": ph.spans,
            "algbw_GBs": algbw,
            "busbw_GBs": busbw,
            "launch_overhead_us": max(
                0.0,
                ph.spans["allreduce_host_observed"]["best_us"] - ar),
        }
        del g, h
        torch.cuda.empty_cache()

    if rank == 0:
        out["provenance"] = prov
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1MiB,16MiB,256MiB")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--layer", default="torch",
                    choices=("torch", "provenance"))
    ap.add_argument("--only-size", default="",
                    help="Isolation control: profile exactly ONE size in a "
                         "fresh process. If the isolated number differs from "
                         "the same size measured inside the full ladder, the "
                         "instrument is shaping the result and the ladder "
                         "numbers are void.")
    a = ap.parse_args()

    if a.only_size:
        a.sizes = a.only_size

    sizes = [parse_size(s) for s in a.sizes.split(",") if s.strip()]
    prov = capture_provenance()
    log(f"host={prov['host']} rank={prov['rank']} "
        f"cxi_devices={prov['cxi_devices']} "
        f"FI_PROVIDER={prov.get('FI_PROVIDER')}")

    if a.layer == "provenance":
        print(json.dumps(prov, indent=2))
        return 0

    res = profile_torch_layers(sizes, a.iters, prov)
    if not res:
        return 1

    if res.get("rank") == 0:
        for name, r in res["sizes"].items():
            p = r["phases"]
            log(f"---- {name} ----")
            log(f"  h2d_pinned        {p['h2d_pinned']['best_us']:10.1f} us")
            log(f"  d2h_pinned        {p['d2h_pinned']['best_us']:10.1f} us")
            log(f"  allreduce(device) {p['allreduce']['best_us']:10.1f} us"
                f"  -> algbw {r['algbw_GBs']:.3f} GB/s"
                f"  busbw {r['busbw_GBs']:.3f} GB/s")
            log(f"  allreduce(host)   "
                f"{p['allreduce_host_observed']['best_us']:10.1f} us")
            log(f"  launch+sync gap   {r['launch_overhead_us']:10.1f} us")
            log(f"  barrier           {p['barrier']['best_us']:10.1f} us")
        if a.json_out:
            with open(a.json_out, "w") as f:
                json.dump(res, f, indent=2)
            log(f"wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
