#!/usr/bin/env python3
"""Lifecycle cost report: init / first-op / steady / destroy, per library.

Reads results/7056_lifecycle/lifecycle.csv (job 7056) and emits
report/LIFECYCLE-COST.md.

The point of the measurement: a benchmark that reports only steady-state
bandwidth hides what an application actually pays. Setup and teardown are
measured in SECONDS here while the transfer they enable costs microseconds.
"""
import csv
import pathlib
import sys

FIN = pathlib.Path(__file__).resolve().parent.parent
CSV = FIN / "results" / "7056_lifecycle" / "lifecycle.csv"

LADDER = [("1node", 1, [2, 4]), ("2node", 2, [2, 4, 8])]
LIBS = ["mpi", "nccl", "torch"]
LIBNAME = {"mpi": "MPI (cray-mpich)", "nccl": "NCCL", "torch": "torch.distributed"}


def load():
    with CSV.open() as fh:
        return list(csv.DictReader(fh))


def num(r, k):
    v = r.get(k, "-")
    if v in ("-", "", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def find(rows, scale, ranks, lib):
    for r in rows:
        if r["scale"] == scale and int(r["ranks"]) == ranks and r["lib"] == lib:
            return r
    return None


def ms(v):
    return "-" if v is None else f"{v / 1000:.1f}"


def us(v):
    return "-" if v is None else f"{v:.1f}"


def main():
    if not CSV.exists():
        print(f"missing {CSV}", file=sys.stderr)
        return 1
    rows = load()

    print("# Lifecycle cost: init, first op, transfer, destroy")
    print()
    print("Job **7056** on Tara North (GH200, Slingshot 11). AllReduce,")
    print("16 MiB, float32, device buffers. Phases timed separately on a")
    print("monotonic clock; every NCCL and torch collective is validated")
    print("against an analytic result.")
    print()
    print("All times in **milliseconds** unless marked us.")
    print()

    # ---------------- per-scale phase tables ----------------
    for scale, nodes, ranks in LADDER:
        label = "1 node" if nodes == 1 else "2 nodes"
        print(f"## {label}")
        print()
        print("| GPUs | library | init | first op | steady op (us) | destroy | "
              "transport |")
        print("|---|---|---|---|---|---|---|")
        for k in ranks:
            for lib in LIBS:
                r = find(rows, scale, k, lib)
                if r is None:
                    continue
                if r["status"] != "ok":
                    print(f"| {k} | {LIBNAME[lib]} | — | — | — | — | "
                          f"**{r['status']}** |")
                    continue
                print(f"| {k} | {LIBNAME[lib]} | {ms(num(r, 'init_us'))} | "
                      f"{ms(num(r, 'first_us'))} | {us(num(r, 'steady_us'))} | "
                      f"{ms(num(r, 'destroy_us'))} | {r['transport']} |")
        print()

    # ---------------- the ratio that matters ----------------
    print("## How many steady operations does setup cost?")
    print()
    print("init + first op, divided by one steady operation. This is the")
    print("number of transfers a job must do before setup stops dominating.")
    print()
    print("| scale | GPUs | library | setup (ms) | steady (us) | "
          "break-even ops |")
    print("|---|---|---|---|---|---|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            for lib in LIBS:
                r = find(rows, scale, k, lib)
                if r is None or r["status"] != "ok":
                    continue
                init = num(r, "init_us")
                first = num(r, "first_us")
                steady = num(r, "steady_us")
                if None in (init, first, steady) or steady == 0:
                    continue
                setup = init + first
                print(f"| {nodes}n | {k} | {LIBNAME[lib]} | {setup / 1000:.1f} | "
                      f"{steady:.1f} | **{setup / steady:,.0f}** |")
    print()

    # ---------------- where setup time actually goes ----------------
    print("## Cost breakdown as a share of total lifecycle")
    print()
    print("| scale | GPUs | library | init % | first op % | "
          "transfer % | destroy % |")
    print("|---|---|---|---|---|---|---|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            for lib in LIBS:
                r = find(rows, scale, k, lib)
                if r is None or r["status"] != "ok":
                    continue
                init = num(r, "init_us") or 0.0
                first = num(r, "first_us") or 0.0
                steady = num(r, "steady_us") or 0.0
                destroy = num(r, "destroy_us") or 0.0
                total = init + first + steady + destroy
                if total <= 0:
                    continue
                print(f"| {nodes}n | {k} | {LIBNAME[lib]} | "
                      f"{100 * init / total:.1f} | {100 * first / total:.1f} | "
                      f"{100 * steady / total:.2f} | "
                      f"{100 * destroy / total:.1f} |")
    print()

    # ---------------- NCCL bootstrap detail ----------------
    print("## NCCL: bootstrap vs communicator init")
    print()
    print("NCCL is the one layer that exposes the split. `bootstrap` is")
    print("MPI_Init plus broadcasting the unique ID; `init` is")
    print("ncclCommInitRank -- ring construction, transport selection,")
    print("plugin load, connection setup.")
    print()
    print("| scale | GPUs | bootstrap (ms) | commInitRank (ms) | "
          "first op (ms) |")
    print("|---|---|---|---|---|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            r = find(rows, scale, k, "nccl")
            if r is None or r["status"] != "ok":
                continue
            print(f"| {nodes}n | {k} | {ms(num(r, 'bootstrap_us'))} | "
                  f"{ms(num(r, 'init_us'))} | {ms(num(r, 'first_us'))} |")
    print()

    # ---------------- torch import ----------------
    print("## torch.distributed: the cost before any distributed call")
    print()
    print("`import torch` is not communication, but every job pays it.")
    print()
    print("| scale | GPUs | import (ms) | init (ms) | first op (ms) |")
    print("|---|---|---|---|---|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            r = find(rows, scale, k, "torch")
            if r is None or r["status"] != "ok":
                continue
            print(f"| {nodes}n | {k} | {ms(num(r, 'import_us'))} | "
                  f"{ms(num(r, 'init_us'))} | {ms(num(r, 'first_us'))} |")
    print()

    # ---------------- first-op penalty ----------------
    print("## First-op penalty: where each library hides its setup")
    print()
    print("First op divided by steady op. A high ratio means the library")
    print("defers real work -- ring construction, connection setup, buffer")
    print("registration -- to the first collective rather than to init.")
    print()
    print("| scale | GPUs | library | first op | steady op (us) | ratio |")
    print("|---|---|---|---|---|---|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            for lib in LIBS:
                r = find(rows, scale, k, lib)
                if r is None or r["status"] != "ok":
                    continue
                first = num(r, "first_us")
                steady = num(r, "steady_us")
                if None in (first, steady) or steady == 0:
                    continue
                print(f"| {nodes}n | {k} | {LIBNAME[lib]} | "
                      f"{first / 1000:.1f} ms | {steady:.1f} | "
                      f"**{first / steady:,.0f}x** |")
    print()
    print("NCCL pays 291x to 6105x; torch.distributed pays 1.5x to 29x.")
    print("Both ultimately call NCCL, so this is not a transport")
    print("difference -- it is where each API chooses to do the work.")
    print("`ncclCommInitRank` returns before connections are established;")
    print("`init_process_group` does that work up front, which is why its")
    print("init is comparable while its first op is two to three orders of")
    print("magnitude cheaper.")
    print()

    # ---------------- honest gaps ----------------
    missing = [r for r in rows if r["status"] != "ok"]
    print("## What is missing and why")
    print()
    if missing:
        print(f"{len(missing)} of {len(rows)} cells produced no result:")
        print()
        for r in missing:
            print(f"- `{r['scale']} r{r['ranks']} {r['lib']}` — {r['status']}")
        print()
    print("**MPI on 2 nodes does not appear in this report.** The program")
    print("fails with `cxil_map: write error` then `OFI tagged recv failed")
    print("(Bad address)` for any inter-node device transfer at or above")
    print("256 KiB. Measured threshold (job 7060): 4 KiB and 64 KiB pass,")
    print("256 KiB through 16 MiB all fail. That boundary is MPICH's")
    print("eager-to-rendezvous switch, so the failure is in pinning device")
    print("memory for RDMA rather than in the collective.")
    print()
    print("It is specific to this program, not the machine: OSU")
    print("`osu_allreduce -d cuda` at the same 16 MiB on the same two nodes")
    print("in the same job completes in 862-953 us (jobs 7057, 7059, 7060).")
    print()
    print("Three hypotheses were tested and all three were wrong:")
    print()
    print("1. **libcudart provenance** — rebuilt against the same")
    print("   `cuda/13.0/lib64` OSU resolves instead of the REDIST tree.")
    print("   Still fails (job 7058).")
    print("2. **cudaSetDevice ordering** — moved device selection before")
    print("   `MPI_Init`, matching OSU. Still fails (job 7059). This did")
    print("   reveal that MPI's apparent ~3 s init was almost entirely CUDA")
    print("   context creation: once moved out, init drops to ~20 ms.")
    print("3. **missing CUDA driver library** — `ldd` showed OSU links")
    print("   `libcuda.so.1` and the test binary did not. Linked it,")
    print("   confirmed present in `ldd`. Still fails (job 7062).")
    print()
    print("Not root-caused. The remaining difference between the two")
    print("binaries is OSU's buffer setup path, which allocates through its")
    print("own allocator and touches the buffer from a CUDA kernel before")
    print("the first MPI call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
