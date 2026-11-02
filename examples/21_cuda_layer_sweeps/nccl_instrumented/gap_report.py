#!/usr/bin/env python3
"""Layer-gap tables from job 7042: every layer, same transport, same shapes.

Answers "where is the gap between layers and libraries" across the
2 -> 4 -> 8 GPU ladder on 1 node and 2 nodes, for five collectives.

Key rule: a layer-to-layer gap is only printed when both rows attested the
SAME transport. Differencing a Libfabric row against a Socket row measures
the network, not the layer -- that error is what made the previous report's
"torch overhead" wrong.
"""
import csv
import sys
from pathlib import Path

FIN = Path(__file__).resolve().parent.parent
CSV = FIN / "results/7047_layer_gap/gap.csv"

PRETTY = {"fi": "fi_pingpong (NIC)", "osu": "OSU (MPI)",
          "nccl_cxi": "NCCL", "torch": "torch.distributed"}
LADDER = [("1node", 1, [2, 4]), ("2node", 2, [2, 4, 8])]


def load():
    with open(CSV) as fh:
        return [r for r in csv.DictReader(fh)]


def val(r):
    if r["status"] != "ok":
        return None
    try:
        return float(r["gbps"])
    except ValueError:
        return None


def main():
    rows = load()
    idx = {}
    for r in rows:
        idx[(r["scale"], int(r["ranks"]), r["layer"], r["op"])] = r

    print("# Where the gap is, layer by layer")
    print()
    print("Job 7047. GH200 / Slingshot 11. AllReduce and four other")
    print("collectives at 16 MiB, GPU ladder 2 -> 4 -> 8.")
    print()
    print("Transport is attested per run from the NCCL debug log. On 1 node")
    print("NCCL reports Socket because intra-node traffic never reaches the")
    print("NIC -- it goes over NVLink/P2P, so that label is expected and the")
    print("bandwidth is NVLink bandwidth.")

    # ---- table 1: the stack, allreduce, across the ladder ----
    print("\n## AllReduce 16 MiB: every layer on the ladder\n")
    print("| scale | GPUs | OSU (MPI) | NCCL | torch.distributed | transport |")
    print("|---|---|---|---|---|---|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            o = idx.get((scale, k, "osu", "all_reduce"))
            n = idx.get((scale, k, "nccl_cxi", "all_reduce"))
            t = idx.get((scale, k, "torch", "all_reduce"))
            vo, vn, vt = (val(x) if x else None for x in (o, n, t))
            tr = n["transport"] if n else "?"
            f = lambda v: f"{v:.2f}" if v is not None else "—"
            print(f"| {nodes} node{'s' if nodes > 1 else ''} | {k} | "
                  f"{f(vo)} | {f(vn)} | {f(vt)} | {tr} |")

    # ---- table 2: the gaps that are legitimate to take ----
    print("\n## The gap at each boundary\n")
    print("Only boundaries where both layers attested the same transport.")
    print()
    print("| scale | GPUs | boundary | gap |")
    print("|---|---|---|---|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            n = idx.get((scale, k, "nccl_cxi", "all_reduce"))
            t = idx.get((scale, k, "torch", "all_reduce"))
            vn, vt = (val(x) if x else None for x in (n, t))
            if vn and vt and n["transport"] == t["transport"]:
                d = 100.0 * (vt - vn) / vn
                print(f"| {nodes}n | {k} | NCCL → torch.distributed | "
                      f"{d:+.1f}% ({vn:.2f} → {vt:.2f}) |")
            o = idx.get((scale, k, "osu", "all_reduce"))
            vo = val(o) if o else None
            if vo and vn:
                print(f"| {nodes}n | {k} | OSU(MPI) → NCCL | "
                      f"{vn / vo:.2f}x ({vo:.2f} → {vn:.2f}) |")

    # ---- table 3: collective coverage ----
    print("\n## Per-collective, NCCL over CXI\n")
    colls = ["all_reduce", "all_gather", "reduce_scatter", "broadcast",
             "alltoall"]
    hdr = "| scale | GPUs | " + " | ".join(colls) + " |"
    print(hdr)
    print("|---" * (len(colls) + 2) + "|")
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            cells = []
            for c in colls:
                r = idx.get((scale, k, "nccl_cxi", c))
                v = val(r) if r else None
                cells.append(f"{v:.2f}" if v is not None else "—")
            print(f"| {nodes}n | {k} | " + " | ".join(cells) + " |")

    # ---- torch stall analysis ----
    import re
    print("\n## torch.distributed: mean vs median vs p90\n")
    print("A mean far above the median means the layer is stalling, not")
    print("running slowly. nccl-tests reports a steady-state number, so")
    print("comparing it against a stall-dominated mean misattributes the gap.")
    print()
    print("| scale | GPUs | mean us | median us | p90 us | p90/median |")
    print("|---|---|---|---|---|---|")
    base = CSV.parent
    for scale, nodes, ranks in LADDER:
        for k in ranks:
            f = base / f"{scale}_r{k}_torch.txt"
            if not f.exists():
                continue
            m = re.search(r"usec=([\d.]+) gbps=\S+ median_usec=([\d.]+) "
                          r"p90_usec=([\d.]+)", f.read_text(errors="ignore"))
            if not m:
                continue
            mean, med, p90 = (float(x) for x in m.groups())
            print(f"| {nodes}n | {k} | {mean:.1f} | {med:.1f} | {p90:.1f} | "
                  f"{p90 / med:.2f}x |")

    # ---- scaling efficiency ----
    print("\n## Scaling 2 → 4 → 8 GPUs (NCCL, AllReduce)\n")
    for scale, nodes, ranks in LADDER:
        base = idx.get((scale, ranks[0], "nccl_cxi", "all_reduce"))
        vb = val(base) if base else None
        if not vb:
            continue
        parts = []
        for k in ranks:
            r = idx.get((scale, k, "nccl_cxi", "all_reduce"))
            v = val(r) if r else None
            if v:
                ideal = vb * (k / ranks[0])
                parts.append(f"{k} GPUs {v:.1f} ({100 * v / ideal:.0f}% of linear)")
        print(f"- {nodes} node{'s' if nodes > 1 else ''}: " + "; ".join(parts))

    return 0


if __name__ == "__main__":
    sys.exit(main())
