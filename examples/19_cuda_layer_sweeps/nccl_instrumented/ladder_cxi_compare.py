#!/usr/bin/env python3
"""Build the CXI-vs-TCP ladder comparison from ladder_results.csv files.

Reads the CSVs produced by layer_ladder.sh and emits the cross-layer table
used in the final report. Refuses to emit a comparison when a NCCL cell's
transport was not recorded, because an unattributed fast number is exactly
the failure mode this exercise exists to expose.
"""
import csv
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    rows = load(sys.argv[1:])
    if not rows:
        print("no rows")
        return 1

    # index: (nodes, ranks, bytes, layer) -> row
    idx = {}
    for r in rows:
        idx[(r["nodes"], r["ranks"], r["bytes"], r["layer"])] = r

    keys = sorted({(r["nodes"], r["ranks"], r["bytes"]) for r in rows},
                  key=lambda k: (int(k[0]), int(k[1]), int(k[2])))

    unattributed = []
    print(f"{'nodes':>5} {'ranks':>5} {'bytes':>10} "
          f"{'nccl_tcp':>12} {'nccl_cxi':>12} {'gain':>7} "
          f"{'osu':>10} {'torch':>10}  transport")
    for nodes, ranks, size in keys:
        tcp = idx.get((nodes, ranks, size, "nccl"))
        cxi = idx.get((nodes, ranks, size, "nccl_cxi"))
        osu = idx.get((nodes, ranks, size, "osu"))
        tor = idx.get((nodes, ranks, size, "torch"))

        t_bw = num(tcp["gbps"]) if tcp else None
        c_bw = num(cxi["gbps"]) if cxi else None
        gain = f"{c_bw / t_bw:.1f}x" if (t_bw and c_bw and t_bw > 0) else "-"

        # transport attribution, from the CSV detail column
        tr_t = (tcp or {}).get("detail", "")
        tr_c = (cxi or {}).get("detail", "")
        trans = f"tcp={tr_t.replace('busbw_net_', '')} cxi={tr_c.replace('busbw_net_', '')}"
        for label, d in (("nccl", tr_t), ("nccl_cxi", tr_c)):
            if "net_unknown" in d:
                unattributed.append((nodes, ranks, size, label))

        print(f"{nodes:>5} {ranks:>5} {size:>10} "
              f"{(f'{t_bw:.2f}' if t_bw else '-'):>12} "
              f"{(f'{c_bw:.2f}' if c_bw else '-'):>12} {gain:>7} "
              f"{(osu['gbps'] if osu else '-'):>10} "
              f"{(tor['gbps'] if tor else '-'):>10}  {trans}")

    print()
    if unattributed:
        print(f"REFUSING TO PUBLISH: {len(unattributed)} NCCL cell(s) have no "
              f"recorded transport (net_unknown).")
        for u in unattributed:
            print(f"  nodes={u[0]} ranks={u[1]} bytes={u[2]} layer={u[3]}")
        print("A fast number with no 'Using network X' line cannot be "
              "attributed to CXI. Re-run with NCCL_DEBUG=INFO.")
        return 1

    print("all NCCL cells carry a recorded transport")
    return 0


if __name__ == "__main__":
    sys.exit(main())
