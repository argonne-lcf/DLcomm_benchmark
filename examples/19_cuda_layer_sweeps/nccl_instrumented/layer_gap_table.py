#!/usr/bin/env python3
"""Build the layer-gap tables: what each layer costs relative to the one below.

The question is "where is the gap between layers and libraries" at
2 -> 4 -> 8 GPUs on 1 node and 2 nodes. This emits, per cell, each layer's
achieved bandwidth and the percentage given up against the layer beneath it,
so the loss is attributable to a specific boundary rather than to the stack
as a whole.

Layer order (bottom to top):
  fi        raw libfabric / CXI, the NIC ceiling
  osu       MPI collective
  nccl_cxi  NCCL over the CXI plugin
  torch     torch.distributed (NCCL backend)
  nixl      NIXL transfer layer

A cell with no valid measurement is printed as such. Nothing is interpolated.
"""
import csv
import sys
from pathlib import Path

FIN = Path(__file__).resolve().parent.parent
LADDER = FIN / "results/7016_7017_ladder_cxi"

ORDER = ["fi", "osu", "nccl_cxi", "torch", "nixl"]
PRETTY = {
    "fi": "fi (libfabric/CXI)",
    "osu": "osu (MPI)",
    "nccl_cxi": "NCCL (CXI)",
    "torch": "torch.distributed",
    "nixl": "NIXL",
    "nccl": "NCCL (stock/TCP)",
}


def load(path, scale):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            r["scale"] = scale
            rows.append(r)
    return rows


def gbps(r):
    try:
        return float(r["gbps"])
    except (ValueError, KeyError, TypeError):
        return None


def transport_of(r):
    """Which network the row actually proved it used.

    A layer-to-layer gap is only meaningful when both rows ran over the same
    transport. The 2-node torch rows carry a bare 'busbw' detail with no
    attestation, while the NCCL rows prove 'Libfabric' -- differencing them
    would report a TCP-vs-CXI delta as a torch overhead, which is how the
    original report got its numbers wrong.
    """
    d = (r.get("detail") or "")
    if "net_Libfabric" in d:
        return "Libfabric"
    if "net_Socket" in d:
        return "Socket"
    if r["layer"] == "osu":
        return "mpi"        # MPI has its own stack; never differenced blind
    return "unattested"


def table(rows, size, scale, nodes):
    """One table: layers down the side, rank counts across."""
    ranks = sorted({int(r["ranks"]) for r in rows
                    if r["bytes"] == str(size) and r["scale"] == scale})
    if not ranks:
        return ""

    out = []
    label = f"{size // (1 << 20)} MiB"
    out.append(f"\n### {nodes} node{'s' if nodes > 1 else ''}, {label}\n")
    hdr = "| layer | " + " | ".join(f"{k} GPUs" for k in ranks) + " |"
    out.append(hdr)
    out.append("|---" * (len(ranks) + 1) + "|")

    by = {}
    for r in rows:
        if r["bytes"] != str(size) or r["scale"] != scale:
            continue
        by[(r["layer"], int(r["ranks"]))] = r

    for layer in ORDER:
        cells = []
        present = False
        for k in ranks:
            r = by.get((layer, k))
            if r is None:
                cells.append("—")
                continue
            v = gbps(r)
            if v is None or r.get("status") != "ok":
                d = (r.get("detail") or r.get("status") or "n/a")
                short = {
                    "see_layer1_per_rail_ceiling": "not on ladder",
                    "pairwise_2_rank_api_only": "2-rank API only",
                    "no_cxi_loopback_same_node": "no loopback",
                }.get(d, d)
                cells.append(f"*{short}*")
                continue
            present = True
            tr = transport_of(r)
            mark = " ᵗ" if tr == "unattested" else ""
            cells.append(f"{v:.2f}{mark}")
        if present or layer in ("fi", "nixl"):
            out.append(f"| {PRETTY[layer]} | " + " | ".join(cells) + " |")

    if any("ᵗ" in l for l in out):
        out.append("")
        out.append("ᵗ = transport not attested in the log; on 2 nodes these ran "
                   "before the CXI plugin was wired in and are TCP numbers.")

    out.append("")
    out.append("**Gap between adjacent layers** (% change vs the layer below):")
    out.append("")
    out.append("| boundary | " + " | ".join(f"{k} GPUs" for k in ranks) + " |")
    out.append("|---" * (len(ranks) + 1) + "|")

    pairs = [("osu", "nccl_cxi"), ("nccl_cxi", "torch")]
    for lo, hi in pairs:
        cells = []
        any_cell = False
        for k in ranks:
            a, b = by.get((lo, k)), by.get((hi, k))
            va = gbps(a) if a and a.get("status") == "ok" else None
            vb = gbps(b) if b and b.get("status") == "ok" else None
            if va is None or vb is None:
                cells.append("—")
                continue
            ta, tb = transport_of(a), transport_of(b)
            # Refuse to difference rows that did not prove the same transport.
            if "unattested" in (ta, tb) and {ta, tb} != {"unattested"}:
                cells.append("*n/c*")
                continue
            any_cell = True
            delta = 100.0 * (vb - va) / va
            cells.append(f"{delta:+.1f}%")
        if any_cell or any(c == "*n/c*" for c in cells):
            cells_s = " | ".join(cells)
            out.append(f"| {PRETTY[lo]} → {PRETTY[hi]} | {cells_s} |")

    if any("n/c" in l for l in out):
        out.append("")
        out.append("*n/c* = not comparable: the two layers did not prove the "
                   "same transport, so the difference would measure the "
                   "network, not the layer.")

    return "\n".join(out)


def main():
    rows = (load(LADDER / "ladder_1node.csv", "1node")
            + load(LADDER / "ladder_2node.csv", "2node"))

    print("# Layer-by-layer gap across the GPU ladder")
    print()
    print("Bandwidth in GB/s (bus bandwidth for collectives). Measured on")
    print("Tara North, GH200 / Slingshot 11 / CXI. Jobs 7016 (1 node) and")
    print("7017 (2 nodes). Cells that produced no valid measurement say why.")

    for scale, nodes in (("1node", 1), ("2node", 2)):
        for size in (16777216, 268435456):
            t = table(rows, size, scale, nodes)
            if t:
                print(t)

    # The headline the question actually asks for.
    print("\n---\n")
    print("## Where the gap is")
    print()
    by = {}
    for r in rows:
        if r.get("status") == "ok" and gbps(r) is not None:
            by[(r["scale"], r["layer"], int(r["ranks"]), r["bytes"])] = gbps(r)

    for scale, nodes in (("1node", 1), ("2node", 2)):
        for size in (16777216, 268435456):
            ks = sorted({k for (s, l, k, b) in by
                         if s == scale and b == str(size)})
            for k in ks:
                mpi = by.get((scale, "osu", k, str(size)))
                nccl = by.get((scale, "nccl_cxi", k, str(size)))
                if mpi and nccl:
                    ratio = nccl / mpi
                    print(f"- {nodes} node / {k} GPUs / {size//(1<<20)} MiB: "
                          f"NCCL(CXI) {nccl:.2f} vs MPI {mpi:.2f} GB/s "
                          f"= {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
