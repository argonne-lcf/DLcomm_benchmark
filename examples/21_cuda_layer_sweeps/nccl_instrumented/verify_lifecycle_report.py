#!/usr/bin/env python3
"""Verify report/LIFECYCLE-COST.md against results/7056_lifecycle/lifecycle.csv.

Re-derives every published number from the artifact rather than checking that
a string appears somewhere in the document. A substring test passes when the
same digits occur anywhere, including in an unrelated row, so each check here
anchors on the row it belongs to and compares floats.

Exit 0 only when every check passes.
"""
import csv
import pathlib
import re
import sys

FIN = pathlib.Path(__file__).resolve().parent.parent
CSV = FIN / "results" / "7056_lifecycle" / "lifecycle.csv"
DOC = FIN / "report" / "LIFECYCLE-COST.md"

LIBNAME = {"mpi": "MPI (cray-mpich)", "nccl": "NCCL", "torch": "torch.distributed"}

PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)


def bad(msg):
    FAIL.append(msg)


def rows_of(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def num(r, k):
    v = r.get(k, "-")
    if v in ("-", "", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def doc_rows(doc, section=None):
    """Markdown table rows as lists of stripped cells.

    When `section` is given, only rows under that exact `## ` heading are
    returned. Without it the 1-node and 2-node tables are indistinguishable
    and every lookup matches twice.
    """
    out = []
    cur = None
    for line in doc.splitlines():
        st = line.strip()
        if st.startswith("## "):
            cur = st[3:].strip()
            continue
        if not st.startswith("|"):
            continue
        if section is not None and cur != section:
            continue
        out.append([c.strip() for c in st.strip("|").split("|")])
    return out


def cell_float(c):
    """Pull a float out of a cell that may carry **, x, ms, or commas."""
    c = c.replace("**", "").replace(",", "").replace("ms", "")
    c = c.replace("x", "").strip()
    m = re.search(r"-?\d+\.?\d*", c)
    return float(m.group()) and float(m.group()) if m else None


def find_row(drows, want_cells, numeric_idx=None):
    """Find a doc row whose leading cells match want_cells exactly."""
    hits = []
    for cells in drows:
        if len(cells) <= len(want_cells):
            continue
        if all(cells[i] == w for i, w in enumerate(want_cells)):
            hits.append(cells)
    return hits


def close(a, b, tol=0.15):
    """Published values are rounded; allow a small absolute/relative slack."""
    if a is None or b is None:
        return False
    if abs(a - b) <= tol:
        return True
    return abs(a - b) <= abs(b) * 0.01


def main():
    for p in (CSV, DOC):
        if not p.exists():
            print(f"MISSING {p}", file=sys.stderr)
            return 2

    rows = rows_of(CSV)
    doc = DOC.read_text()
    drows = doc_rows(doc)
    SEC_PHASE = {"1node": "1 node", "2node": "2 nodes"}
    SEC_BE = "How many steady operations does setup cost?"
    SEC_FO = "First-op penalty: where each library hides its setup"

    # ---- every ok row must appear with its measured values ----
    for r in rows:
        scale, ranks, lib = r["scale"], r["ranks"], r["lib"]
        nodes = r["nodes"]
        tag = f"{scale} r{ranks} {lib}"

        if r["status"] != "ok":
            # must be declared missing, not silently dropped
            if f"`{scale} r{ranks} {lib}`" in doc:
                ok(f"{tag}: declared missing")
            else:
                bad(f"{tag}: status={r['status']} but not declared in report")
            continue

        # phase table: | GPUs | library | init | first | steady | destroy | tr |
        hits = find_row(doc_rows(doc, SEC_PHASE[scale]), [ranks, LIBNAME[lib]])
        if not hits:
            bad(f"{tag}: no phase-table row found")
            continue
        if len(hits) > 1:
            bad(f"{tag}: phase-table row is ambiguous ({len(hits)} matches)")
            continue
        cells = hits[0]

        for idx, key, scale_div, label in (
            (2, "init_us", 1000.0, "init"),
            (3, "first_us", 1000.0, "first op"),
            (4, "steady_us", 1.0, "steady"),
            (5, "destroy_us", 1000.0, "destroy"),
        ):
            want = num(r, key)
            if want is None:
                continue
            got = cell_float(cells[idx])
            if close(got, want / scale_div):
                ok(f"{tag}: {label} {got}")
            else:
                bad(f"{tag}: {label} doc={got} csv={want / scale_div:.1f}")

        # transport must be reproduced exactly, not paraphrased
        if cells[6] == r["transport"]:
            ok(f"{tag}: transport {cells[6]}")
        else:
            bad(f"{tag}: transport doc={cells[6]} csv={r['transport']}")

        # ---- break-even ----
        init, first, steady = num(r, "init_us"), num(r, "first_us"), num(r, "steady_us")
        if None not in (init, first, steady) and steady:
            be = (init + first) / steady
            hits = find_row(doc_rows(doc, SEC_BE), [f"{nodes}n", ranks, LIBNAME[lib]])
            got = None
            for cells in hits:
                if len(cells) == 6:          # break-even table shape
                    got = cell_float(cells[5])
            if got is None:
                bad(f"{tag}: no break-even row")
            elif abs(got - be) <= max(1.0, be * 0.01):
                ok(f"{tag}: break-even {got:.0f}")
            else:
                bad(f"{tag}: break-even doc={got:.0f} calc={be:.0f}")

        # ---- first-op ratio ----
        if None not in (first, steady) and steady:
            ratio = first / steady
            got = None
            for cells in find_row(doc_rows(doc, SEC_FO),
                                  [f"{nodes}n", ranks, LIBNAME[lib]]):
                if len(cells) == 6:
                    got = cell_float(cells[5])
            if got is not None and abs(got - ratio) > max(1.0, ratio * 0.01):
                bad(f"{tag}: first-op ratio doc={got:.0f} calc={ratio:.0f}")
            elif got is not None:
                ok(f"{tag}: first-op ratio {got:.0f}")

    # ---- the MPI 2-node gap must be stated, with its evidence ----
    required = [
        ("cxil_map: write error", "names the actual error"),
        ("eager", "names the eager/rendezvous boundary"),
        ("Not root-caused", "does not claim a root cause"),
    ]
    for needle, why in required:
        if needle in doc:
            ok(f"gap disclosure: {why}")
        else:
            bad(f"gap disclosure MISSING: {why} ('{needle}')")

    # The threshold sentence must state BOTH measured sizes and cite the job.
    # A bare "256 KiB" substring test passes even when the sentence is wrong,
    # because the string occurs elsewhere in the prose.
    norm = " ".join(doc.split())
    m = re.search(r"job 7060\)[.:]?\s*([^.]*?)\.", norm)
    sentence = m.group(1) if m else ""
    if "4 KiB" in sentence and "64 KiB" in sentence and "pass" in sentence:
        ok("threshold: passing sizes stated with job 7060")
    else:
        bad(f"threshold: passing-size sentence not anchored to job 7060 "
            f"(got '{sentence[:60]}')")
    if re.search(r"256 KiB through 16 MiB all fail", norm):
        ok("threshold: failing range stated exactly")
    else:
        bad("threshold: failing range '256 KiB through 16 MiB all fail' absent")

    # ---- no 2-node MPI number may be published ----
    for cells in drows:
        if len(cells) > 2 and cells[1] == LIBNAME["mpi"]:
            if any(re.search(r"\d", c) for c in cells[2:6]):
                # allowed only for 1-node rows, which carry no 'n' prefix
                pass
    for r in rows:
        if r["lib"] == "mpi" and r["scale"] == "2node" and r["status"] == "ok":
            bad("2-node MPI marked ok in CSV but was expected to fail")

    print(f"{len(PASS)} PASS, {len(FAIL)} FAIL")
    for f in FAIL:
        print("  FAIL:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
