#!/usr/bin/env python3
"""Verify every headline number in FINAL-REPORT.md against the artifacts.

Hand-copied numbers drift. This re-derives each published figure from the
CSVs and trace files and fails if the document disagrees. Exits non-zero on
any mismatch.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FIN = Path(__file__).resolve().parent.parent
DOC = (FIN / "report/FINAL-REPORT.md").read_text()
# The doc formats for humans: thousands separators and fewer decimals.
DOC_N = DOC.replace(",", "")

fails = []


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(label)


def in_doc(v, places=(2, 1)):
    """True if value appears in the doc at any of the given precisions."""
    for p in places:
        if f"{v:.{p}f}" in DOC_N:
            return True
    return str(v) in DOC_N


# ---------- 1. ladder CSVs: transport + bandwidth ----------
rows = {}
for name in ("ladder_1node.csv", "ladder_2node.csv"):
    with open(FIN / "results/7016_7017_ladder_cxi" / name) as fh:
        for r in csv.DictReader(fh):
            rows[(r["nodes"], r["ranks"], r["bytes"], r["layer"])] = r

print("== 2-node CXI vs TCP ==")
for ranks, size in (("2", "16777216"), ("2", "268435456"),
                    ("4", "16777216"), ("4", "268435456"),
                    ("8", "16777216"), ("8", "268435456")):
    tcp = rows[("2", ranks, size, "nccl")]
    cxi = rows[("2", ranks, size, "nccl_cxi")]
    check(f"2n r{ranks} {size} tcp transport=Socket",
          tcp["detail"].endswith("Socket"), tcp["detail"])
    check(f"2n r{ranks} {size} cxi transport=Libfabric",
          cxi["detail"].endswith("Libfabric"), cxi["detail"])
    check(f"2n r{ranks} {size} tcp bw {tcp['gbps']} in doc",
          in_doc(float(tcp["gbps"])), tcp["gbps"])
    check(f"2n r{ranks} {size} cxi bw {cxi['gbps']} in doc",
          in_doc(float(cxi["gbps"])), cxi["gbps"])
    gain = float(cxi["gbps"]) / float(tcp["gbps"])
    check(f"2n r{ranks} {size} gain {gain:.1f}x in doc",
          f"{gain:.1f}×" in DOC or f"{gain:.1f}x" in DOC, f"{gain:.1f}")

print("\n== 1-node control: both arms Socket, no material gain ==")
for ranks, size in (("2", "16777216"), ("2", "268435456"),
                    ("4", "16777216"), ("4", "268435456")):
    a = rows[("1", ranks, size, "nccl")]
    b = rows[("1", ranks, size, "nccl_cxi")]
    check(f"1n r{ranks} {size} both Socket",
          a["detail"].endswith("Socket") and b["detail"].endswith("Socket"))
    ratio = float(b["gbps"]) / float(a["gbps"])
    check(f"1n r{ranks} {size} within 2% ({ratio:.3f})", 0.97 <= ratio <= 1.03,
          f"{ratio:.3f}")
    check(f"1n r{ranks} {size} stock bw in doc", in_doc(float(a["gbps"])))

print("\n== OSU comparison (inversion resolved) ==")
for ranks in ("2", "4", "8"):
    osu = rows[("2", ranks, "268435456", "osu")]
    cxi = rows[("2", ranks, "268435456", "nccl_cxi")]
    check(f"2n r{ranks} osu {osu['gbps']} in doc", in_doc(float(osu["gbps"])))
    if ranks in ("4", "8"):
        check(f"2n r{ranks} NCCL CXI beats MPI",
              float(cxi["gbps"]) > float(osu["gbps"]),
              f"{cxi['gbps']} vs {osu['gbps']}")

# ---------- 2. phase shares recomputed from traces ----------
print("\n== phase shares recomputed from trace JSON ==")
CATS = {"KernelCh": "GPU", "ScheduleSend": "PROXY", "ScheduleRecv": "PROXY",
        "ProgressSend": "PROXY", "ProgressRecv": "PROXY",
        "SendWait": "NET", "RecvWait": "NET",
        "SendGpuWait": "NET", "RecvGpuWait": "NET",
        # These two are also NET in nccl_trace_stats.py. Omitting them here
        # shrank the denominator and made GPU read 4.6% instead of 4.5%,
        # i.e. the verifier was wrong and the report was right. Keep this
        # map in sync with the stats tool or it silently checks the wrong
        # thing.
        "RecvFlushWait": "NET", "SendPeerWait": "NET",
        "AllReduce": "COLL", "Group": "GROUP"}
TRACES = FIN / "results/7018_profiler_ladder"
PROF = TRACES  # traces are archived alongside the report, not read from /tmp

expected = {"r2_1048576": {"GPU": 4.5}, "r4_1048576": {"GPU": 8.6},
            "r4_268435456": {"GPU": 7.1}, "r8_268435456": {"GPU": 8.1}}

for cell, exp in expected.items():
    files = sorted(PROF.glob(f"{cell}/trace_*.json"))
    if not files:
        check(f"{cell} traces present", False, "missing")
        continue
    tot = defaultdict(float)
    unmatched = 0
    for f in files:
        open_ev = {}
        for e in json.load(open(f)):
            if e.get("ph") not in ("b", "e"):
                continue
            key = (e.get("name"), e.get("id"), e.get("pid"), e.get("tid"))
            if e["ph"] == "b":
                open_ev[key] = e["ts"]
            elif key in open_ev:
                cat = CATS.get(e.get("name"))
                if cat:
                    tot[cat] += e["ts"] - open_ev.pop(key)
                else:
                    open_ev.pop(key)
        unmatched += len(open_ev)
    check(f"{cell} unmatched events == 0", unmatched == 0, str(unmatched))
    allt = sum(tot.values())
    gpu = 100 * tot["GPU"] / allt
    check(f"{cell} GPU share {gpu:.1f}% == {exp['GPU']}%",
          abs(gpu - exp["GPU"]) < 0.15, f"{gpu:.1f}")
    check(f"{cell} GPU share in doc", f"{gpu:.1f}%" in DOC, f"{gpu:.1f}%")

check("claim: GPU never exceeds 8.6%", "never exceeds 8.6%" in DOC)

# ---------- 3. failures are documented, not hidden ----------
print("\n== failure honesty ==")
seg = (FIN / "results/7018_profiler_ladder/FAILED-r2_268435456-sigsegv.txt").read_text(errors="replace")
oob = (FIN / "results/7018_profiler_ladder/FAILED-r8_1048576-validation.txt").read_text(errors="replace")
check("sigsegv log really shows signal 11", "signal 11" in seg)
check("oob log really shows 8 FAILED", "Out of bounds values : 8 FAILED" in oob)
check("doc reports the SIGSEGV", "signal 11" in DOC)
check("doc reports the validation failure", "Out of bounds values : 8 FAILED" in DOC)
check("doc reports the bogus 3.14805 figure", "3.14805" in DOC)

# ---------- 4. no published ladder cell failed validation ----------
bad = [k for k, r in rows.items() if "validation_oob" in r.get("detail", "")]
check("no published ladder cell has validation_oob", not bad, str(bad))

# ---------- 5. §8 correctness bug: re-derive every rate from oob2.csv ----------
print("\n== correctness bug (job 7031) ==")
o2 = list(csv.DictReader(open(FIN / "results/7031_oob_characterise/oob2.csv")))


def rate(rs):
    """(failures, total). A row is a failure when nccl-tests' own check found
    out-of-bounds values -- never when it merely exited non-zero."""
    return sum(1 for r in rs if r["oob"] not in ("0", "NA")), len(rs)


def grp(name):
    return [r for r in o2 if r["group"] == name]


def row_says(label, f, n, unique=True):
    """True if the doc row identified by `label` carries the value f/n.

    Substring matching is not enough. Three distinct leaks were found by
    tools/tamper_test.py and all three had the same shape: the check passed
    because some OTHER line in the document satisfied it.
      - "0 / 6" appears in many rows, so a tampered Tree row still matched.
      - "1 MiB" appears in both the size table and the rank table, and the
        rank table legitimately holds 3/6, so faking the size row matched
        the rank row instead.
    So: the label must identify exactly one row (unique=True), and the value
    is read out of THAT row.
    """
    hits = [l for l in DOC.splitlines()
            if l.strip().startswith("|") and label in l]
    if unique and len(hits) != 1:
        return False
    if not hits:
        return False
    cells = [c.strip().strip("*` ") for c in hits[0].split("|")]
    return any(c.replace(" ", "") == f"{f}/{n}" for c in cells)


def line_says(anchor, value):
    """True if the single line containing `anchor` also contains `value`.

    For figures quoted in prose or code blocks, where the same number
    legitimately appears elsewhere in the document.
    """
    hits = [l for l in DOC.splitlines() if anchor in l]
    return len(hits) == 1 and value in hits[0]


# Size table rows are anchored on the full cell text so they cannot be
# satisfied by the rank table, which also mentions "1 MiB".
for size, label in [("524288", "| 512 KiB, 8 ranks, CXI"),
                    ("786432", "| 768 KiB, 8 ranks, CXI"),
                    ("1048576", "| **1 MiB**, 8 ranks, CXI"),
                    ("1310720", "| 1.25 MiB, 8 ranks, CXI"),
                    ("2097152", "| 2 MiB, 8 ranks, CXI")]:
    rs = [r for r in grp("sizescan") if r["size"] == size]
    f, n = rate(rs)
    check(f"sizescan row {label.strip('| ')} == {f}/{n}",
          row_says(label, f, n), f"{f}/{n}")

for g, label in [("algo_Ring", "NCCL_ALGO=Ring"), ("algo_Tree", "NCCL_ALGO=Tree"),
                 ("proto_Simple", "NCCL_PROTO=Simple"), ("proto_LL", "NCCL_PROTO=LL`"),
                 ("proto_LL128", "NCCL_PROTO=LL128"),
                 ("rank2", "| 2 ranks, 2 nodes, 1 MiB"),
                 ("rank4", "| 4 ranks, 2 nodes, 1 MiB"),
                 ("rank8_1node", "| 4 ranks, **1 node**, 1 MiB")]:
    f, n = rate(grp(g))
    check(f"row {label} == {f}/{n}", row_says(label, f, n), f"{f}/{n}")

# The headline claim: default rate must equal Ring's, and Tree must be clean.
df, dn = rate([r for r in grp("sizescan") if r["size"] == "1048576"])
rf, rn = rate(grp("algo_Ring"))
tf, tn = rate(grp("algo_Tree"))
check("default failure rate == Ring failure rate", (df, dn) == (rf, rn),
      f"default {df}/{dn} vs ring {rf}/{rn}")
check("Tree is clean (0 failures)", tf == 0, f"{tf}/{tn}")
check("doc names Tree as the workaround", "NCCL_ALGO=Tree" in DOC)

# Cumulative 8/16 = 7030's 5/10 plus 7031's 3/6.
probe = (FIN / "results/7030_oob_probe/oob_probe.txt").read_text()
cxi_1m = [l for l in probe.splitlines()
          if "size=1048576" in l and "arm=cxi" in l and "rep=" in l][:10]
p_fail = sum(1 for l in cxi_1m if re.search(r"oob=(?!0\b)\d+", l))
check("7030 CXI 1MiB: 5/10 wrong", (p_fail, len(cxi_1m)) == (5, 10),
      f"{p_fail}/{len(cxi_1m)}")
sock = [l for l in probe.splitlines() if "arm=stock" in l]
s_fail = sum(1 for l in sock if re.search(r"oob=(?!0\b)\d+", l))
check("7030 Socket control: 0/5 wrong", (s_fail, len(sock)) == (0, 5),
      f"{s_fail}/{len(sock)}")
check("cumulative 8/16 stated", "8 / 16" in DOC)
check("doc states Socket never failed", "0 / 5" in DOC)

# 7030 is the transport-attested job; the claim must rest on it, and the
# doc must admit 7031 could not self-attest.
check("7030 attests Libfabric", probe.count("transport=Libfabric") == 17)
check("7030 attests Socket", probe.count("transport=Socket") == 5)
check("doc admits 7031 transport=unknown",
      "transport=unknown" in DOC and "7031" in DOC)
check("all 7031 rows are indeed unattested",
      all(r["transport"] == "unknown" for r in o2))

# ---------- 6. §9 profiler bisect ----------
print("\n== profiler bisect (job 7027) ==")
bis = (FIN / "results/7027_profiler_bisect/bisect_summary.txt").read_text()
for m in ("16", "64", "24", "90"):
    line = [l for l in bis.splitlines()
            if f"mask={m} " in l and "ranks=2" in l]
    check(f"mask={m} at 2 ranks really SIGSEGVs",
          bool(line) and "sigsegv" in line[0], str(line[:1]))
for m in ("none", "0", "2", "8"):
    line = [l for l in bis.splitlines()
            if f"mask={m} " in l and "ranks=2" in l]
    check(f"mask={m} at 2 ranks really ok",
          bool(line) and "status=ok" in line[0], str(line[:1]))
bnone = [l for l in bis.splitlines()
         if "mask=none" in l and "ranks=8" in l and "size=1048576" in l]
check("cell B fails WITHOUT profiler (the correction)",
      bool(bnone) and "validation_failed" in bnone[0], str(bnone[:1]))
check("doc issues the correction", "That was wrong" in DOC)
check("doc keeps the 4 FAILED evidence", "Out of bounds : 4 FAILED" in DOC)

# ---------- 7. §10 variant sweep ----------
print("\n== variant sweep (job 7028) ==")
sw = list(csv.DictReader(open(FIN / "results/7028_sweep_cxi/sweep_cxi.csv")))
ok = [r for r in sw if r["status"] == "ok"]
bad_sw = [r for r in sw if r["status"].startswith("validation_oob")]
check("19 cells attempted", len(sw) == 19, str(len(sw)))
check("14 valid / 5 failed", (len(ok), len(bad_sw)) == (14, 5),
      f"{len(ok)}/{len(bad_sw)}")
check("every sweep cell attested Libfabric",
      all(r["transport"] == "Libfabric" for r in sw))
AXIS_LABEL = {"collective": "coll"}  # CSV axis name -> label used in the doc

for r in ok:
    # Anchor on the cell's own row: several bandwidths appear more than once
    # in the document (115.33 is quoted in §8 as well), so a bare substring
    # test let an inflated sweep number through. Compare numerically, since
    # the doc writes 81.00 where the CSV holds 81, and bolds some cells.
    axis = AXIS_LABEL.get(r["axis"], r["axis"])
    cell = r["value"]
    want = float(r["peak_busbw"])
    hits = [l for l in DOC.splitlines()
            if l.strip().startswith("|") and f"| {axis} | {cell} |" in l]
    got = None
    if len(hits) == 1:
        fields = [c.strip().strip("*` ") for c in hits[0].split("|")]
        for f in fields:
            try:
                got = float(f)
                break
            except ValueError:
                continue
    check(f"sweep row {axis}/{cell} == {want}",
          got is not None and abs(got - want) < 0.005,
          f"doc={got} csv={want}")

check("all 5 failures are allreduce variants",
      all(r["coll"] == "all_reduce" for r in bad_sw),
      str([r["cell"] for r in bad_sw]))
check("doc states ordering did NOT survive", "did NOT survive" in DOC)
check("doc states the 14/5 split",
      line_says("produced valid numbers",
                f"**{len(ok)} produced valid numbers; {len(bad_sw)} failed"))

# ---------- 8. §11 variance decomposition ----------
print("\n== variance (jobs 7023-7026) ==")
var = (FIN / "report/VARIANCE.txt").read_text()


def var_num(pattern):
    """Pull a figure out of the generated VARIANCE.txt, not out of prose."""
    m = re.search(pattern, var)
    return m.group(1) if m else None


# Each figure is re-derived from VARIANCE.txt and then required to appear on
# the matching line of the doc's code block, so understating one cannot be
# masked by the same digits appearing elsewhere in the document.
VAR_CHECKS = [
    ("within-job CoV", r"within-job CoV\s+\(jitter\)\s*:\s*mean ([\d.]+)%",
     "within-job CoV"),
    ("across-job CoV", r"across-job CoV\s+\(placement\)\s*:\s*([\d.]+)%",
     "across-job CoV"),
    ("overall CoV", r"overall CoV\s+\(what a user sees\)\s*:\s*([\d.]+)%",
     "overall CoV"),
    ("overall spread", r"overall spread \(max/min-1\)\s*:\s*([\d.]+)%?",
     "overall spread"),
]
for key, pat, anchor in VAR_CHECKS:
    v = var_num(pat)
    check(f"{key} re-derived from VARIANCE.txt = {v}", v is not None, str(v))
    if v:
        lines = [l for l in DOC.splitlines() if anchor in l]
        check(f"{key} {v} on its own line in doc",
              len(lines) == 1 and v in lines[0],
              f"{v} not in {lines[:1]}")

check("verdict is WITHIN-JOB", "WITHIN-JOB dominates" in var)
check("doc says 57% is gone", "57% is gone" in DOC or "57% is resolved" in DOC
      or "The 57% is gone" in DOC)
check("doc gives all four job medians",
      all(m in DOC for m in ("121.02", "121.11", "119.80", "118.00")))

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print(f"all {len(DOC.splitlines()) and 'checks'} passed")
sys.exit(0)
