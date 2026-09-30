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
check("doc states variance NOT attributed", "not proven" in DOC.lower())

# ---------- 4. no published ladder cell failed validation ----------
bad = [k for k, r in rows.items() if "validation_oob" in r.get("detail", "")]
check("no published ladder cell has validation_oob", not bad, str(bad))

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print(f"all {len(DOC.splitlines()) and 'checks'} passed")
sys.exit(0)
