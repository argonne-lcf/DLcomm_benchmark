#!/usr/bin/env python3
"""Verify the NIXL-coverage prose against the raw job artifacts.

Same discipline as verify_prose.py: every figure I typed into the new section
must be re-derived from the JSON/logs job 6959 produced. Numbers I wrote by
hand are exactly the ones that have been wrong before.
"""
import json
import re
import sys
from pathlib import Path

REPORT = Path("/home/kvelusamy/Desktop/tara-layer-sweeps-final/report/PER-LAYER-REPORT.md").read_text()
R = Path("/home/kvelusamy/Desktop/tara-layer-sweeps-final/results/6959/2node")

fail = []


def check(label, ok, detail):
    print(f"{'OK ' if ok else 'BAD'} {label}: {detail}")
    if not ok:
        fail.append(label)


# --- ladder table rows must match the JSON exactly
d = json.loads((R / "nixl_LADDER_VRAM.json").read_text())
rows = d if isinstance(d, list) else d.get("rows", d.get("results", []))
pts = [(int(r.get("bytes", r.get("size"))), float(r.get("best_gbps", r.get("gbps"))),
        float(r["best_us"])) for r in rows]
check("ladder has 19 points", len(pts) == 19, f"{len(pts)} points")

missing = [b for b, _, _ in pts if f"| {b:,} |" not in REPORT]
check("every ladder size in table", not missing, f"missing={missing}")

# the flat-floor claim
sm = [u for b, _, u in pts if b <= 1 << 20]
lo, hi = min(sm), max(sm)
m = re.search(r"best time stays between ([\d,]+) and ([\d,]+) us", REPORT)
check("flat-floor window cited from data",
      m and int(m.group(1).replace(",", "")) == round(lo)
        and int(m.group(2).replace(",", "")) == round(hi),
      f"data {lo:,.0f}-{hi:,.0f} us; prose {m.groups() if m else None}")

# 1 GiB figure
gib = [g for b, g, _ in pts if b == 1 << 30][0]
m = re.search(r"Only at 1 GiB \(([\d.]+) GB/s\)", REPORT)
check("1 GiB rate cited from data",
      m and abs(float(m.group(1)) - gib) < 0.05,
      f"data {gib:.4f}; prose {m.group(1) if m else None}")

# --- prepped speedups
txt = (R / "nixl_MODE_prepped.txt").read_text(errors="replace")
sp = re.findall(r"speedup\s+([\d.]+)x", txt)
cited = re.search(r"Measured speedup over the naive path: ([0-9.x, ]+)", REPORT)
cited_s = cited.group(1) if cited else ""
check("all prepped speedups cited",
      all(f"{x}x" in cited_s for x in sp),
      f"log={sp}; prose='{cited_s}'")

# the prose claims the two jobs disagree; verify that against 6958's log
R58 = R.parent.parent / "6958/2node/nixl_MODE_prepped.txt"
if R58.exists():
    sp58 = re.findall(r"speedup\s+([\d.]+)x", R58.read_text(errors="replace"))
    check("6958 vs 6959 speedups really do disagree",
          sp58 != sp and len(sp58) == len(sp),
          f"6958={sp58} 6959={sp}")
    for a, b in (("1.22", sp58[-1] if sp58 else ""), ("0.78", sp[-1] if sp else "")):
        check(f"1 GiB figure {a} is real", a == b, f"log has {b}")
    check("prose does not claim a speedup trend",
          "not measurably faster" in REPORT,
          "noise caveat present")

# preparation cost claim
pr = [float(x) for x in re.findall(r"prep once\s+([\d.]+) ms", txt)]
check("prep cost ~0.02 ms is real", pr and max(pr) < 0.05, f"log={pr}")

# --- batch: the failure must be described with the real numbers
bt = (R / "nixl_MODE_batch.txt").read_text(errors="replace")
descs = re.findall(r"(\d+) descs \| total \S+ \| best\s+([\d.]+) us", bt)
ok_n = [int(n) for n, _ in descs]
check("batch passed only at N=1,4", ok_n == [1, 4], f"N with results={ok_n}")
check("rail error is real, not inferred",
      "Failed to process completions on rail" in bt,
      "found in 6959 batch log")
check("prose names the N=16 failure", "fails at N=16" in REPORT, "cited")

per4 = [float(u) for n, u in descs if n == "4"]
per1 = [float(u) for n, u in descs if n == "1"]
if per1 and per4:
    d4 = per4[0] / 4
    check("per-descriptor N=4 figure cited",
          f"{d4:,.0f}".replace(",", ",") in REPORT.replace(" us/desc", ""),
          f"computed {d4:,.0f} us/desc")

# --- introspection claims
it = (R / "nixl_MODE_introspect.txt").read_text(errors="replace")
for probe, token in (("query_memory", "NIXL_ERR_NOT_SUPPORTED"),
                     ("get_xfer_telemetry", "NIXL_ERR_NO_TELEMETRY")):
    check(f"{probe} error is real", token in it, token)
check("estimate_xfer_cost UNKNOWN is real", "'UNKNOWN'" in it, "in log")
check("notif NOT SEEN is real",
      "NOT SEEN" in (R / "nixl_MODE_notif.txt").read_text(errors="replace"),
      "in log")

print()
if fail:
    print("FAILED:", ", ".join(fail))
    sys.exit(1)
print("ALL NIXL-COVERAGE CLAIMS VERIFIED AGAINST JOB 6959 ARTIFACTS")
