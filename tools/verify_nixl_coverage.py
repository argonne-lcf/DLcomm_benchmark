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

# --- structural: every declared layer present at BOTH scales
LAYER_ORDER = ("fi", "osu", "cpp_ccl", "torch_dist", "torchcomms", "nixl")
secs = re.split(r"\n## ", REPORT)
scale_secs = {s.split("\n")[0]: s for s in secs
              if re.match(r"\d+ nodes? / \d+ GPUs", s.split("\n")[0])}
check("both scale sections present", len(scale_secs) == 2, list(scale_secs))
for name, body in scale_secs.items():
    subs = re.findall(r"\n### Layer \d+ — ([^\n]+)", body)
    check(f"[{name}] has 6 layer subsections", len(subs) == 6, f"{len(subs)}: {subs}")
    check(f"[{name}] names torchcomms explicitly",
          any("torchcomms" in s for s in subs), "Layer 6 present")
    check(f"[{name}] names NIXL explicitly",
          any("NIXL" in s for s in subs), "Layer 5 present")

# a layer that is absent must say why, not just be empty
for name, body in scale_secs.items():
    m = re.search(r"### Layer 6 — torchcomms\n(.*?)(?=\n### |\Z)", body, re.S)
    txt = (m.group(1) if m else "")
    check(f"[{name}] torchcomms absence is explained",
          "not installed" in txt and len(txt.strip()) > 80,
          f"{len(txt.strip())} chars of reason")

# --- 1-node NIXL must show the 4 unsupported cells and the loopback reason
one = next((b for n, b in scale_secs.items() if n.startswith("1 node")), "")
m = re.search(r"### Layer 5 — NIXL[^\n]*\n(.*?)(?=\n### |\Z)", one, re.S)
nx1 = m.group(1) if m else ""
check("1-node NIXL lists 4 cells", nx1.count("| unsupported |") == 4,
      f"{nx1.count('| unsupported |')} unsupported rows")
check("1-node NIXL explains loopback", "no loopback" in nx1, "reason given")

# --- cell totals in the Coverage table must match the CSVs
import csv as _csv
for jn, sub, label in (("6956", "1node", "1 node"), ("6957", "2node", "2 nodes")):
    p = R.parent.parent / jn / sub / "sweep_results.csv"
    if p.exists():
        n = len(list(_csv.DictReader(open(p))))
        check(f"coverage total for {label} matches CSV",
              re.search(rf"\| {label}[^|]*\| {jn} \|.*\| {n} \|", REPORT) is not None,
              f"CSV has {n} cells")

# --- the batch row must not be labelled a plain timeout
m = re.search(r"\| MODE_batch \| failed \| \d+ \| ([^|]+)\|", REPORT)
check("batch detail corrected from 'timeout'",
      m and "rail_completion_failure" in m.group(1),
      m.group(1).strip() if m else "row missing")

# --- status semantics must be spelled out, not left to the reader
import csv as _csv
check("coverage explains unsupported != success",
      "Only the `ok` column is a measurement" in REPORT
      and "not a fully measured stack" in REPORT,
      "semantics paragraph present")
check("non-ok cells are itemised with reasons",
      "### Every cell that produced no measurement" in REPORT,
      "itemised table present")
for reason in ("needs_2_nodes_CXI_has_no_loopback_fi_domain_ret_-38",
               "cxi_rma_write_unsupported"):
    check(f"reason '{reason[:34]}...' surfaced", reason in REPORT, "cited")

# the 'measured' percentage must equal ok/total from the CSV
for jn, sub, label in (("6956", "1node", "1 node"), ("6957", "2node", "2 nodes")):
    p = R.parent.parent / jn / sub / "sweep_results.csv"
    if p.exists():
        rr = list(_csv.DictReader(open(p)))
        pct = round(100.0 * sum(1 for x in rr if x["status"] == "ok") / len(rr))
        check(f"measured% for {label} matches CSV",
              re.search(rf"\| {label}[^|]*\| {jn} \|.*\| {pct}% \|", REPORT) is not None,
              f"CSV gives {pct}%")

# --- hardware provenance
check("hardware section present at both scales",
      REPORT.count("### Hardware actually used") == 2, "2 sections")
check("GH200 devices listed with PCI addresses",
      REPORT.count("NVIDIA GH200 120GB") >= 12
      and "0009:01:00" in REPORT, "rank->device->PCI map present")
check("per-layer GPU usage table present",
      REPORT.count("| layer | ranks | GPUs used | how it was launched |") == 2,
      "both scales")
check("layer 1 stated as using no GPU",
      "**0 GPUs**" in REPORT, "fi_pingpong is host-memory")
check("NIXL's 1-GPU-per-node geometry stated",
      "**1 GPU per node**" in REPORT, "cited")
check("aggregate-vs-point-to-point caveat present",
      "not evidence that NIXL is slower per GPU" in REPORT, "caveat present")

# hosts in the report must be the hosts in the logs
hosts_log = set(re.findall(r"on (x\d+c\d+s\d+b\d+n\d+) device",
                           (R.parent.parent / "6957/2node/nccl_coll_all_reduce.txt")
                           .read_text(errors="replace")))
check("2-node hosts match the NCCL log",
      hosts_log and all(h in REPORT for h in hosts_log),
      f"{sorted(hosts_log)}")

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
