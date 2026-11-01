#!/usr/bin/env python3
"""Tamper-test verify_final_report.py: each edit MUST make it fail."""
import shutil
import subprocess
import sys
from pathlib import Path

FIN = Path("/home/kvelusamy/Desktop/tara-layer-sweeps-final")
DOC = FIN / "report/FINAL-REPORT.md"
BAK = Path("/tmp/fr_pristine.md")
PY = "/tmp/dlcvenv/bin/python"

shutil.copy(DOC, BAK)
orig = BAK.read_text()

TAMPERS = [
    ("Tree row clean -> 2/6",
     "| `NCCL_ALGO=Tree` | **0 / 6** |", "| `NCCL_ALGO=Tree` | **2 / 6** |"),
    ("1 MiB sizescan 3/6 -> 0/6",
     "| **1 MiB**, 8 ranks, CXI | **3 / 6** |", "| **1 MiB**, 8 ranks, CXI | **0 / 6** |"),
    ("inflate Tree bandwidth", "| algo | Tree | 115.33 |", "| algo | Tree | 155.33 |"),
    ("understate overall CoV", "overall CoV                   : 6.13%",
     "overall CoV                   : 2.13%"),
    ("hide the Socket control", "| 1 MiB, 8 ranks, **Socket**, default | **0 / 5** |",
     "| 1 MiB, 8 ranks, **Socket**, default | **4 / 5** |"),
    ("drop the correction", "That was wrong", "That was fine"),
    ("fake a sweep cell count", "**14 produced valid numbers; 5 failed validation.**",
     "**19 produced valid numbers; 0 failed validation.**"),
]

bad = []
for name, old, new in TAMPERS:
    if old not in orig:
        print(f"SKIP  {name}: anchor not found -- test itself is stale")
        bad.append(name)
        continue
    DOC.write_text(orig.replace(old, new, 1))
    r = subprocess.run([PY, "tools/verify_final_report.py"], cwd=FIN,
                       capture_output=True, text=True)
    caught = r.returncode != 0
    fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    print(f"{'CAUGHT' if caught else 'MISSED'}  {name}"
          + (f"  -> {fails[0].strip()}" if fails else ""))
    if not caught:
        bad.append(name)

DOC.write_text(orig)
r = subprocess.run([PY, "tools/verify_final_report.py"], cwd=FIN,
                   capture_output=True, text=True)
print(f"\nrestored: exit={r.returncode} "
      f"({sum(1 for l in r.stdout.splitlines() if l.startswith('PASS'))} PASS)")
if r.returncode != 0:
    print("RESTORE FAILED")
    sys.exit(2)
if bad:
    print(f"\n{len(bad)} TAMPER(S) NOT CAUGHT: {bad}")
    sys.exit(1)
print(f"\nall {len(TAMPERS)} tampers caught")
