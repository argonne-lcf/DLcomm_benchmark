#!/bin/bash
# Count the cells the sweep will actually emit, by asking the driver itself
# rather than trusting the README. Uses the offline stub world so no cluster
# is involved: if these numbers disagree with the README, the README is wrong.
set -uo pipefail
cd /home/kvelusamy/Desktop/dlcomm-hardening/DLcomm_benchmark
/tmp/dlcvenv/bin/python - <<'PY'
import sys, csv, shutil
sys.path.insert(0, "tests")
from test_sweep_layers import SweepHarness

class Counter(SweepHarness):
    def runTest(self): pass

for nodes in ("1", "2"):
    h = Counter(); h.setUp()
    total = {}
    for layer in ("fi", "osu", "nccl", "torch_dist", "nixl"):
        shutil.rmtree(h.run, ignore_errors=True)
        _, rows = h.sweep(layer, nodes=nodes)
        total[layer] = len(rows)
    h.tearDown()
    print(f"--- {nodes} node(s) ---")
    for k, v in total.items():
        print(f"  {k:<12} {v:>3} cells")
    print(f"  {'TOTAL':<12} {sum(total.values()):>3} cells")
PY
