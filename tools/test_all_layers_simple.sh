#!/usr/bin/env bash
# test_all_layers_simple.sh -- the smallest end-to-end check of all six layers.
#
# Purpose: answer one question quickly, on any machine, with no GPUs and no
# cluster: "does every layer still parse, score, and refuse correctly?"
#
# This is NOT a performance run and claims no hardware result. It drives each
# layer's REAL scoring path with tiny synthetic fixtures -- one clean and one
# corrupt per layer -- and asserts the clean one is accepted and the corrupt
# one is refused. A layer that cannot refuse is the fail-open class of defect
# this project keeps finding, so that is exactly what gets checked.
#
# Usage:  bash tools/test_all_layers_simple.sh
# Exit 0 = every layer accepted its clean fixture and refused its corrupt one.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP="$ROOT/examples/21_cuda_layer_sweeps/sweep_layers_cuda.sh"
PY="${PYTHON:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
check() { # check <desc> <expected> <actual>
    if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (expected '$2', got '$3')"; fi
}

# Source the shipped scoring functions rather than reimplementing them.
GATES="$WORK/gates.sh"
sed -n "/^validation_failed()/,/^# csv <layer>/p" "$SWEEP" > "$GATES"
# shellcheck disable=SC1090
. "$GATES"

echo "== layer 1: fi (fabric provider inventory) =="
printf 'provider: cxi\n  fabric: cxi\n' > "$WORK/fi_ok.txt"
score_cell "$WORK/fi_ok.txt" 0 1 1; check "clean fi inventory accepted" ok "$ST"
printf 'provider: cxi\nfi_getinfo: -61\nError: runtime failure\n' > "$WORK/fi_bad.txt"
score_cell "$WORK/fi_bad.txt" 0 1 1; check "fi log with runtime error refused" failed "$ST"

echo "== layer 2: osu (point-to-point) =="
printf '# OSU MPI Latency Test\n1 1.23\n2 1.25\n' > "$WORK/osu_ok.txt"
score_cell "$WORK/osu_ok.txt" 0 2 1; check "clean osu accepted" ok "$ST"
printf '# OSU MPI Latency Test\n1 1.23\nSegmentation fault\n' > "$WORK/osu_bad.txt"
score_cell "$WORK/osu_bad.txt" 139 1 1; check "osu segfault refused" failed "$ST"

echo "== layer 3: nccl (collectives) =="
NHDR='#   size count type redop root time algbw busbw #wrong time algbw busbw #wrong'
NROW='     1048576 262144 float sum -1 12.3 85.2 85.2 0 12.4 84.9 84.9 0'
NSUM='# Out of bounds values : 0 OK
# Avg bus bandwidth : 85.2'
printf '%s\n%s\n%s\n' "$NHDR" "$NROW" "$NSUM" > "$WORK/nccl_ok.txt"
W=$(nccl_rows_bad "$WORK/nccl_ok.txt"); V=$(nccl_no_verdict "$WORK/nccl_ok.txt")
check "clean nccl rows accepted" "" "$W$V"
# field 9 = out-of-place wrong count
BROW='     1048576 262144 float sum -1 12.3 85.2 85.2 7 12.4 84.9 84.9 0'
printf '%s\n%s\n%s\n' "$NHDR" "$BROW" "$NSUM" > "$WORK/nccl_bad.txt"
W=$(nccl_rows_bad "$WORK/nccl_bad.txt")
check "nccl wrong-element row refused" "nccl_wrong_oop_7" "$W"
printf '%s\n%s\n' "$NHDR" "$NROW" > "$WORK/nccl_trunc.txt"
V=$(nccl_no_verdict "$WORK/nccl_trunc.txt")
check "nccl run with no summary refused" "nccl_no_summary_verdict" "$V"

echo "== layer 4: torch (torchcomms / dist) =="
printf 'allreduce 1048576 12.3 us\nVERIFIED=1\n' > "$WORK/torch_ok.txt"
score_cell "$WORK/torch_ok.txt" 0 1 1; check "clean torch accepted" ok "$ST"
printf 'allreduce 1048576 12.3 us\nSTATUS=FAILED correctness_failures=3\n' \
    > "$WORK/torch_bad.txt"
score_cell "$WORK/torch_bad.txt" 0 1 1
check "torch correctness failure refused" failed "$ST"

echo "== layer 5: nixl (RDMA) =="
printf 'READ 4194304 23.99 GB/s\nbyte-exact check: PASS\n' > "$WORK/nixl_ok.txt"
score_cell "$WORK/nixl_ok.txt" 0 1 1; check "clean nixl accepted" ok "$ST"
printf 'READ 4194304 23.99 GB/s\nbyte-exact check: FAIL\n' > "$WORK/nixl_bad.txt"
score_cell "$WORK/nixl_bad.txt" 0 1 1; check "nixl byte-exact FAIL refused" failed "$ST"
# corruption must dominate a capability exemption
printf 'WRITE 4194304\nNIXL_ERR_BACKEND\nMISMATCH at offset 0\n' > "$WORK/nixl_cap.txt"
score_cell "$WORK/nixl_cap.txt" 0 1 1
check "nixl corruption dominates capability exemption" failed "$ST"

echo "== layer 6: profile (report joins) =="
# build_report.py reads sys.argv[1:3] at module scope and ends in a bare
# main() call, so it is loaded as source up to that call, with argv padded.
"$PY" - "$ROOT" "$WORK/in" "$WORK/out.md" <<'PYEOF'
import sys, types
from pathlib import Path
root = Path(sys.argv[1])
src = (root / "tools/build_report.py").read_text()
src = src[:src.rindex("\nmain()")]
m = types.ModuleType("br"); m.__file__ = str(root / "tools/build_report.py")
exec(compile(src, m.__file__, "exec"), m.__dict__)

fails = []
def chk(desc, got, want):
    if got != want:
        fails.append(f"{desc}: expected {want}, got {got}")
        print(f"  \033[31mFAIL\033[0m  {desc}")
    else:
        print(f"  \033[32mPASS\033[0m  {desc}")

rows = [
    {"layer": "nccl",  "cell": "coll_alltoall",  "status": "failed"},
    {"layer": "torch", "cell": "coll_alltoall",  "status": "ok"},
    {"layer": "osu",   "cell": "reduce_host",    "status": "failed"},
    {"layer": "osu",   "cell": "allreduce_host", "status": "ok"},
]
chk("report join accepts a passing cell",
    m.cell_ok(rows, "torch_coll_alltoall.txt", "torch", "coll_alltoall"), True)
chk("report join refuses a failed cell",
    m.cell_ok(rows, "nccl_coll_alltoall.txt", "nccl", "coll_alltoall"), False)
chk("report join refuses an unrecorded cell",
    m.cell_ok(rows, "nccl_ghost.txt", "nccl", "ghost"), False)
chk("report join is not fooled by a substring sibling",
    m.cell_ok(rows, "osu_allreduce_host.txt", "osu", "allreduce_host"), True)
sys.exit(1 if fails else 0)
PYEOF
if [ $? -eq 0 ]; then PASS=$((PASS+4)); else FAIL=$((FAIL+1)); fi

echo
echo "layers exercised: fi, osu, nccl, torch, nixl, profile"
echo "TOTAL: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
echo "NOTE: synthetic fixtures only -- this proves the gates refuse bad input,"
echo "      not that any hardware measurement is correct."
