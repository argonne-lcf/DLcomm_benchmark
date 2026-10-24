#!/bin/bash
#PBS -N dlcomm_nixl_remote
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q workq
# -----------------------------------------------------------------------------
# Cross-node NIXL RDMA: one rank writes directly into a remote rank's VRAM.
#
# Submit:
#   qsub -A <account> -l select=2 jobscript_nixl_remote.sh
#   qsub -A <account> -l select=4 jobscript_nixl_remote.sh
#
# Why this exists next to 18_nixl_transfer: that example measures NIXL between
# buffers on ONE node. Every transfer there is node-local, so a 4-node run is
# just 4 independent node-local sets and the fabric never appears. This example
# is the one that crosses it -- the transfer a disaggregated serving job
# performs when a prefill node hands KV cache to a decode node.
#
# READ THIS BEFORE TRUSTING A NUMBER FROM THIS EXAMPLE (measured on Tara,
# Slingshot/GH200, 2026-10-24): NIXL's pip-installed UCX bundles only the
# ib/mlx5/efa/cma/cuda transport providers. On a Slingshot fabric (/dev/cxi*,
# no Mellanox IB device) none of them can drive the NIC, so UCX falls back to
# TCP over the hsn IP interfaces and cross-node NIXL measures ~0.2 GB/s while
# MPI between the SAME TWO NODES measures 12-20 GB/s. That ~50x gap is a
# transport-availability problem in the NIXL build, not a property of the
# fabric and not something this benchmark can tune around. Step 1 below runs
# the MPI baseline every time precisely so the comparison is in the log rather
# than in someone's memory.
# -----------------------------------------------------------------------------
set -uo pipefail

PROJ="${PROJ:-$PWD}"
REPO="${REPO:-$PROJ/DLcomm_benchmark}"

# Activate the project venv unless the caller already supplied an interpreter.
# PBS starts a login shell whose `python3` on this system is old enough to
# reject `from __future__ import annotations`, so every rank dies with a
# SyntaxError before importing anything -- a failure that looks like broken
# code rather than a missing venv. VENV defaults to the layout used by the
# other examples; override it or PYBIN if yours differs.
VENV="${VENV:-$PROJ/tcvenv}"
if [[ -z "${PYBIN:-}" ]]; then
    if [[ -x "$VENV/bin/python3" ]]; then
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        PYBIN="$VENV/bin/python3"
    else
        PYBIN="python3"
    fi
fi

# Fail loudly and early rather than letting every rank emit the same
# SyntaxError, which buries the real cause under 8 identical tracebacks.
if ! "$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "VERDICT=BAD_PYTHON ($PYBIN is $("$PYBIN" --version 2>&1); need >= 3.8 " \
         "-- set VENV=/path/to/venv or PYBIN=/path/to/python3)"
    exit 1
fi
RESULTS="${RESULTS:-$PROJ/results/nixl_remote_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS"
LOG="$RESULTS/nixl_remote.log"

# grep -c . not wc -l: a PBS nodefile with no trailing newline makes wc -l
# return nodes-1, and 0 for a single node, which then launches nothing.
NNODES=$(grep -c . "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)
[[ "$NNODES" -lt 1 ]] && NNODES=1
RANKS_PER_NODE="${RANKS_PER_NODE:-4}"
NRANKS=$(( NNODES * RANKS_PER_NODE ))

{
  echo "NODES=$NNODES RANKS_PER_NODE=$RANKS_PER_NODE NRANKS=$NRANKS"
  echo "REPO=$REPO"
  echo "DATE=$(date -Is)"
  echo "NODELIST=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | tr '\n' ' ')"
} | tee "$RESULTS/environment.txt"

if [[ "$NNODES" -lt 2 ]]; then
    echo "VERDICT=SINGLE_NODE (this example measures the fabric; submit -l select=2 or more)" | tee -a "$LOG"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:$REPO"

# ---------------------------------------------------------------------------
# Step 1: fabric baseline through MPI, NOT NIXL.
# Without this, a slow NIXL number is unattributable -- it could be the fabric,
# the placement, or NIXL. With it, the comparison is on the same two nodes in
# the same allocation, which is the only version of the claim worth making.
# ---------------------------------------------------------------------------
echo "=== MPI fabric baseline (same nodes, same allocation) ===" | tee -a "$LOG"
MPICH_GPU_SUPPORT_ENABLED=0 mpiexec --np 2 -ppn 1 \
    --env PYTHONPATH="$PYTHONPATH" \
    "$PYBIN" "$REPO/examples/19_nixl_remote_rdma/mpi_fabric_baseline.py" \
    2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Step 2: the NIXL cross-node measurement.
# NIXL moves device memory itself and does not use Cray MPICH's GPU path, so
# MPICH_GPU_SUPPORT_ENABLED must NOT be set here -- setting it globally aborts
# any step whose binary is not GTL-linked.
# ---------------------------------------------------------------------------
echo "=== NIXL cross-node RDMA ===" | tee -a "$LOG"
mpiexec --np "$NRANKS" -ppn "$RANKS_PER_NODE" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$PYBIN" "$REPO/examples/19_nixl_remote_rdma/run_nixl_remote.py" \
        --sizes "${SIZES:-1048576,16777216,67108864}" \
        --iterations "${ITERATIONS:-10}" \
        --pairing "${PAIRING:-split}" \
        --ranks-per-node "$RANKS_PER_NODE" \
    2>&1 | tee -a "$LOG"

EXIT_STATUS=${PIPESTATUS[0]}

# --- Gates. Exit 0 is not evidence: a run that skipped everything also exits 0.
RECORDS=$(grep -c "^NIXL_REMOTE_RESULT" "$LOG" 2>/dev/null || true)
echo "REMOTE_RECORDS=${RECORDS:-0}" | tee -a "$LOG"
if [[ "${RECORDS:-0}" -eq 0 ]]; then
    echo "VERDICT=NO_RECORDS (exit $EXIT_STATUS, nothing measured)" | tee -a "$LOG"
    exit 1
fi

# Corruption is a hard failure, never a slow-but-usable result.
if grep -q "^NIXL_REMOTE_CORRUPT" "$LOG"; then
    echo "VERDICT=CORRUPT (transfer completed but delivered wrong bytes)" | tee -a "$LOG"
    exit 1
fi

# Every result must be verified. An unverified row is a bandwidth number for a
# transfer nobody checked actually moved the sender's data.
UNVERIFIED=$(awk '/^NIXL_REMOTE_RESULT/ && $NF != "True"' "$LOG" | wc -l)
echo "UNVERIFIED_ROWS=$UNVERIFIED" | tee -a "$LOG"
if [[ "$UNVERIFIED" -ne 0 ]]; then
    echo "VERDICT=UNVERIFIED (byte-level check did not pass on every row)" | tee -a "$LOG"
    exit 1
fi

# Every pairing must actually cross a node boundary, or this measured the wrong
# thing while looking entirely healthy.
SAMENODE=$(awk '/^NIXL_RMAP/ && $NF != "True"' "$LOG" | wc -l)
echo "SAME_NODE_PAIRS=$SAMENODE EXPECTED=0" | tee -a "$LOG"
if [[ "$SAMENODE" -ne 0 ]]; then
    echo "VERDICT=NOT_CROSS_NODE (pairing degenerated to same-node traffic)" | tee -a "$LOG"
    exit 1
fi

# Each rank must own a distinct device; ranks sharing one inflate per-device
# bandwidth while every other check still passes.
DEVS=$(awk '/^NIXL_RMAP /{print $2, $6}' "$LOG" | sort -u | wc -l)
echo "DISTINCT_HOST_DEV=$DEVS EXPECTED=$NRANKS" | tee -a "$LOG"
if [[ "$DEVS" -ne "$NRANKS" ]]; then
    echo "VERDICT=DEVICE_COLLAPSE (ranks sharing devices; numbers not trustworthy)" | tee -a "$LOG"
    exit 1
fi

# Report the gap against the MPI baseline rather than leaving it to the reader.
BASE=$(awk '/^MPI_BASELINE/ {print $NF}' "$LOG" | tail -1 | sed 's/oneway_GB\/s=//')
NIXLBW=$(awk '/^NIXL_REMOTE_RESULT/ {print $5}' "$LOG" | sort -g | tail -1)
echo "FABRIC_BASELINE_GBPS=${BASE:-unknown} NIXL_BEST_GBPS=${NIXLBW:-unknown}" | tee -a "$LOG"

echo "VERDICT=OK" | tee -a "$LOG"
exit "$EXIT_STATUS"
