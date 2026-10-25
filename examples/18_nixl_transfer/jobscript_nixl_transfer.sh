#!/bin/bash
#PBS -N dlcomm_transfer_nixl
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -q workq
# -----------------------------------------------------------------------------
# NIXL transfer layer: h2d / d2h / d2d on NVIDIA (Tara, GH200).
#
# Submit:
#   qsub -A <account> -l select=1 jobscript_nixl_transfer.sh
#   qsub -A <account> -l select=4 jobscript_nixl_transfer.sh
#
# Why this exists next to 15_transfer_h2d_d2h_d2d: that example measures the
# same three directions through the framework's own copy path (CUDA/SYCL
# memcpy). This one measures them through NIXL, which is the path a
# disaggregated serving job actually uses -- registered memory, asynchronous
# transfer handles, a pluggable backend. The two numbers are expected to
# differ; the gap is the point of the comparison.
#
# Portability: nothing below is site-specific except the module list. The NIXL
# backend is NOT hardcoded -- measure_nixl() resolves it from the plugins the
# local build registered (UCX, then GDS, then POSIX). Override with
# --backend if you want a specific one.
# -----------------------------------------------------------------------------
set -euo pipefail

PROJ="${PROJ:-$PWD}"
REPO="${REPO:-$PROJ/DLcomm_benchmark}"
PYBIN="${PYBIN:-python3}"
RESULTS="${RESULTS:-$PROJ/results/nixl_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS"
LOG="$RESULTS/nixl_transfer.log"

# Node/rank counting. NOTE: `wc -l` is wrong on systems whose PBS nodefile has
# no trailing newline -- it returns nodes-1, and 0 for a single node, which
# then launches nothing. `grep -c .` counts the last line whether or not it
# ends in a newline.
if [ -n "${PBS_NODEFILE:-}" ]; then
    # grep -c . not `wc -l`: PBS nodefiles on some systems (Tara) have no
    # trailing newline, so wc -l undercounts by one and a 1-node allocation
    # builds `mpiexec --np 0`, which aborts with "Invalid -n/--np value 0".
    # NOTE: grep -c prints the count AND exits 1 when the count is 0, so a
    # `|| echo 1` fallback would emit two lines. Let it print 0 and let the
    # guard below reject it. See tools/nodefile.sh for the measurement.
    NNODES=$(grep -c . "$PBS_NODEFILE" 2>/dev/null || true)
    case "${NNODES:-}" in ""|*[!0-9]*) NNODES=0 ;; esac
    if [ "$NNODES" -lt 1 ]; then
        echo "FATAL: counted $NNODES nodes in $PBS_NODEFILE" >&2; exit 1
    fi
else
    NNODES=1
fi
[[ "$NNODES" -lt 1 ]] && NNODES=1
RANKS_PER_NODE="${RANKS_PER_NODE:-4}"
NRANKS=$(( NNODES * RANKS_PER_NODE ))

{
  echo "NODES=$NNODES RANKS_PER_NODE=$RANKS_PER_NODE NRANKS=$NRANKS"
  echo "REPO=$REPO"
  echo "DATE=$(date -Is)"
} | tee "$RESULTS/environment.txt"

export PYTHONPATH="${PYTHONPATH:-}:$REPO"

# NIXL moves device memory itself; it does not go through Cray MPICH's GPU
# path, so MPICH_GPU_SUPPORT_ENABLED must NOT be set for this step. Setting it
# globally in a multi-layer jobscript aborts any step whose binary is not
# GTL-linked.
mpiexec --np "$NRANKS" -ppn "$RANKS_PER_NODE" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$PYBIN" "$REPO/examples/18_nixl_transfer/run_nixl_transfer.py" \
        --bytes "${BYTES:-67108864}" \
        --iterations "${ITERATIONS:-10}" \
        --directions h2d,d2h,d2d \
    2>&1 | tee -a "$LOG"

EXIT_STATUS=${PIPESTATUS[0]}

# Exit 0 is not evidence that anything was measured: a run that skipped every
# direction also exits 0. Require records.
RECORDS=$(grep -c "^NIXL_RESULT" "$LOG" 2>/dev/null || true)
echo "NIXL_RECORDS=${RECORDS:-0}" | tee -a "$LOG"
if [[ "${RECORDS:-0}" -eq 0 ]]; then
    echo "VERDICT=NO_RECORDS (exit $EXIT_STATUS, nothing measured)" | tee -a "$LOG"
    exit 1
fi

# Each rank must own a distinct device. Ranks sharing one silently halve the
# device count and inflate per-device bandwidth.
DEVS=$(awk '/^NIXL_MAP /{print $2, $4}' "$LOG" | sort -u | wc -l)
echo "DISTINCT_HOST_DEV=$DEVS EXPECTED=$NRANKS" | tee -a "$LOG"
if [[ "$DEVS" -ne "$NRANKS" ]]; then
    echo "VERDICT=DEVICE_COLLAPSE (ranks sharing devices; numbers not trustworthy)" | tee -a "$LOG"
    exit 1
fi

echo "VERDICT=OK" | tee -a "$LOG"
exit "$EXIT_STATUS"
