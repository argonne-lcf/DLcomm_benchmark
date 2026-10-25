#!/bin/bash -x
#PBS -A datascience
#PBS -k doe
#PBS -l select=1:ncpus=208
#PBS -q debug-scaling
#PBS -l walltime=00:20:00
#PBS -l filesystems=flare
#PBS -j oe
#PBS -o /dev/null

# ============================================================================
# Point-to-point and vector collectives on one node, 12 XPU ranks.
#
# sendrecv_async is the reason this example has a watchdog: it deadlocks on
# the torchcomms 0.1.0 stack shipped with frameworks/2025.3.1. Without
# DLCOMM_WATCHDOG the job burns its full walltime and produces no stack.
# ============================================================================

module load frameworks

if [[ -n "${PBS_O_WORKDIR:-}" && "${PBS_ENVIRONMENT:-}" == "PBS_BATCH" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SRC="${BASH_SOURCE[0]:-$0}"
    SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd -P)"
fi

EXAMPLES_DIR="$SCRIPT_DIR/.."
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

# ============================================================================
# ENVIRONMENT
#
# CCL_ATL_TRANSPORT=mpi is required on Aurora; the ofi transport does not
# work with this stack. The four FI_CXI settings are the ALCF-recommended
# oneCCL set -- see docs/fixes/24.
# ============================================================================
export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"
export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_SHM=0
export CCL_PROCESS_LAUNCHER=pmix
export TORCH_CPP_LOG_LEVEL=error
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export PALS_PMI=pmix
export CCL_OP_SYNC=1
export CCL_ENABLE_AUTO_CACHE=0
export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=30

# Wall-clock budget. On expiry every rank's Python stack is dumped via
# faulthandler and the job exits non-zero, so a hang leaves evidence.
export DLCOMM_WATCHDOG=600

# Bound the final cross-rank verdict reduction (fix 11).
export DLCOMM_VERDICT_TIMEOUT=120

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
export RUN_LOG_DIR
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

CONFIG_NAME="14_p2p_and_vector_xccl"

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
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# --pmi=pmix is required for torch.distributed to bootstrap on Aurora, and
# --envall propagates the CCL/FI settings above to every rank (fix 25).
mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --depth 16 \
        --cpu-bind depth \
        --pmi=pmix \
        --envall \
        python3 -m dl_comm.dl_comm_main --config-path="$SCRIPT_DIR" --config-name="$CONFIG_NAME" 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}
exit $EXIT_STATUS
