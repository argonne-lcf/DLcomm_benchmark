#!/bin/bash -l
#PBS -A datascience_collab
#PBS -l select=2:ncpus=208
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -j oe
#PBS -o /dev/null

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
module use /soft/modulefiles
module load conda/2024-04-29
conda activate base

# ============================================================================
# DIRECTORY SETUP
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
fi
EXAMPLES_DIR="$SCRIPT_DIR/../.."
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
export CCL_PROCESS_LAUNCHER=pmix
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export CCL_ENABLE_AUTO_CACHE=1
# GPU-aware MPI is OFF for this step by default: these runs drive the GPUs
# through NCCL, not through MPI, so MPI only ever sees host buffers.
#
# Do NOT change this to =1 in place. Setting MPICH_GPU_SUPPORT_ENABLED=1
# without the Cray GTL actually loaded aborts every rank at MPI init with
# "MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not
# linked" -- loading craype-accel-nvidia90 is NOT sufficient for python
# ranks, because the GTL only gets linked when the cc wrapper builds the
# binary. If a step needs MPI on device buffers, source tools/gpu_aware_mpi.sh
# and call dlcomm_enable_gpu_aware_mpi, which LD_PRELOADs the GTL and puts its
# matching CUDA runtime on the path.
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="8.0"
export CCL_LOG_LEVEL=info
export CCL_ENABLE_PROFILING=1
export TORCH_CPP_LOG_LEVEL=ERROR

# ============================================================================
# LOG DIRECTORY SETUP
# ============================================================================
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
export RUN_LOG_DIR
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

# ============================================================================
# JOB EXECUTION
# ============================================================================
CONFIG_NAME="11_sp_nccl"
 

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

RANKS_PER_NODE=4
NRANKS=$(( NNODES * RANKS_PER_NODE ))

mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --depth 16 \
        --cpu-bind depth \
        python3 -m dl_comm.dl_comm_main --config-path="$SCRIPT_DIR" --config-name="$CONFIG_NAME" 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}
exit $EXIT_STATUS