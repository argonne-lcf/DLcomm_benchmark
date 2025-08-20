#!/bin/bash -l
#PBS -A datascience_collab
#PBS -l select=2:ncpus=256
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -q prod
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
if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
EXAMPLES_DIR="$SCRIPT_DIR"
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="8.0"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_ALGO=Ring
export TORCH_CPP_LOG_LEVEL=ERROR
export CUDA_LAUNCH_BLOCKING=1
unset CCL_ATL_TRANSPORT
unset CCL_ATL_SHM
unset CCL_PROCESS_LAUNCHER
unset CCL_KVS_MODE
unset CCL_KVS_CONNECTION_TIMEOUT
unset CCL_OP_SYNC
unset CCL_ENABLE_AUTO_CACHE
unset PALS_PMI

# ============================================================================
# LOG DIRECTORY SETUP
# ============================================================================
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$EXAMPLES_DIR/logs/run_${RUN_TIMESTAMP}"
export RUN_LOG_DIR
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

# ============================================================================
# JOB EXECUTION
# ============================================================================
if [ -n "$PBS_NODEFILE" ]; then
    NNODES=`wc -l < $PBS_NODEFILE`
else
    NNODES=1
fi

RANKS_PER_NODE=4
NRANKS=$(( NNODES * RANKS_PER_NODE ))

mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --depth 16 \
        --cpu-bind depth \
        python3 -m dl_comm.dl_comm_main --config-path="$EXAMPLES_DIR" --config-name=config 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}
exit $EXIT_STATUS