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
CONFIG_NAME="config_nccl"

 
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
        python3 -m dl_comm.dl_comm_main --config-path="$EXAMPLES_DIR" --config-name="$CONFIG_NAME" 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}
exit $EXIT_STATUS