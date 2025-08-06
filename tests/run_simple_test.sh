#!/bin/bash

# Simple job script for NCCL algorithm override test
# Based on examples/jobscript_nccl.sh

# Load modules
module use /soft/modulefiles
module load conda/2024-04-29
conda activate base

# Get script directory and set up paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$SCRIPT_DIR"
WORKDIR="$TEST_DIR"
cd "$WORKDIR"

# Add parent directory to PYTHONPATH so Python can find dl_comm module
export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"

# Set number of ranks (adjust as needed)
NRANKS=2
RANKS_PER_NODE=2

# Explicitly disable GPU-aware MPI to avoid GTL library issue
export MPICH_GPU_SUPPORT_ENABLED=0

# Set up GPU environment
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Ensure PyTorch can find CUDA
export TORCH_CUDA_ARCH_LIST="8.0"

# NCCL environment variables for NVIDIA GPUs
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
 

# PyTorch settings for NCCL
export TORCH_CPP_LOG_LEVEL=ERROR

# Force GPU usage - this is critical for NCCL
export CUDA_LAUNCH_BLOCKING=1

# Disable Intel CCL environment variables
unset CCL_ATL_TRANSPORT
unset CCL_ATL_SHM
unset CCL_PROCESS_LAUNCHER
unset CCL_KVS_MODE
unset CCL_KVS_CONNECTION_TIMEOUT
unset CCL_OP_SYNC
unset CCL_ENABLE_AUTO_CACHE
unset PALS_PMI

# Create timestamped directory for this run
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$TEST_DIR/logs/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"

export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"

 

# Run the test
mpirun -np $NRANKS python3 simple_nccl_test.py 2>&1 | tee "$TERMINAL_LOG_FILE"
 