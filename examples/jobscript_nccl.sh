#!/bin/bash -l
#PBS -A datascience_collab
#PBS -l select=2:ncpus=256
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -j oe

# Load modules
module use /soft/modulefiles
module load conda/2024-04-29
conda activate base


# Use the directory where the script is located (portable for any user)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR"
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

# Add parent directory to PYTHONPATH so Python can find dl_comm module
export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"



if [ -n "$PBS_NODEFILE" ]; then
    NNODES=`wc -l < $PBS_NODEFILE`
else
    NNODES=1  # Default to 1 node when running directly
fi

 
# Polaris has 4 NVIDIA A100 GPUs per node
RANKS_PER_NODE=4  # 4 GPUs per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# Explicitly disable GPU-aware MPI to avoid GTL library issue
export MPICH_GPU_SUPPORT_ENABLED=0

# Set up GPU environment
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Ensure PyTorch can find CUDA
export TORCH_CUDA_ARCH_LIST="8.0"

# NCCL environment variables for NVIDIA GPUs
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_ALGO=Ring

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
RUN_LOG_DIR="$EXAMPLES_DIR/logs/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"

 
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"

# Create a config file for NCCL backend
CONFIG_FILE="$RUN_LOG_DIR/config_nccl.yaml"
sed 's/ccl_backend : ccl/ccl_backend : nccl/' "$EXAMPLES_DIR/config.yaml" > "$CONFIG_FILE"

# Use mpiexec according to Polaris documentation
mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --depth 16 \
        --cpu-bind depth \
        python3 -m dl_comm.dl_comm_main --config-path="$RUN_LOG_DIR" --config-name=config_nccl 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}


 
exit $EXIT_STATUS