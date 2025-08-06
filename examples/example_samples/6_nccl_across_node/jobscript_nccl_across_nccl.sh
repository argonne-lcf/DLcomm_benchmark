#!/bin/bash -l
#PBS -A datascience_collab
#PBS -l select=2:ncpus=256
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -q prod
#PBS -j oe

module use /soft/modulefiles
module load conda/2024-04-29
conda activate base

export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_SHM=1
export CCL_PROCESS_LAUNCHER=pmix
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export CCL_ENABLE_AUTO_CACHE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR/../.."
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"

if [ -n "$PBS_NODEFILE" ]; then
    NNODES=`wc -l < $PBS_NODEFILE`
else
    NNODES=1
fi

RANKS_PER_NODE=4
NRANKS=$(( NNODES * RANKS_PER_NODE ))

export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="8.0"
export CCL_LOG_LEVEL=info
export CCL_ENABLE_PROFILING=1
export TORCH_CPP_LOG_LEVEL=ERROR

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"

export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"

CONFIG_FILE=$(find "$SCRIPT_DIR" -name "*.yaml" -type f | head -1)
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)

mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --depth 16 \
        --cpu-bind depth \
        python3 -m dl_comm.dl_comm_main --config-path="$SCRIPT_DIR" --config-name="$CONFIG_NAME" 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}
exit $EXIT_STATUS