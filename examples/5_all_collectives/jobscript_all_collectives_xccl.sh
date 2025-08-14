#!/bin/bash -x
#PBS -A datascience_collab
#PBS -k doe
#PBS -l select=2:ncpus=208
#PBS -q debug
#PBS -l walltime=00:05:00
#PBS -l filesystems=flare
#PBS -j oe
#PBS -o /dev/null


# Activate PyTorch 2.8 environment
source /opt/aurora/24.347.0/oneapi/intel-conda-miniforge/etc/profile.d/conda.sh
conda activate /lus/flare/projects/datascience_collab/mcim/for-musa/sam_build/conda_pt2.8

# Load frameworks after conda to ensure missing modules are available
module load frameworks


echo $PYTHONUSERBASE



if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
EXAMPLES_DIR="$SCRIPT_DIR/../.."
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

# 
export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"



NNODES=`wc -l < $PBS_NODEFILE`


 
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))

 
CPU_BINDING="list:4:9:14:19:20:25:56:61:66:71:74:79"
 
# Critical CCL environment variables 
 
export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_SHM=0
export CCL_PROCESS_LAUNCHER=pmix
export TORCH_CPP_LOG_LEVEL=error
export FI_MR_CACHE_MONITOR=userfaultfd

#export CCL_LOG_LEVEL=debug

export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600 
export PALS_PMI=pmix # Required by Aurora mpich
 

export CCL_OP_SYNC=1
export CCL_ENABLE_AUTO_CACHE=0

export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_RX_MATCH_MODE=hybrid
 
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=30


# Create timestamped directory for this run
RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"

 
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"

CONFIG_FILE=$(find "$SCRIPT_DIR" -name "*.yaml" -type f | head -1)
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)

mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --cpu-bind ${CPU_BINDING} \
        python3 -m dl_comm.dl_comm_main --config-path="$SCRIPT_DIR" --config-name="$CONFIG_NAME" 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}


 
exit $EXIT_STATUS