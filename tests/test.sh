#!/bin/bash -x
#PBS -A datascience_collab
#PBS -k doe
#PBS -l select=2:ncpus=208
#PBS -q dev
#PBS -l walltime=00:05:00
#PBS -l filesystems=flare
#PBS -j oe
#PBS -o /lus/flare/projects/datascience_collab/mcim/workspace/gpu-comm/DLcomm_benchmark/tests/pbs_job_${PBS_JOBID}.out

source /opt/aurora/24.347.0/oneapi/intel-conda-miniforge/etc/profile.d/conda.sh
conda activate /lus/flare/projects/datascience_collab/mcim/for-musa/sam_build/conda_pt2.8

module load frameworks

SCRIPT_DIR="/lus/flare/projects/datascience_collab/mcim/workspace/gpu-comm/DLcomm_benchmark/tests"
WORKDIR="$SCRIPT_DIR"

cd "$WORKDIR"

export PYTHONPATH="$WORKDIR/..:$PYTHONPATH"

NNODES=`wc -l < $PBS_NODEFILE`

RANKS_PER_NODE=4
NRANKS=$(( NNODES * RANKS_PER_NODE ))

CPU_BINDING="list:4:9:14:19"

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

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
TEST_LOG_DIR="$SCRIPT_DIR/logs/test_${RUN_TIMESTAMP}"

mkdir -p "$TEST_LOG_DIR"

PBS_OUTPUT_FILE="$SCRIPT_DIR/pbs_job_${PBS_JOBID}.out"
trap "if [[ -f '$PBS_OUTPUT_FILE' ]]; then mv '$PBS_OUTPUT_FILE' '$TEST_LOG_DIR/'; fi" EXIT

export TERMINAL_LOG_FILE="$TEST_LOG_DIR/terminal_output.log"

mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --cpu-bind ${CPU_BINDING} \
        bash -c "cd '$WORKDIR' && PYTHONPATH='$PYTHONPATH' python3 test.py" 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}

exit $EXIT_STATUS