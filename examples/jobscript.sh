#!/bin/bash -x
#PBS -A datascience_collab
#PBS -k doe
#PBS -l select=2:ncpus=208
#PBS -q debug
#PBS -l walltime=00:05:00
#PBS -l filesystems=flare
#PBS -j oe
#PBS -o /dev/null


module load frameworks

WORKDIR=$(pwd)
cd "$WORKDIR"

echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Working directory: $PBS_O_WORKDIR"
echo "Node list:"
cat $PBS_NODEFILE

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
RUN_LOG_DIR="$WORKDIR/logs/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"

 
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"


mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --cpu-bind ${CPU_BINDING} \
        python3 -m dl_comm.dl_comm_main 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}

echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"

 
exit $EXIT_STATUS