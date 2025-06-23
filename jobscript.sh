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


 
RANKS_PER_NODE=6
NRANKS=$(( NNODES * RANKS_PER_NODE ))

 
CPU_BINDING="list:4:9:14:19:20:25:56:61:66:71:74:79"
 
#export CCL_LOG_LEVEL=debug
export FI_MR_CACHE_MONITOR=userfaultfd
 
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