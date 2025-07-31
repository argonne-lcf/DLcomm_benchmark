#!/bin/bash
#PBS -A datascience_collab
#PBS -k doe
#PBS -l select=2:ncpus=208
#PBS -q debug
#PBS -l walltime=00:10:00
#PBS -l filesystems=flare
#PBS -j oe

source /opt/aurora/24.347.0/oneapi/intel-conda-miniforge/etc/profile.d/conda.sh
 

module load frameworks

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/../../../..:$PYTHONPATH"

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=4
NRANKS=$(( NNODES * RANKS_PER_NODE ))
CPU_BINDING="list:4:9:14:19"

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


mpiexec --np ${NRANKS} --ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING} python across.py