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

export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_SHM=0
export CCL_PROCESS_LAUNCHER=pmix
export CCL_KVS_MODE=mpi
export PALS_PMI=pmix

mpiexec --np ${NRANKS} --ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING} python within.py