#!/bin/bash -x
#PBS -A datascience
#PBS -k doe
#PBS -l select=2:ncpus=208
#PBS -q debug-scaling
#PBS -l walltime=00:05:00
#PBS -l filesystems=flare
#PBS -j oe

# Paths are derived from the submission directory rather than hardcoded, so the
# script runs from any checkout. It previously pointed at another user's
# workspace and activated a conda env under their directory, which made it
# unrunnable for anyone else (docs/fixes/06-packaging.md).

set -uo pipefail

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKDIR="$SCRIPT_DIR"
cd "$WORKDIR"

# The frameworks module supplies torch and oneCCL. No external conda env: the
# previous one lived in a directory this project does not own.
set +u
module load frameworks
set -u

export PYTHONPATH="$WORKDIR/..:${PYTHONPATH:-}"

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=4
NRANKS=$(( NNODES * RANKS_PER_NODE ))
CPU_BINDING="list:4:9:14:19"

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

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

PBS_OUTPUT_FILE="$SCRIPT_DIR/pbs_job_${PBS_JOBID:-local}.out"
trap "if [[ -f '$PBS_OUTPUT_FILE' ]]; then mv '$PBS_OUTPUT_FILE' '$TEST_LOG_DIR/'; fi" EXIT

export TERMINAL_LOG_FILE="$TEST_LOG_DIR/terminal_output.log"

# Absolute PALS launcher: an activated env can put a different mpiexec ahead of
# it on PATH, which silently degrades the job to a single rank.
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
[[ -x "$MPIEXEC" ]] || MPIEXEC=$(command -v mpiexec)

"$MPIEXEC" --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --cpu-bind ${CPU_BINDING} \
        bash -c "cd '$WORKDIR' && PYTHONPATH='$PYTHONPATH' python3 test.py" 2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}

exit $EXIT_STATUS
