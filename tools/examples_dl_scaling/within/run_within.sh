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

if [ -n "${PBS_NODEFILE:-}" ]; then
    # grep -c . not `wc -l`: PBS nodefiles on some systems (Tara) have no
    # trailing newline, so wc -l undercounts by one and a 1-node allocation
    # builds `mpiexec --np 0`, which aborts with "Invalid -n/--np value 0".
    # NOTE: grep -c prints the count AND exits 1 when the count is 0, so a
    # `|| echo 1` fallback would emit two lines. Let it print 0 and let the
    # guard below reject it. See tools/nodefile.sh for the measurement.
    NNODES=$(grep -c . "$PBS_NODEFILE" 2>/dev/null || true)
    case "${NNODES:-}" in ""|*[!0-9]*) NNODES=0 ;; esac
    if [ "$NNODES" -lt 1 ]; then
        echo "FATAL: counted $NNODES nodes in $PBS_NODEFILE" >&2; exit 1
    fi
else
    NNODES=1
fi
RANKS_PER_NODE=4
NRANKS=$(( NNODES * RANKS_PER_NODE ))
CPU_BINDING="list:4:9:14:19"

export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_SHM=0
export CCL_PROCESS_LAUNCHER=pmix
export CCL_KVS_MODE=mpi
export PALS_PMI=pmix

# Resolve the interpreter explicitly instead of letting mpiexec inherit a bare
# `python` from PATH. On tara the system python3 is 3.6.15, which predates
# `from __future__ import annotations` and has no mpi4py, so a bare name there
# fails on every rank before any measurement runs. Honour PYTHON_BIN if the
# caller set one; otherwise take whatever the conda activate above selected,
# resolved to an absolute path now so each rank runs the same binary.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: no python interpreter found; set PYTHON_BIN" >&2
    exit 1
fi
echo "using interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
mpiexec --np ${NRANKS} --ppn ${RANKS_PER_NODE} --cpu-bind ${CPU_BINDING} "$PYTHON_BIN" within.py