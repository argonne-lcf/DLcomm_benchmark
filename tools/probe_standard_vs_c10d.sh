#!/bin/bash
#PBS -N tc_std_vs_c10d
#PBS -l select=1:ncpus=208
#PBS -l walltime=00:15:00
#PBS -q debug-scaling
#PBS -l filesystems=flare
#PBS -A datascience
#PBS -k doe

# Compare torchcomms' native TorchComm API against c10d for reductions.
# See docs/findings/05-torchcomms-standard-vs-c10d.md.
#
# Submit from the repository root:
#   qsub tools/probe_standard_vs_c10d.sh
#
# The exit status is not a reliable verdict: this stack has a teardown race
# (docs/torchcomms-stack.md). Read the printed table.

set -uo pipefail
cd "$PBS_O_WORKDIR"

set +u
module load frameworks
set -u

STACK=/lus/flare/projects/datascience/kaushik/stacks/torchcomms_0.3.0
export PYTHONPATH="$STACK/torchcomms:$STACK/pytorch:${PYTHONPATH:-}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ATL_TRANSPORT=mpi
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export CCL_KVS_MODE=mpi
export PALS_PMI=pmix
export TORCH_CPP_LOG_LEVEL=error
export FI_MR_CACHE_MONITOR=userfaultfd

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# Absolute PALS launcher: a `mpiexec` from PATH may come from an activated
# environment rather than from PALS, which silently yields single-rank runs.
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec

export MASTER_ADDR=$(head -1 "$PBS_NODEFILE")
export MASTER_PORT=29513
export WORLD_SIZE=$NRANKS

echo "NNODES=$NNODES NRANKS=$NRANKS"
python -c "import torch, torchcomms; print('torch', torch.__version__); print('tc', torchcomms.__file__)"

"$MPIEXEC" --np ${NRANKS} --ppn ${RANKS_PER_NODE} \
    --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100 \
    -genv RANK_OFFSET=0 \
    bash -c 'export RANK=$PALS_RANKID; export LOCAL_RANK=$PALS_LOCAL_RANKID; exec python tools/probe_standard_vs_c10d.py'
