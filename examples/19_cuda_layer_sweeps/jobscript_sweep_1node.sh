#!/bin/bash
#PBS -N sweep_1node
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=1:ngpus=4
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Per-layer option sweep, 1 node, all 4 GPUs.
#
# Layers are selected with DLCOMM_LAYERS so a known-broken layer can be
# excluded without editing the driver:
#   qsub -v DLCOMM_LAYERS=fi,osu,nccl,torch_dist jobscript_sweep_1node.sh
set -uo pipefail
W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
cd "$W"

module load cray-mpich/9.1.0 2>/dev/null
module load cuda/13.0 2>/dev/null
module unload darshan 2>/dev/null   # injects a lib that breaks every binary

# PALS sets TMPDIR=/var/run/palsd/<uuid>, which the job cannot write; anything
# that opens a temp file dies with PermissionError [Errno 13].
export TMPDIR=/tmp/${PBS_JOBID}
mkdir -p "$TMPDIR"

export DLCOMM_WORKDIR=$W
export DLCOMM_NNODES=1
export DLCOMM_GPUS_PER_NODE=4
export DLCOMM_RUN_DIR=$W/sweeps_${PBS_JOBID}/1node
export DLCOMM_LAYERS=${DLCOMM_LAYERS:-fi,osu,nccl,torch_dist,nixl}
export DLCOMM_QUICK=${DLCOMM_QUICK:-0}

echo "=== node inventory ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | head -6
echo "=== nodefile (raw order; rank 0 = first line) ==="
cat "$PBS_NODEFILE"
echo "=== image (regression check vs compute_south-setup_20260922T163916) ==="
cat /etc/modprobe.d/50-cxi-ss1.conf 2>/dev/null || echo "(no 50-cxi-ss1.conf)"
grep -o 'iommu[^ ]*' /proc/cmdline 2>/dev/null || true

bash "$W/sweep_layers_cuda.sh"
echo "JOBSCRIPT_EXIT=$?"
