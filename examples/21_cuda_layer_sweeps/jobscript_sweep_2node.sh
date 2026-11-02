#!/bin/bash
#PBS -N sweep_2node
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Per-layer option sweep, 2 nodes, all 8 GPUs.
#
# Same driver as the 1-node job; only DLCOMM_NNODES changes. Keeping one
# driver means a 1-node and a 2-node result are directly comparable rather
# than being two scripts that have drifted apart.
set -uo pipefail
W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
cd "$W"

module load cray-mpich/9.1.0 2>/dev/null
module load cuda/13.0 2>/dev/null
module unload darshan 2>/dev/null

export TMPDIR=/tmp/${PBS_JOBID}
mkdir -p "$TMPDIR"

export DLCOMM_WORKDIR=$W
export DLCOMM_NNODES=2
export DLCOMM_GPUS_PER_NODE=4
export DLCOMM_RUN_DIR=$W/sweeps_${PBS_JOBID}/2node
export DLCOMM_LAYERS=${DLCOMM_LAYERS:-fi,osu,nccl,torch_dist,nixl}
export DLCOMM_QUICK=${DLCOMM_QUICK:-0}

echo "=== nodefile (raw order; rank 0 = FIRST line, never sorted) ==="
cat "$PBS_NODEFILE"
echo "=== per-node image check ==="
MPIEXEC=$(command -v mpiexec)
$MPIEXEC -n 2 -ppn 1 bash -c \
  'echo "$(hostname -s): $(cat /etc/modprobe.d/50-cxi-ss1.conf 2>/dev/null | tr -d "\n")"' 2>&1 | head -4

bash "$W/sweep_layers_cuda.sh"
echo "JOBSCRIPT_EXIT=$?"
