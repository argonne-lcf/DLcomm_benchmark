#!/bin/bash -x
#SBATCH -A GEN008
#SBATCH -J jax_allreduce
#SBATCH -N 2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -t 00:05:00
#SBATCH -p batch

module load miniforge3/23.11.0-0
module load cray-python
module load rocm/6.2.4

eval "$(/sw/frontier/miniforge3/23.11.0-0/bin/conda shell.bash hook)"
conda activate jax_env-frontier

export PYTHONPATH=/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/packs:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/DLcomm_benchmark
export LD_LIBRARY_PATH=/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib64:$LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=rocm

# Set rank 0 node as the coordinator
export COORDINATOR_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)

srun --ntasks=16 --gpus-per-task=1 --cpus-per-task=1 --export=ALL \
    python example.py
