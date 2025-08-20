#!/bin/bash -x
#SBATCH -A GEN008
#SBATCH -J dlcomm
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --gpus-per-node=8

# ============================================================================
# JOB CONFIGURATION
# ============================================================================
RANKS_PER_NODE=8

# ============================================================================
# DIRECTORY SETUP
# ============================================================================
SCRIPT_DIR="$(pwd)"
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S_%6N')"
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
export RUN_LOG_DIR
TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
mkdir -p "$RUN_LOG_DIR"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
module load miniforge3/23.11.0-0
module load cray-python
module load rocm/6.2.4
module load cray-python

# Enable conda shell integration from the correct location
eval "$(/sw/frontier/miniforge3/23.11.0-0/bin/conda shell.bash hook)"

# Activate JAX environment
conda activate jax_env-frontier

export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=/opt/cray/pe/python/3.11.7/:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/packs:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/DLcomm_benchmark
export LD_LIBRARY_PATH=/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib64:$LD_LIBRARY_PATH

 

# ============================================================================
# COMMUNICATION SETTINGS
# ============================================================================
export MASTER_ADDR=$(ip -4 addr show dev hsn0 | awk '/inet/{print $2}' | cut -d/ -f1)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_CPP_LOG_LEVEL=ERROR

# JAX specific settings
export JAX_PLATFORMS=rocm

# Coordinator settings for JAX
export COORDINATOR_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export COORDINATOR_PORT=1234

# ============================================================================
# JOB EXECUTION
# ============================================================================
srun --ntasks=16 --gpus-per-task=1 --cpus-per-task=1 --export=ALL \
  python3 -m dl_comm.dl_comm_main \
  --config-path="$SCRIPT_DIR" \
  --config-name=9_rccl_jax \
  -- \
  hydra.run.dir="$RUN_LOG_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=False \
  > "$TERMINAL_LOG_FILE" 2>&1