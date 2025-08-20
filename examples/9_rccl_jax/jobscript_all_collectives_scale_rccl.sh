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

# ============================================================================
# DIRECTORY SETUP
# ============================================================================
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(pwd)"
fi

EXAMPLES_DIR="$SCRIPT_DIR/.."
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

echo Jobid: $SLURM_JOBID
echo $SLURM_NODELIST
echo Script Directory: $SCRIPT_DIR

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=/opt/cray/pe/python/3.11.7/:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/packs:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/DLcomm_benchmark
export LD_LIBRARY_PATH=/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib64:$LD_LIBRARY_PATH

# Clear proxy settings
unset http_proxy
unset https_proxy
unset ftp_proxy
unset all_proxy
unset no_proxy

# Network and communication settings
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
# LOG DIRECTORY SETUP
# ============================================================================
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S_%6N')"
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
export RUN_LOG_DIR
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

# ============================================================================
# JOB EXECUTION
# ============================================================================
CONFIG_FILE="$(find "$SCRIPT_DIR" -maxdepth 1 -name '*.yaml' -type f | head -1)"
CONFIG_NAME="$(basename "$CONFIG_FILE" .yaml)"

RANKS_PER_NODE=8
NRANKS=$(( SLURM_JOB_NUM_NODES * RANKS_PER_NODE ))

srun --ntasks=${NRANKS} --gpus-per-task=1 --cpus-per-task=1 --export=ALL \
  python3 -m dl_comm.dl_comm_main \
  --config-path="$SCRIPT_DIR" \
  --config-name="$CONFIG_NAME" \
  -- \
  hydra.run.dir="$RUN_LOG_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=False \
  2>&1 | tee "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}
exit $EXIT_STATUS