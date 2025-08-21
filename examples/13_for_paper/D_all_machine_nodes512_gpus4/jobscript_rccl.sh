#!/bin/bash -x
#SBATCH -A GEN008
#SBATCH -J dlcomm
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 512
#SBATCH --gpus-per-node=4

# ============================================================================
# JOB CONFIGURATION
# ============================================================================
RANKS_PER_NODE=4

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
export PYTHONPATH=/opt/cray/pe/python/3.11.7/:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/packs:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/dlcomm-pip:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/DLcomm_benchmark

# ============================================================================
# COMMUNICATION SETTINGS
# ============================================================================
export MASTER_ADDR=$(ip -4 addr show dev hsn0 | awk '/inet/{print $2}' | cut -d/ -f1)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_CPP_LOG_LEVEL=ERROR    

# ============================================================================
# JOB EXECUTION
# ============================================================================
srun --ntasks=2048 --export=ALL --cpu-bind=threads \
  python3 -m dl_comm.dl_comm_main \
  --config-path="$SCRIPT_DIR" \
  --config-name=rccl \
  -- \
  hydra.run.dir="$RUN_LOG_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=False \
  > "$TERMINAL_LOG_FILE" 2>&1