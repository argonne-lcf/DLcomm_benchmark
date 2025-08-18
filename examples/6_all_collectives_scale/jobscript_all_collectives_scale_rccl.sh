#!/bin/bash
#SBATCH -A <projid>          # OLCF project ID goes here
#SBATCH -J dl_comm_2n4r
#SBATCH -N 2
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --threads-per-core=1
#SBATCH -o %x-%j.out

module load PrgEnv-cray craype-accel-amd-gfx90a rocm

if [ -n "$SLURM_SUBMIT_DIR" ]; then
  cd "$SLURM_SUBMIT_DIR"
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  cd "$SCRIPT_DIR"
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"
TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"

CONFIG_FILE="$(find "$SCRIPT_DIR" -maxdepth 1 -name '*.yaml' -type f | head -1)"
CONFIG_NAME="$(basename "$CONFIG_FILE" .yaml)"

srun --gpus-per-task=1 --gpu-bind=closest --cpu-bind=threads \
  python3 -m dl_comm.dl_comm_main --config-path="$SCRIPT_DIR" --config-name="$CONFIG_NAME" \
  2>&1 | tee "$TERMINAL_LOG_FILE"
