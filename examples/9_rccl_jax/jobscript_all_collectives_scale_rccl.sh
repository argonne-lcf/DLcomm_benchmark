#!/bin/bash -x
#SBATCH -A GEN008
#SBATCH -J dlcomm
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --gpus-per-node=8

# salloc --network=single_node_vni,job_vni,def_tles=0   --exclusive   -A GEN008 -J copper_spindle -t 01:00:00  -N 2 # For interactive allocation
#or 
# salloc --network=single_node_vni,job_vni,def_tles=0   --exclusive   -A GEN008 -J copper_spindle -t 01:00:00  -N 2 # For interactive allocation
#or 
# salloc -A GEN008 -J copper_spindle -t 01:00:00  -N 2 # For interactive allocation
# squeue -l -u $USER
# scontrol show job 3676536
# scancel 3676536


 





#cd $SLURM_SUBMIT_DIR
echo Jobid: $SLURM_JOBID
echo $SLURM_NODELIST
RANKS_PER_NODE=8
 

SCRIPT_DIR="$(pwd)"
echo $SCRIPT_DIR

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S_%6N')"  # microsecond precision avoids duplicate runs
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
export RUN_LOG_DIR

TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
mkdir -p "$RUN_LOG_DIR"

CONFIG_FILE="$(find "$SCRIPT_DIR" -maxdepth 1 -name '*.yaml' -type f | head -1)"
CONFIG_NAME="$(basename "$CONFIG_FILE" .yaml)"

module load miniforge3/23.11.0-0
module load cray-python
module load rocm/6.2.4
module load cray-python
export PYTHONPATH=/opt/cray/pe/python/3.11.7/:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/packs:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/DLcomm_benchmark
 

export LD_LIBRARY_PATH=/opt/rocm-6.2.4/lib:/opt/rocm-6.2.4/lib64:$LD_LIBRARY_PATH


# Enable conda shell integration from the correct location
eval "$(/sw/frontier/miniforge3/23.11.0-0/bin/conda shell.bash hook)"

# Activate JAX environment
conda activate jax_env-frontier


export PATH=$CONDA_PREFIX/bin:$PATH
 

unset http_proxy
unset https_proxy
unset ftp_proxy
unset all_proxy
unset no_proxy

export MASTER_ADDR=$(ip -4 addr show dev hsn0 | awk '/inet/{print $2}' | cut -d/ -f1)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0

# (2) Optional: silence c10d warnings
export TORCH_CPP_LOG_LEVEL=ERROR   # suppresses socket.cpp warnings only; keeps errors.  # docs: torch env vars

 # Pin JAX to ROCm and skip TPU plugin entirely
export JAX_PLATFORMS=rocm




 # Coordinator host and port (simple & consistent with your working example)
export COORDINATOR_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export COORDINATOR_PORT=1234

 
srun --ntasks=16 --gpus-per-task=1 --cpus-per-task=1 --export=ALL   \
  python3 -m dl_comm.dl_comm_main \
  --config-path="$SCRIPT_DIR" \
  --config-name=9_rccl_jax \
  -- \
  hydra.run.dir="$RUN_LOG_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=False \
  > "$TERMINAL_LOG_FILE" 2>&1



 