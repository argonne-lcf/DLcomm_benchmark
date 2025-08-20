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
export PYTHONPATH=/opt/cray/pe/python/3.11.7/:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/packs:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/dlcomm-pip:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/DLcomm_benchmark
 


#export MASTER_ADDR=$(hostname -i)
#export NCCL_SOCKET_IFNAME=hsn0
#export MASTER_PORT=3442
#socket problem
export MASTER_ADDR=$(ip -4 addr show dev hsn0 | awk '/inet/{print $2}' | cut -d/ -f1)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0

# (2) Optional: silence c10d warnings
export TORCH_CPP_LOG_LEVEL=ERROR   # suppresses socket.cpp warnings only; keeps errors.  # docs: torch env vars

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'



 
#srun --ntasks=16 --gpus-per-task=1 --cpu-bind=threads python3 -m dl_comm.dl_comm_main --config-path="$SCRIPT_DIR" --config-name=8_rccl_simple hydra.run.dir="$RUN_LOG_DIR" > "$TERMINAL_LOG_FILE" 2>&1

srun --ntasks=16 --export=ALL --cpu-bind=threads \
  python3 -m dl_comm.dl_comm_main \
  --config-path="$SCRIPT_DIR" \
  --config-name=8_rccl_pytorch \
  -- \
  hydra.run.dir="$RUN_LOG_DIR" \
  hydra.output_subdir=null \
  hydra.job.chdir=False \
  > "$TERMINAL_LOG_FILE" 2>&1



# { srun -N ${SLURM_JOB_NUM_NODES} --ntasks-per-node=${RANKS_PER_NODE} --export=ALL --cpu-bind=map_cpu:1,9,17,25,33,41,49,57  python3 -c "import torch; print(torch.__file__)" > output_${SLURM_JOB_NUM_NODES}_${SLURM_JOBID}_1.txt 2>&1; } 2> time_${SLURM_JOB_NUM_NODES}_${SLURM_JOBID}_1.txt


# For socket issues and proxy settings, refer to:
#https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#proxy-settings
#https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html#c10d-socket-warnings
#https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#open-issues-w-workaround 

