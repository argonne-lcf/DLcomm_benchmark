#!/bin/bash -x
#SBATCH -A GEN008
#SBATCH -J allreduce_test
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=8              # 8 GPUs per node
#SBATCH --gpus-per-node=8                # Matches tasks-per-node
#SBATCH --cpus-per-task=14
#SBATCH --threads-per-core=1
#SBATCH --export=NONE                    # Crucial for a clean env

# Load recommended modules
module load cray-python
export PYTHONPATH=/opt/cray/pe/python/3.11.7/:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/packs:/lustre/orion/gen008/proj-shared/kaushik/dlcomm-test/dlcomm-pip
 

# Set rendezvous variables
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0

# Optional NCCL tuning (can be adjusted later)
export FI_PROVIDER=efa,cxi,sockets

export NCCL_IB_DISABLE=1
 

# Clear proxies if necessary
unset http_proxy https_proxy ftp_proxy all_proxy

# Run your script with arguments (you must accept --master_addr and --master_port in your script)
srun -N 2 -n 16 --gpus-per-task=1 --gpu-bind=closest \
     python3 allreduce_example.py --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT
