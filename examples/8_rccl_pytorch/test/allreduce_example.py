import os
import torch
import torch.distributed as dist
from mpi4py import MPI

def main():
    # Step 1: MPI-based process info
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    world_size = comm.Get_size()
    num_gpus = torch.cuda.device_count()
    local_rank = mpi_rank % num_gpus

    # Step 2: Set environment variables (Slurm batch script sets MASTER_ADDR and MASTER_PORT)
    os.environ["RANK"] = str(mpi_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"

    # Step 3: Initialize DDP
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # Step 4: AllReduce
    tensor = torch.ones(1, device="cuda")
    dist.all_reduce(tensor)
    print(f"Rank {rank}: Result = {tensor.item()}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
