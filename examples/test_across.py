#!/usr/bin/env python3

import torch
import torch.distributed as dist
from mpi4py import MPI
import socket
import datetime

def test_across_node():
    # MPI auto-initializes, just get rank/size
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    
 
    
    # Test config matching DLcomm grad-sync
    gpu_ids_per_node = [0, 2, 3]
    num_compute_nodes = 2
    
    # MPI rank coordination (same as dlcomm_main)
    if mpi_rank == 0:
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2256
    else:
        MASTER_ADDR = None
        MASTER_PORT = None
    
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    MASTER_PORT = MPI.COMM_WORLD.bcast(MASTER_PORT, root=0)
    
     
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        world_size=mpi_size,
        rank=mpi_rank,
        timeout=datetime.timedelta(seconds=3600)
    )
    


    # Device assignment logic from comm_setup (across_node mode)
    node_id = mpi_rank // len(gpu_ids_per_node)
    gpu_idx_in_node = mpi_rank % len(gpu_ids_per_node)
    assigned_gpu_id = gpu_ids_per_node[gpu_idx_in_node]
    
    hostname = socket.gethostname()
 
    # Initialize PyTorch distributed (same as dlcomm_main)

     
    
    # Create across-node groups (same logic as comm_setup)
    for gpu_idx, required_gpu_id in enumerate(gpu_ids_per_node):
        group_ranks = []
        for node in range(num_compute_nodes):
            rank = node * len(gpu_ids_per_node) + gpu_idx
            group_ranks.append(rank)
        
         
        
        if group_ranks:
            group = dist.new_group(ranks=group_ranks, use_local_synchronization=True)
                        # Set device
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{assigned_gpu_id}")
                torch.cuda.set_device(assigned_gpu_id)
                print(f"Rank {mpi_rank} on {hostname} assigned to {device}")
                backend = 'nccl'
            else:
                device = torch.device('cpu')
                print(f"Rank {mpi_rank} using CPU")
                backend = 'gloo'
                
            if mpi_rank in group_ranks:
                print(f"ACROSS GROUP {gpu_idx}: Rank {mpi_rank} on {hostname} using {device}")
    
    print(f"Rank {mpi_rank}: Test completed")
    dist.destroy_process_group()

if __name__ == "__main__":
    test_across_node()