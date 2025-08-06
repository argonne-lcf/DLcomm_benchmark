 

import os
import socket
import torch
import torch.distributed as dist
from mpi4py import MPI

def main():
    
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    
     
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "COLL,TUNING"
    os.environ["NCCL_ALGO"] = "NVLS"  # Override  
    
 
    if mpi_rank == 0:
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 12357
    else:
        MASTER_ADDR = None
        MASTER_PORT = None
    
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    MASTER_PORT = MPI.COMM_WORLD.bcast(MASTER_PORT, root=0)
    
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    if mpi_rank == 0:
        print(f"[TEST] NCCL_ALGO set to: {os.environ['NCCL_ALGO']}")
        print(f"[TEST] Running with {mpi_size} ranks")
        print(f"[TEST] MASTER_ADDR: {MASTER_ADDR}")
    
    
    dist.init_process_group(backend="nccl", rank=mpi_rank, world_size=mpi_size)
    
  
    device_id = mpi_rank % torch.cuda.device_count()
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    
    if mpi_rank == 0:
        print(f"[TEST] Using device: {device}")
    
     
    tensor = torch.ones( 1073741824 , dtype=torch.float32).to(device)
    
    if mpi_rank == 0:
        print("[TEST] Running allreduce...")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
   
    expected = float(mpi_size)
    actual = tensor[0].item()
    
    if mpi_rank == 0:
        if abs(actual - expected) < 1e-6:
            print(f"[TEST] SUCCESS: allreduce result {actual} == {expected}")
        else:
            print(f"[TEST] FAILED: allreduce result {actual} != {expected}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()