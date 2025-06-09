# test_ccl_logging.py

from mpi4py import MPI
import os
import intel_extension_for_pytorch      
import oneccl_bindings_for_pytorch     
 
import torch
import torch.distributed as dist

def main():
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    # initialize the PyTorch CCL backend (via env://)
    dist.init_process_group(
        backend="ccl",
        init_method="env://",
        rank=rank,
        world_size=size,
    )

    # trivial all-reduce
    t = torch.tensor([rank], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"[Rank 0] sum of ranks 0..{size-1} = {t.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
