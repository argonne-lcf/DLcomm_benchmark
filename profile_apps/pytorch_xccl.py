

import sys
from time import perf_counter
import torch
import torch.distributed as dist
from mpi4py import MPI
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m profile_apps.profile_pytorch_xccl <buffer_bytes>")
        sys.exit(1)

    buf_bytes = int(sys.argv[1])
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

    # Broadcast HOST info for torch.distributed
    if mpi_rank == 0:
        import socket
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2345
    else:
        MASTER_ADDR = None
        MASTER_PORT = None

    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    MASTER_PORT = MPI.COMM_WORLD.bcast(MASTER_PORT, root=0)

    import os
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    # Initialize oneCCL (via torch.distributed)
    dist.init_process_group(backend="ccl", init_method="env://", world_size=mpi_size, rank=mpi_rank)
    

    # so we receive dist ranks here
    dist_my_rank        = dist.get_rank()
    dist_world_size     = dist.get_world_size()

    def get_default_device():
        if torch.xpu.is_available():
            return torch.device(f"xpu:{dist_my_rank%12}")
        else:
            return torch.device('cpu')

    device  = get_default_device()

    # all‐reduce 
    latencies = []
    for _ in range(5):  
        x = torch.ones(buf_bytes // 4, dtype=torch.float32).to(device, non_blocking=True)
        t0 = perf_counter()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        MPI.COMM_WORLD.Barrier()
        t1 = perf_counter()
        latencies.append(t1 - t0)

    if mpi_rank == 0:
        print("Latencies (s):", latencies)

if __name__ == "__main__":
    main()
