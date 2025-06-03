

import sys
from time import perf_counter
import os
from mpi4py import MPI
import datetime

t0 = perf_counter() 
import intel_extension_for_pytorch   
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch
t1 = perf_counter() 
import_timer = t1 - t0





DTYPES = {
    "float16": (torch.float16, 2),
    "float32": (torch.float32, 4),
    "float64": (torch.float64, 8),
    "int32":   (torch.int32,   4),
    "int64":   (torch.int64,   8),
}

def main():

    buf_bytes   = int(sys.argv[1])        
    iters       = int(sys.argv[2])        
    dtype_str   = sys.argv[3] 
    torch_dtype, elem_size = DTYPES[dtype_str]

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

    MPI.COMM_WORLD.Barrier()
    t2 = perf_counter() 
    dist.init_process_group(backend = "ccl", init_method = 'env://', world_size = mpi_size, rank = mpi_rank, timeout = datetime.timedelta(seconds=3600))
    t3 = perf_counter() 
    init_timer = t3 - t2
    MPI.COMM_WORLD.Barrier()

    if mpi_rank == 0:
        print(f"[TIMERS] import time  = {import_timer:.6f} s")
        print(f"[TIMERS] init time    = {init_timer:.6f} s")



    # so we receive dist ranks here
    dist_my_rank        = dist.get_rank()
    dist_world_size     = dist.get_world_size()

    def get_default_device(rank: int):
        if torch.xpu.is_available():
             
            return torch.device(f"xpu:{rank % torch.xpu.device_count()}")
        else:
            return torch.device("cpu")

    device  = get_default_device(dist_my_rank)
    num_elems = buf_bytes // elem_size
    # all‐reduce 
    latencies = []
    for _ in range(5):  
        x = torch.ones(num_elems , dtype=torch_dtype).to(device, non_blocking=True)
        t0 = perf_counter()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        MPI.COMM_WORLD.Barrier()
        t1 = perf_counter()
        latencies.append(t1 - t0)

    if mpi_rank == 0:
        print(f"[TIMERS] Latencies (s): {latencies}")
        print(f"[TIMERS] Bandwiths (bytes/sec): {buf_bytes/latencies[0]}")

if __name__ == "__main__":
    main()
