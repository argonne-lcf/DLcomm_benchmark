

import sys
from time import perf_counter
import os
from mpi4py import MPI
import datetime
from collectives import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from timer import timer, print_all_times


with timer("import time"):
    import intel_extension_for_pytorch   
    import torch.nn.parallel
    import torch.distributed as dist
    import oneccl_bindings_for_pytorch
    






def main():
    #argv map
    # ---------------------------------------------------------------------- #
    framework   = sys.argv[1].lower()
    coll_name   = sys.argv[2].lower()
    op_name     = sys.argv[3].lower()
    buf_bytes   = int(sys.argv[4])
    iters       = int(sys.argv[5])
    dtype_str   = sys.argv[6].lower()
    # ----------------------------------------------------------------------- #
    torch_dtype, elem_size = DTYPES[dtype_str]

    # look-ups  
    collective_fn = COLLECTIVES[coll_name]
    op_obj = OP_MAP[op_name] if coll_name in OPS_NEED_REDUCE else None


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
    with timer("init time"):
        dist.init_process_group(backend = "ccl", init_method = 'env://', world_size = mpi_size, rank = mpi_rank, timeout = datetime.timedelta(seconds=3600))
    MPI.COMM_WORLD.Barrier()



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
   

    if mpi_rank == 0:
        print("\n[MPI][SETUP] ------------------------------------------------------")
        print(f"[MPI][SETUP] Framework      : {framework}")
        print(f"[MPI][SETUP] Collective     : {coll_name}")
        print(f"[MPI][SETUP] Operation      : {op_name if op_obj else 'N/A'}")
        print(f"[MPI][SETUP] DType          : {dtype_str}")
        print(f"[MPI][SETUP] Buffer Size    : {buf_bytes}")
        print(f"[MPI][SETUP] Iterations     : {iters}")
        print(f"[MPI][SETUP] World Size     : {mpi_size}")
        print("[MPI][SETUP] ------------------------------------------------------\n")

     
    for _ in range(5):  

        x = torch.ones(num_elems , dtype=torch_dtype).to(device, non_blocking=True)
       
        with timer("Latencies (s)"):
            collective_fn(x, op_obj)
            MPI.COMM_WORLD.Barrier()
        
     
    if mpi_rank == 0:
        print_all_times()

if __name__ == "__main__":
    main()



