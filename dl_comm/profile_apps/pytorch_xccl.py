

import sys
from time import perf_counter
import os
from mpi4py import MPI
import datetime
from dl_comm.collectives import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from dl_comm.timer import timer, print_all_times, print_all_bandwidths
from dl_comm.utils.utility import DLIOLogger, Profile

log = DLIOLogger.get_instance()
dlp = Profile("DL_COMM")     






with timer("import time"):
    import intel_extension_for_pytorch   
    import torch.nn.parallel
    import torch.distributed as dist
    import oneccl_bindings_for_pytorch
    






def main():
    #argv map
    # ---------------------------------------------------------------------- #
    framework  = sys.argv[1].lower()   
    coll_name  = sys.argv[2].lower()   
    op_name    = sys.argv[3].lower()    
    buf_bytes  = int(sys.argv[4])      
    iters      = int(sys.argv[5])       
    dtype_str  = sys.argv[6].lower()   
    tp_size    = int(sys.argv[7])      
    dp_size    = int(sys.argv[8])  
    flatview   = sys.argv[9].lower() == "true"

    # ----------------------------------------------------------------------- #
    if MPI.COMM_WORLD.Get_size() != tp_size * dp_size:
        raise RuntimeError("world_size mismatch")




    torch_dtype, elem_size = DTYPES[dtype_str]

    # look-ups  
    run_collective = COLLECTIVES[coll_name]
    op_obj = OP_MAP[op_name] if coll_name in OPS_NEED_REDUCE else None

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

    # Broadcast HOST info for torch.distributed
    if mpi_rank == 0:
        import socket
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2348
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



    # ----------------------------------------------------------------------
    #  building TP,DP groups
    # ----------------------------------------------------------------------
    tp_rank = mpi_rank % tp_size
    dp_rank = mpi_rank // tp_size

    # tensor-parallel 
    tp_ranks =  [n * tp_size + tp_rank for n in range(dp_size)]

    # data-parallel 
    dp_ranks =  list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))

    # forming groups 
    tp_group    = dist.new_group(ranks=tp_ranks)
    dp_group    = dist.new_group(ranks=dp_ranks)
     

    # flatview
    world_group = dist.group.WORLD



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
        log.info("")   
        log.info("[MPI][SETUP] ------------------------------------------------------")
        log.info(f"[MPI][SETUP] Framework      : {framework}")
        log.info(f"[MPI][SETUP] Collective     : {coll_name}")
        log.info(f"[MPI][SETUP] Operation      : {op_name if op_obj else 'N/A'}")
        log.info(f"[MPI][SETUP] DType          : {dtype_str}")
        log.info(f"[MPI][SETUP] Buffer Size    : {buf_bytes}")
        log.info(f"[MPI][SETUP] Iterations     : {iters}")
        log.info(f"[MPI][SETUP] World Size     : {mpi_size}")
        log.info("[MPI][SETUP] ------------------------------------------------------")
        log.info("")  

    for _ in range(iters):  
        x = torch.ones(num_elems , dtype=torch_dtype).to(device, non_blocking=True)

        if not flatview:

            with timer("Latencies (s) (TP)"):
                run_collective(x, op_obj, group=tp_group)
                MPI.COMM_WORLD.Barrier()

            with timer("Latencies (s) (DP)"):
                run_collective(x, op_obj, group=dp_group)
                MPI.COMM_WORLD.Barrier()
    
        else:

            with timer("Latencies (s) (Flatview)"):
                run_collective(x, op_obj, group=world_group)
                MPI.COMM_WORLD.Barrier()
        
    if mpi_rank == 0:
        print_all_times()
        print_all_bandwidths(buf_bytes, coll_name)
       


if __name__ == "__main__":
    main()


