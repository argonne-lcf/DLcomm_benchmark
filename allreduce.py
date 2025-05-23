 
import os
import socket
import datetime
from time import perf_counter
from mpi4py import MPI
import torch
import torch.distributed as dist



def allreduce_ccl_ring(cfg):
    
    t1 = perf_counter()
    import oneccl_bindings_for_pytorch  
    import intel_extension_for_pytorch
    import torch.nn.parallel
    t2 = perf_counter()
    import_timer = t2 - t1
 
    MPI.COMM_WORLD.Barrier()
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    mpi_world_size = MPI.COMM_WORLD.Get_size()
    mpi_my_rank = MPI.COMM_WORLD.Get_rank()

    if mpi_my_rank == 0:
        master_addr = socket.gethostname()
        master_port = 2345
    else:
        master_addr = None
        master_port = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    master_port = MPI.COMM_WORLD.bcast(master_port, root=0)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

     
    MPI.COMM_WORLD.Barrier()
    t3 = perf_counter()
    dist.init_process_group(
        backend="ccl",
        init_method="env://",
        world_size=mpi_world_size,
        rank=mpi_my_rank,
        timeout=datetime.timedelta(seconds=3600)
    )
    t4 = perf_counter()
    init_timer = t4 - t3
    MPI.COMM_WORLD.Barrier()

    dist_my_rank = dist.get_rank()
    dist_world_size = dist.get_world_size()

 
    def get_default_device():
        if torch.xpu.is_available():
            return torch.device(f"xpu:{dist_my_rank % 12}")
        else:
            return torch.device("cpu")

    device = get_default_device()

    dim_size = cfg.buffer_size // 4
    elapsed = []

    for _ in range(50):
        x = torch.ones([1, dim_size], dtype=torch.float32).to(device, non_blocking=True)
        t5 = perf_counter()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        MPI.COMM_WORLD.Barrier()
        t6 = perf_counter()
        elapsed.append(t6 - t5)

    if mpi_my_rank == 0:
        print(f"[Rank 0] Import time: {import_timer:.4f}s")
        print(f"[Rank 0] Init time: {init_timer:.4f}s")
        for i, e in enumerate(elapsed):
            print(f"Iteration {i+1}: {e:.6f} sec")





def allreduce_ring_torch(cfg):
    raise NotImplementedError("Not yet implemented.")

def allreduce_tree_ccl(cfg):
    raise NotImplementedError("Not yet implemented.")

def allreduce_ring_mpi(cfg):
    raise NotImplementedError("Not yet implemented.")
