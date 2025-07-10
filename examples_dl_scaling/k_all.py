import os
import socket
from mpi4py import MPI
import torch
import torch.distributed as dist
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

os.environ['RANK'] = str(mpi_rank)
os.environ['WORLD_SIZE'] = str(mpi_size)

# Setup MASTER_ADDR and MASTER_PORT
if mpi_rank == 0:
    master_addr = socket.gethostname()
    master_port = 2342
else:
    master_addr = None
    master_port = None

master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = str(master_port)

MPI.COMM_WORLD.Barrier()

# 1. Initialize distributed environment
dist.init_process_group(backend='ccl', init_method='env://', world_size=mpi_size, rank=mpi_rank)
rank = dist.get_rank()
world_size = dist.get_world_size()
loc_size = 12
loc_rank = rank % loc_size
parallel_group = dist.new_group(
    [loc_rank, loc_rank+loc_size], use_local_synchronization=True
)

print(f"parallel_group ranks: {dist.get_process_group_ranks(parallel_group)}", flush=True)
world_group = dist.group.WORLD
parallel_world_size = dist.get_world_size(parallel_group)

# 2. Create input and output buff
tensor = torch.rand(1024**2, device=loc_rank)
out_world = [torch.rand(1024**2, device=loc_rank) for _ in range(world_size)]
out_parallel = [torch.rand(1024**2, device=loc_rank) for _ in range(parallel_world_size)]

# 3. Regular all_gather (works)
dist.all_gather(out_world, tensor, group=world_group)
dist.barrier()
if rank == 0:
    print("Regular all-gather succeded", flush=True)

# 4. Parallel all_gather (hangs)
dist.all_gather(out_parallel, tensor, group=parallel_group)
dist.barrier()
if rank == 0:
    print("Parallel all-gather succeded", flush=True)