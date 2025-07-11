import os
from mpi4py import MPI
import torch
import torch.distributed as dist
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

os.environ['RANK'] = str(MPI.COMM_WORLD.Get_rank())
os.environ['WORLD_SIZE'] = str(MPI.COMM_WORLD.Get_size())


# 1. Initialize distributed environment
# dist.init_process_group(backend='gloo')
dist.init_process_group(backend='ccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
loc_size = 12
loc_rank = rank % loc_size
parallel_group = dist.new_group([loc_rank, loc_rank+loc_size],use_local_synchronization=True)
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