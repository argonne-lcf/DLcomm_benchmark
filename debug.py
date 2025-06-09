import os
import torch
import torch.distributed as dist
from mpi4py import MPI

import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

rank = MPI.COMM_WORLD.Get_rank()
world_size = MPI.COMM_WORLD.Get_size()

os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)

print(f"[Rank {rank}] About to init_process_group", flush=True)
dist.init_process_group(backend="ccl", rank=rank, world_size=world_size)
print(f"[Rank {rank}] Finished init_process_group", flush=True)

x = torch.tensor([rank + 1.0])
print(f"[Rank {rank}] Before all_reduce", flush=True)
dist.all_reduce(x)
print(f"[Rank {rank}] All-reduce result: {x.item()}", flush=True)

dist.destroy_process_group()
