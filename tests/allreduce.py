import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI
import torch
import torch.distributed as dist
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch



mpi_world_size = MPI.COMM_WORLD.Get_size()
mpi_my_rank = MPI.COMM_WORLD.Get_rank()
os.environ['RANK'] = str(mpi_my_rank)
os.environ['WORLD_SIZE'] = str(mpi_world_size)

if mpi_my_rank == 0:
   master_addr = socket.gethostname()
   sock = socket.socket()
   sock.bind(('',0))
   master_port = 2350
else:
   master_addr = None
   master_port = None

master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = str(master_port)

MPI.COMM_WORLD.Barrier()

dist.init_process_group(backend="ccl", init_method='env://', world_size=mpi_world_size, rank=mpi_my_rank, timeout=datetime.timedelta(seconds=3600))

dist_my_rank = dist.get_rank()
dist_world_size = dist.get_world_size()

if dist_my_rank == 0:
    print("=== SIMPLE ALLREDUCE TEST ===")
    print(f"World size: {dist_world_size}")

MPI.COMM_WORLD.Barrier()

if torch.xpu.is_available():
    device = torch.device(f"xpu:{dist_my_rank % torch.xpu.device_count()}")
    torch.xpu.set_device(device)
else:
    device = torch.device('cpu')

# Create subgroups - split world into groups of 2
group_size = 2
my_group = None
group_id = dist_my_rank // group_size

for gid in range((dist_world_size + group_size - 1) // group_size):
    start_rank = gid * group_size
    end_rank = min(start_rank + group_size, dist_world_size)
    group_ranks = list(range(start_rank, end_rank))
    
    group = dist.new_group(ranks=group_ranks, use_local_synchronization=True)
    
    if dist_my_rank in group_ranks:
        my_group = group
        break

if dist_my_rank == 0:
    print(f"Created subgroups of size {group_size}")

MPI.COMM_WORLD.Barrier()

x = torch.full((2,), float(dist_my_rank + 1), dtype=torch.bfloat16).to(device)
buffer_size = x.numel() * x.element_size()

print(f"Before AllReduce - Rank {dist_my_rank}: {x} (buffer size: {buffer_size} bytes)")

MPI.COMM_WORLD.Barrier()

# AllReduce within subgroup
if my_group is not None:
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_group)

print(f"After AllReduce - Rank {dist_my_rank}: {x} (buffer size: {buffer_size} bytes)")

MPI.COMM_WORLD.Barrier()

def check_correctness(tensor_after, my_group, group_id, op):
    if my_group is None:
        return
        
    group_ranks = dist.get_process_group_ranks(my_group)
    input_values = [rank + 1 for rank in group_ranks]
    
    if op == dist.ReduceOp.SUM:
        expected_value = sum(input_values)
        op_name = "SUM"
    elif op == dist.ReduceOp.MAX:
        expected_value = max(input_values)
        op_name = "MAX"
    elif op == dist.ReduceOp.MIN:
        expected_value = min(input_values)
        op_name = "MIN"
    elif op == dist.ReduceOp.PRODUCT:
        expected_value = 1
        for val in input_values:
            expected_value *= val
        op_name = "PRODUCT"
    
    expected_tensor = torch.full_like(tensor_after, expected_value)
    is_correct = torch.allclose(tensor_after, expected_tensor, rtol=1e-6)
    
    correct_tensor = torch.tensor([1 if is_correct else 0], dtype=torch.int32).to(device)
    dst_rank = min(group_ranks)
    
    if dist_my_rank == dst_rank:
        group_world_size = dist.get_world_size(my_group)
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(group_world_size)]
        dist.gather(correct_tensor, gathered_results, dst=dst_rank, group=my_group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == group_world_size:
            print(f"[CORRECTNESS][Group-{group_id}] AllReduce {op_name} [PASSED] - Ranks {group_ranks} received correct value: {expected_value}")
        else:
            failed_ranks = [group_ranks[i] for i, result in enumerate(gathered_results) if result.item() == 0]
            print(f"[CORRECTNESS][Group-{group_id}] AllReduce {op_name} [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=dst_rank, group=my_group)

# Correctness check for subgroup
check_correctness(x, my_group, group_id, dist.ReduceOp.SUM)

MPI.COMM_WORLD.Barrier()

dist.destroy_process_group()








