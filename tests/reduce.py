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

MPI.COMM_WORLD.Barrier()

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
    print("=== SIMPLE REDUCE TEST ===")
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

x = torch.full((2,), float(dist_my_rank + 1), dtype=torch.float32).to(device)
buffer_size = x.numel() * x.element_size()

print(f"Before Reduce - Rank {dist_my_rank}: {x} (buffer size: {buffer_size} bytes)")

MPI.COMM_WORLD.Barrier()

# Reduce within subgroup - result goes to root (min rank) of each group
if my_group is not None:
    group_ranks = dist.get_process_group_ranks(my_group)
    root_rank = min(group_ranks)
    dist.reduce(x, dst=root_rank, op=dist.ReduceOp.SUM, group=my_group)

print(f"After Reduce - Rank {dist_my_rank}: {x} (buffer size: {buffer_size} bytes)")

MPI.COMM_WORLD.Barrier()

def check_reduce_correctness(tensor_after, my_group, group_id, op, root_rank):
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
    
    # Only root rank should have the reduced value
    if dist_my_rank == root_rank:
        expected_tensor = torch.full_like(tensor_after, expected_value)
        is_correct = torch.allclose(tensor_after, expected_tensor, rtol=1e-6)
        
        if is_correct:
            print(f"[CORRECTNESS][Group-{group_id}] Reduce {op_name} [PASSED] - Root rank {root_rank} received correct value: {expected_value}")
        else:
            print(f"[CORRECTNESS][Group-{group_id}] Reduce {op_name} [FAILED] - Root rank {root_rank} received incorrect value")
    else:
        # Non-root ranks don't have defined values after reduce
        print(f"[INFO] Rank {dist_my_rank} (non-root) completed reduce operation")

# Correctness check for subgroup
if my_group is not None:
    group_ranks = dist.get_process_group_ranks(my_group)
    root_rank = min(group_ranks)
    check_reduce_correctness(x, my_group, group_id, dist.ReduceOp.SUM, root_rank)

MPI.COMM_WORLD.Barrier()

if dist_my_rank == 0:
    print("=== TEST COMPLETE ===")

dist.destroy_process_group()