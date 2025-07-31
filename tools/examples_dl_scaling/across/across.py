 
import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI





t1 = perf_counter_ns() 
import intel_extension_for_pytorch
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch
t2 = perf_counter_ns() 
import_timer = t2 - t1

across_config = {
    'num_nodes': 2,
    'num_gpus': 4,  # Match RANKS_PER_NODE=4 in job script
    'gpu_ids_per_node': [0, 1, 2, 3],  # 4 GPUs per node
    'dim_size_param': 1024  # Default dimension size parameter
}

num_nodes = across_config['num_nodes']
num_gpus = across_config['num_gpus']
gpu_ids_per_node = across_config['gpu_ids_per_node']
 
MPI.COMM_WORLD.Barrier()

mpi_world_size = MPI.COMM_WORLD.Get_size()
mpi_my_rank = MPI.COMM_WORLD.Get_rank()
os.environ['RANK'] = str(mpi_my_rank)
os.environ['WORLD_SIZE'] = str(mpi_world_size)

if mpi_my_rank == 0:
   master_addr = socket.gethostname()
   sock = socket.socket()
   sock.bind(('',0))
   master_port = 2342
else:
   master_addr = None
   master_port = None

master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = str(master_port)

 
MPI.COMM_WORLD.Barrier()
t3 = perf_counter_ns()

dist.init_process_group(backend="ccl", init_method='env://', world_size=mpi_world_size, rank=mpi_my_rank, timeout=datetime.timedelta(seconds=3600))

gpu_index = mpi_my_rank % num_gpus
node_id = mpi_my_rank // num_gpus

 

across_groups = []
for i in range(num_gpus):
    group_ranks = []
    for node in range(num_nodes):
        rank = node * num_gpus + i
        group_ranks.append(rank)
 
    across_groups.append(dist.new_group(ranks=group_ranks, use_local_synchronization=True))

my_across_group = across_groups[gpu_index]

 
t4 = perf_counter_ns() 
init_timer = t4 - t3
MPI.COMM_WORLD.Barrier()

dist_my_rank = dist.get_rank()
dist_world_size = dist.get_world_size()

def get_device_for_rank(rank):
    gpu_idx = rank % num_gpus
    
    if torch.xpu.is_available():
        device_id = gpu_ids_per_node[gpu_idx]
        device = torch.device(f"xpu:{device_id}")
        return device
    else:
        return torch.device('cpu')

device = get_device_for_rank(dist_my_rank)

# Print configuration - only once from rank 0
if dist_my_rank == 0:
    print("=== ACROSS-NODE COMMUNICATION BENCHMARK ===")
    print(f"Configuration:")
    print(f"  Communication type: across_node")
    print(f"  Total nodes: {num_nodes}")
    print(f"  GPUs per node: {num_gpus}")
    print(f"  GPU IDs: {gpu_ids_per_node}")
    print(f"  Backend: ccl")
    print(f"  Collective: ALL_GATHER")

dim_size = int(int(across_config['dim_size_param'])/4)

if dist_my_rank == 0:
    print(f"  Tensor dimension: [1, {dim_size}]")
    print(f"  Data type: float32")
    print("Starting benchmark...")

# Calculate group info for results printing
group_min_rank = gpu_index  # minimum rank in this GPU group

MPI.COMM_WORLD.Barrier()

elapsed1 = []

for i in range(5):
    x = torch.ones([1, dim_size], dtype=torch.float32).to(device, non_blocking=True)
    
    group_size = dist.get_world_size(my_across_group)
    tensor_list = [torch.empty_like(x) for _ in range(group_size)]
    
    t5 = perf_counter_ns()
    dist.all_gather(tensor_list, x, group=my_across_group)
    t6 = perf_counter_ns()
    
    MPI.COMM_WORLD.Barrier()
    
    elapsed1.append(t6 - t5)

# Print timing results - each group's minimum rank prints its results
if dist_my_rank == group_min_rank:
    print(f"\n=== BENCHMARK RESULTS (Group {gpu_index}) ===")
    print(f"Iterations completed: {len(elapsed1)}")
    print("Communication times per iteration:")
    print("Iteration | Time (ms)")
    print("----------|----------")
    for i, time_ns in enumerate(elapsed1):
        print(f"    {i+1:2d}    | {time_ns / 1e6:8.3f}")
    print("----------|----------")
    print(f"Setup times:")
    print(f"  Import time: {import_timer / 1e6:.3f} ms")
    print(f"  Init time: {init_timer / 1e6:.3f} ms")
    print(f"=== BENCHMARK COMPLETE (Group {gpu_index}) ===")


