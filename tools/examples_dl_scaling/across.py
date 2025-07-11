 
import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI

print(f"[RANK {MPI.COMM_WORLD.Get_rank()}] Starting imports...", flush=True)

# Fix CCL_WORKER_AFFINITY issue
if 'CCL_WORKER_AFFINITY' in os.environ:
    print(f"[RANK {MPI.COMM_WORLD.Get_rank()}] Removing CCL_WORKER_AFFINITY: {os.environ['CCL_WORKER_AFFINITY']}", flush=True)
    del os.environ['CCL_WORKER_AFFINITY']

t1 = perf_counter_ns() 
import intel_extension_for_pytorch
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch
t2 = perf_counter_ns() 
import_timer = t2 - t1
print(f"[RANK {MPI.COMM_WORLD.Get_rank()}] Imports complete", flush=True)

across_config = {
    'num_nodes': 2,
    'num_gpus': 2,
    'gpu_ids_per_node': [3, 4]
}

num_nodes = across_config['num_nodes']
num_gpus = across_config['num_gpus']
gpu_ids_per_node = across_config['gpu_ids_per_node']
 
print(f"[RANK {MPI.COMM_WORLD.Get_rank()}] Starting MPI setup...", flush=True)
MPI.COMM_WORLD.Barrier()

mpi_world_size = MPI.COMM_WORLD.Get_size()
mpi_my_rank = MPI.COMM_WORLD.Get_rank()
os.environ['RANK'] = str(mpi_my_rank)
os.environ['WORLD_SIZE'] = str(mpi_world_size)

print(f"[RANK {mpi_my_rank}] Setting up master address...", flush=True)
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

print(f"[RANK {mpi_my_rank}] About to init process group...", flush=True)
MPI.COMM_WORLD.Barrier()
t3 = perf_counter_ns()

dist.init_process_group(backend="ccl", init_method='env://', world_size=mpi_world_size, rank=mpi_my_rank, timeout=datetime.timedelta(seconds=3600))
print(f"[RANK {mpi_my_rank}] Process group initialized successfully", flush=True)

gpu_index = mpi_my_rank % num_gpus
node_id = mpi_my_rank // num_gpus

print(f"[RANK {mpi_my_rank}] Creating across-node groups...", flush=True)
across_groups = []
for i in range(num_gpus):
    group_ranks = []
    for node in range(num_nodes):
        rank = node * num_gpus + i
        group_ranks.append(rank)
    
    print(f"[RANK {mpi_my_rank}] Creating group {i} with ranks: {group_ranks}", flush=True)
    across_groups.append(dist.new_group(ranks=group_ranks, use_local_synchronization=True))

my_across_group = across_groups[gpu_index]
print(f"[RANK {mpi_my_rank}] My across group: {gpu_index}, ranks: {dist.get_process_group_ranks(my_across_group)}", flush=True)

t4 = perf_counter_ns() 
init_timer = t4 - t3
print(f"[RANK {mpi_my_rank}] Group creation complete", flush=True)
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
print(f"[RANK {mpi_my_rank}] Using device: {device}", flush=True)

dim_size = int(int(sys.argv[1])/4)
print(f"[RANK {mpi_my_rank}] Tensor dimension size: {dim_size}", flush=True)
MPI.COMM_WORLD.Barrier()

elapsed1 = []

print(f"[RANK {mpi_my_rank}] Creating tensor...", flush=True)
x = torch.ones([1, dim_size], dtype=torch.float32).to(device, non_blocking=True)

print(f"[RANK {mpi_my_rank}] Preparing tensor list for allgather...", flush=True)
group_size = dist.get_world_size(my_across_group)
tensor_list = [torch.empty_like(x) for _ in range(group_size)]

print(f"[RANK {mpi_my_rank}] About to call allgather with group size: {group_size}", flush=True)
t5 = perf_counter_ns()
dist.all_gather(tensor_list, x, group=my_across_group)
t6 = perf_counter_ns()
print(f"[RANK {mpi_my_rank}] Allgather completed successfully!", flush=True)

elapsed1.append(t6 - t5)
print(f"[RANK {mpi_my_rank}] Script finished. Time: {elapsed1[0]} ns", flush=True)


