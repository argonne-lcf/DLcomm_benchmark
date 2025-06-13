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

combined_config = {
    'within_node': {
        'num_gpus': 4,
        'gpu_ids_per_node': [5, 6, 8, 10]
    },
    'across_node': {
        'num_nodes': 2,
        'gpu_ids_per_node': [5, 10]
    }
}

num_gpus = combined_config['within_node']['num_gpus']
gpu_ids_per_node = combined_config['within_node']['gpu_ids_per_node']
num_nodes = combined_config['across_node']['num_nodes']
across_gpu_ids = combined_config['across_node']['gpu_ids_per_node']

MPI.COMM_WORLD.Barrier()

mpi_world_size = MPI.COMM_WORLD.Get_size()
mpi_my_rank = MPI.COMM_WORLD.Get_rank()
os.environ['RANK'] = str(mpi_my_rank)
os.environ['WORLD_SIZE'] = str(mpi_world_size)

if mpi_my_rank == 0:
   master_addr = socket.gethostname()
   sock = socket.socket()
   sock.bind(('',0))
   master_port = 2345
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

node_id = mpi_my_rank // num_gpus
gpu_index = mpi_my_rank % num_gpus

within_groups = []

for node in range(num_nodes):
    group_ranks = []
    for gpu in range(num_gpus):
        rank = node * num_gpus + gpu
        group_ranks.append(rank)
    within_groups.append(dist.new_group(ranks=group_ranks))
  

my_within_group = within_groups[node_id] 

across_groups = []

for gpu_id in across_gpu_ids:
    gpu_idx = gpu_ids_per_node.index(gpu_id)
    group_ranks = []
    for node in range(num_nodes):
        rank = node * num_gpus + gpu_idx
        group_ranks.append(rank)
    across_groups.append(dist.new_group(ranks=group_ranks))
    

current_gpu_id = gpu_ids_per_node[gpu_index]
my_across_group = None
if current_gpu_id in across_gpu_ids:
    across_idx = across_gpu_ids.index(current_gpu_id)
    my_across_group = across_groups[across_idx]


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

dim_size = int(int(sys.argv[1])/4)
MPI.COMM_WORLD.Barrier()

elapsed_within = []
elapsed_across = []
elapsed_total = []

for i in range(50):
    x = torch.ones([1, dim_size], dtype=torch.float32).to(device, non_blocking=True)
    
    if i == 0 and mpi_my_rank == 0:
        print(f"Rank {mpi_my_rank}: Before any allreduce - tensor sum: {x.sum()}")
    
    t_start = perf_counter_ns()
    
    t5 = perf_counter_ns()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_within_group)
    t6 = perf_counter_ns()
    elapsed_within.append(t6 - t5)
    
    if i == 0 and mpi_my_rank == 0:
        print(f"Rank {mpi_my_rank}: After within-node allreduce - tensor sum: {x.sum()}")
    
    MPI.COMM_WORLD.Barrier()
    
    t7 = perf_counter_ns()
    if my_across_group is not None:
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_across_group)
    t8 = perf_counter_ns()
    elapsed_across.append(t8 - t7)
    
    if i == 0 and mpi_my_rank == 0:
        print(f"Rank {mpi_my_rank}: After across-node allreduce - tensor sum: {x.sum()}")
    
    MPI.COMM_WORLD.Barrier()
    t_end = perf_counter_ns()
    elapsed_total.append(t_end - t_start)

    