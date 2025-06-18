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

within_config = {
    'num_nodes': 2,
    'num_gpus': 4,
    'gpu_ids_per_node': [ 8, 9, 10, 11]
}

num_nodes = within_config['num_nodes']
num_gpus = within_config['num_gpus']
gpu_ids_per_node = within_config['gpu_ids_per_node']

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

 
t4 = perf_counter_ns() 
init_timer = t4 - t3
MPI.COMM_WORLD.Barrier()

 

dist_my_rank = dist.get_rank()
dist_world_size = dist.get_world_size()

node_id=dist_my_rank // num_gpus

within_groups = []

for node in range(num_nodes):
    group_ranks = []
    for gpu in range(num_gpus):
        rank = node * num_gpus + gpu
        group_ranks.append(rank)
    within_groups.append(dist.new_group(ranks=group_ranks))
  

my_within_group = within_groups[node_id] 








def get_device_for_rank(rank):
    if torch.xpu.is_available():
        device_id = gpu_ids_per_node[rank % len(gpu_ids_per_node)]
        device = torch.device(f"xpu:{device_id}")
        return device
    else:
        return torch.device('cpu')

device = get_device_for_rank(dist_my_rank)
 
dim_size = int(int(sys.argv[1])/4)
MPI.COMM_WORLD.Barrier()

elapsed1 = []

for i in range(50):
    x = torch.ones([1, dim_size], dtype=torch.float32).to(device, non_blocking=True)
    if i == 0 and mpi_my_rank == 0:
        print(f"Rank {mpi_my_rank}: Before allreduce - tensor sum: {x.sum()}")
    t5 = perf_counter_ns()
    dist.all_reduce(x, op=dist.ReduceOp.SUM,group=my_within_group)
    if i == 0 and mpi_my_rank == 0:
        print(f"Rank {mpi_my_rank}: After allreduce - tensor sum: {x.sum()}")
    MPI.COMM_WORLD.Barrier()
    t6 = perf_counter_ns()
    elapsed1.append(t6 - t5)