import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI

# ---------------- Import and Timer ----------------
t1 = perf_counter_ns()
import intel_extension_for_pytorch
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch
t2 = perf_counter_ns()
import_timer = t2 - t1

# ---------------- Config ----------------
combined_config = {
    'within_node': {
        'num_gpus': 4,
        'gpu_ids_per_node': [0, 1, 2, 3]
    },
    'across_node': {
        'num_nodes': 2,
        'gpu_ids_per_node': [0, 1, 2, 3]
    }
    ,

    'dim_size_param': 1024 
}
dim_size=combined_config['dim_size_param']
num_gpus = combined_config['within_node']['num_gpus']
gpu_ids_per_node = combined_config['within_node']['gpu_ids_per_node']
num_nodes = combined_config['across_node']['num_nodes']
across_gpu_ids = combined_config['across_node']['gpu_ids_per_node']

# ---------------- MPI Setup ----------------
MPI.COMM_WORLD.Barrier()
mpi_world_size = MPI.COMM_WORLD.Get_size()
mpi_my_rank = MPI.COMM_WORLD.Get_rank()
os.environ['RANK'] = str(mpi_my_rank)
os.environ['WORLD_SIZE'] = str(mpi_world_size)

# MASTER_ADDR and MASTER_PORT setup
if mpi_my_rank == 0:
    master_addr = socket.gethostname()
    sock = socket.socket()
    sock.bind(('', 0))
    master_port = sock.getsockname()[1]
    sock.close()
else:
    master_addr = None
    master_port = None

master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = str(master_port)

# ---------------- PyTorch Init ----------------
MPI.COMM_WORLD.Barrier()
t3 = perf_counter_ns()
dist.init_process_group(backend="ccl", init_method='env://', world_size=mpi_world_size, rank=mpi_my_rank, timeout=datetime.timedelta(seconds=3600))
t4 = perf_counter_ns()
init_timer = t4 - t3

# ---------------- Group Info ----------------
node_id = mpi_my_rank // num_gpus
gpu_index = mpi_my_rank % num_gpus

# Build within-node groups
within_groups = []
for node in range(num_nodes):
    group_ranks = [node * num_gpus + g for g in range(num_gpus)]
    within_groups.append(dist.new_group(ranks=group_ranks))

my_within_group = within_groups[node_id]

# Build across-node groups
across_groups = []
for gpu_id in across_gpu_ids:
    gpu_idx = gpu_ids_per_node.index(gpu_id)
    group_ranks = [node * num_gpus + gpu_idx for node in range(num_nodes)]
    across_groups.append(dist.new_group(ranks=group_ranks))

current_gpu_id = gpu_ids_per_node[gpu_index]
my_across_group = across_groups[across_gpu_ids.index(current_gpu_id)] if current_gpu_id in across_gpu_ids else None

# ---------------- Device Assignment ----------------
def get_device_for_rank(rank):
    gpu_idx = rank % num_gpus
    if torch.xpu.is_available():
        device_id = gpu_ids_per_node[gpu_idx]
        return torch.device(f"xpu:{device_id}")
    else:
        return torch.device('cpu')

device = get_device_for_rank(mpi_my_rank)

# ---------------- Benchmark ----------------
 
MPI.COMM_WORLD.Barrier()

elapsed_within, elapsed_across, elapsed_total = [], [], []

for i in range(5):
    x = torch.ones([1, dim_size], dtype=torch.float32).to(device, non_blocking=True)
    
    t_start = perf_counter_ns()

    t5 = perf_counter_ns()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_within_group)
    t6 = perf_counter_ns()
    elapsed_within.append(t6 - t5)

    MPI.COMM_WORLD.Barrier()

    t7 = perf_counter_ns()
    if my_across_group is not None:
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_across_group)
    t8 = perf_counter_ns()
    elapsed_across.append(t8 - t7)

    MPI.COMM_WORLD.Barrier()
    t_end = perf_counter_ns()
    elapsed_total.append(t_end - t_start)

# ---------------- Reporting ----------------
group_min_rank_within = node_id * num_gpus
group_min_rank_across = gpu_index

if mpi_my_rank == group_min_rank_within:
    print(f"\n=== WITHIN-NODE RESULTS (Node {node_id}) ===")
    print("Iteration | Time (ms)")
    print("----------|----------")
    for i, t in enumerate(elapsed_within):
        print(f"   {i+1:2d}     | {t / 1e6:8.3f}")
    print(f"Import time: {import_timer / 1e6:.3f} ms")
    print(f"Init time  : {init_timer / 1e6:.3f} ms")

if my_across_group is not None and mpi_my_rank == group_min_rank_across:
    print(f"\n=== ACROSS-NODE RESULTS (GPU {gpu_index}) ===")
    print("Iteration | Time (ms)")
    print("----------|----------")
    for i, t in enumerate(elapsed_across):
        print(f"   {i+1:2d}     | {t / 1e6:8.3f}")
    print("")

