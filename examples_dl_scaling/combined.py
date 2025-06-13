import datetime
from time import perf_counter
import sys
import os
import socket
from mpi4py import MPI
import torch
 
t1 = perf_counter() 
import intel_extension_for_pytorch  # Added Extra
import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch
t2 = perf_counter() 
import_timer = t2 - t1
 
MPI.COMM_WORLD.Barrier()
#Here we are setting env for each process independently
os.environ['RANK']          = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE']    = str(os.environ.get('PMI_SIZE', 1))
mpi_world_size              = MPI.COMM_WORLD.Get_size()
mpi_my_rank                 = MPI.COMM_WORLD.Get_rank()



##
#We want all processes (ranks) to agree on who the "master" is and how to contact it.
##
if mpi_my_rank == 0:
   master_addr              = socket.gethostname()
   sock                     = socket.socket()
   sock.bind(('',0))
   # master_port  = sock.getsockname()[1] 
   master_port              = 2345
else:
   master_addr              = None
   master_port              = None


##
# All ranks learn the info here, root 0 means ranks 0 is broadcasting
##
master_addr                 = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port                 = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"]   = master_addr
os.environ["MASTER_PORT"]   = str(master_port)

MPI.COMM_WORLD.Barrier()
t3 = perf_counter() 
dist.init_process_group(backend = "ccl", init_method = 'env://', world_size = mpi_world_size, rank = mpi_my_rank, timeout = datetime.timedelta(seconds=3600))
t4 = perf_counter() 
init_timer = t4 - t3
MPI.COMM_WORLD.Barrier()

# so we receive dist ranks here
dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()

#each rank has their device
def get_default_device():
    if torch.xpu.is_available():
        return torch.device(f"xpu:{dist_my_rank%12}")
    else:
        return torch.device('cpu')

device  = get_default_device()

dim_size=int(int(sys.argv[1])/4)
tp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 12
dp_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2

if tp_size * dp_size != dist_world_size:
    if dist_my_rank == 0:
        print(f"Error: TP_SIZE ({tp_size}) * DP_SIZE ({dp_size}) = {tp_size * dp_size} != WORLD_SIZE ({dist_world_size})")
    sys.exit(1)

tp_rank = dist_my_rank % tp_size
dp_rank = dist_my_rank // tp_size

tp_groups = []
for i in range(dp_size):
    tp_group_ranks = list(range(i * tp_size, (i + 1) * tp_size))
    tp_group = dist.new_group(tp_group_ranks)
    tp_groups.append(tp_group)

my_tp_group = tp_groups[dp_rank]

dp_groups = []
for i in range(tp_size):
    dp_group_ranks = list(range(i, dist_world_size, tp_size))
    dp_group = dist.new_group(dp_group_ranks)
    dp_groups.append(dp_group)

my_dp_group = dp_groups[tp_rank]

if dist_my_rank == 0:
    print(f"Configuration: TP_SIZE={tp_size}, DP_SIZE={dp_size}, WORLD_SIZE={dist_world_size}")
    print(f"TP Groups: {[list(range(i * tp_size, (i + 1) * tp_size)) for i in range(dp_size)]}")
    print(f"DP Groups: {[list(range(i, dist_world_size, tp_size)) for i in range(tp_size)]}")

MPI.COMM_WORLD.Barrier()

elapsed_tp = []
elapsed_dp = []
elapsed_total = []

for _ in range(50):
    x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
    
    t_start = perf_counter()
    
    t5 = perf_counter() 
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_tp_group)
    MPI.COMM_WORLD.Barrier()
    t6 = perf_counter()
    elapsed_tp.append(t6 - t5)
    
    t7 = perf_counter() 
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=my_dp_group)
    MPI.COMM_WORLD.Barrier()
    t8 = perf_counter()
    elapsed_dp.append(t8 - t7)
    
    t_end = perf_counter()
    elapsed_total.append(t_end - t_start)

if mpi_my_rank == 0:
    print(f"Rank {mpi_my_rank}: TP_RANK={tp_rank}, DP_RANK={dp_rank}")
    print(f"Import timer: {import_timer}")
    print(f"Init timer: {init_timer}")
    print("TP_TIMINGS:")
    for e in elapsed_tp:
        print(e)
    print("DP_TIMINGS:")
    for e in elapsed_dp:
        print(e)
    print("TOTAL_TIMINGS:")
    for e in elapsed_total:
        print(e) 