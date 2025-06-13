import datetime
from time import perf_counter
import sys
import os
import socket
from mpi4py import MPI
import torch
import intel_extension_for_pytorch
import torch.distributed as dist
import oneccl_bindings_for_pytorch

MPI.COMM_WORLD.Barrier()

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
mpi_world_size = MPI.COMM_WORLD.Get_size()
mpi_my_rank = MPI.COMM_WORLD.Get_rank()

if mpi_my_rank == 0:
    master_addr = socket.gethostname()
    master_port = 2345
else:
    master_addr = None
    master_port = None

master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = str(master_port)

tp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 12
dp_size = int(sys.argv[3]) if len(sys.argv) > 3 else 2

tp_rank = mpi_my_rank % tp_size
dp_rank = mpi_my_rank // tp_size

MPI.COMM_WORLD.Barrier()

try:
    os.environ['RANK'] = str(tp_rank)
    os.environ['WORLD_SIZE'] = str(tp_size)
    os.environ['MASTER_PORT'] = str(2346)
    dist.init_process_group(backend="ccl", init_method='env://', world_size=tp_size, rank=tp_rank, timeout=datetime.timedelta(seconds=60))
    print(f"Rank {mpi_my_rank}: TP init SUCCESS")
except Exception as e:
    print(f"Rank {mpi_my_rank}: TP init ERROR: {e}")

MPI.COMM_WORLD.Barrier()

try:
    os.environ['RANK'] = str(dp_rank)
    os.environ['WORLD_SIZE'] = str(dp_size)
    os.environ['MASTER_PORT'] = str(2347)
    dist.init_process_group(backend="ccl", init_method='env://', world_size=dp_size, rank=dp_rank, timeout=datetime.timedelta(seconds=60))
    print(f"Rank {mpi_my_rank}: DP init SUCCESS")
except Exception as e:
    print(f"Rank {mpi_my_rank}: DP init ERROR: {e}")





"""

Rank 0: TP init SUCCESS
Rank 0: DP init ERROR: trying to initialize the default process group twice!
Rank 1: TP init SUCCESS
Rank 1: DP init ERROR: trying to initialize the default process group twice!
Rank 2: TP init SUCCESS
Rank 2: DP init ERROR: trying to initialize the default process group twice!
Rank 3: TP init SUCCESS
Rank 3: DP init ERROR: trying to initialize the default process group twice!
Rank 4: TP init SUCCESS
Rank 4: DP init ERROR: trying to initialize the default process group twice!
Rank 5: TP init SUCCESS
Rank 5: DP init ERROR: trying to initialize the default process group twice!
Rank 6: TP init SUCCESS
Rank 6: DP init ERROR: trying to initialize the default process group twice!
Rank 7: TP init SUCCESS
Rank 7: DP init ERROR: trying to initialize the default process group twice!
Rank 8: TP init SUCCESS
Rank 8: DP init ERROR: trying to initialize the default process group twice!
Rank 9: TP init SUCCESS
Rank 9: DP init ERROR: trying to initialize the default process group twice!
Rank 12: TP init SUCCESS
Rank 12: DP init ERROR: trying to initialize the default process group twice!
Rank 13: TP init SUCCESS
Rank 13: DP init ERROR: trying to initialize the default process group twice!
Rank 14: TP init SUCCESS
Rank 14: DP init ERROR: trying to initialize the default process group twice!
Rank 10: TP init SUCCESS
Rank 10: DP init ERROR: trying to initialize the default process group twice!
Rank 15: TP init SUCCESS
Rank 15: DP init ERROR: trying to initialize the default process group twice!
Rank 16: TP init SUCCESS
Rank 16: DP init ERROR: trying to initialize the default process group twice!
Rank 11: TP init SUCCESS
Rank 11: DP init ERROR: trying to initialize the default process group twice!
Rank 17: TP init SUCCESS
Rank 17: DP init ERROR: trying to initialize the default process group twice!
Rank 18: TP init SUCCESS
Rank 18: DP init ERROR: trying to initialize the default process group twice!
Rank 19: TP init SUCCESS
Rank 19: DP init ERROR: trying to initialize the default process group twice!
Rank 20: TP init SUCCESS
Rank 20: DP init ERROR: trying to initialize the default process group twice!
Rank 21: TP init SUCCESS
Rank 21: DP init ERROR: trying to initialize the default process group twice!
Rank 22: TP init SUCCESS
Rank 22: DP init ERROR: trying to initialize the default process group twice!
Rank 23: TP init SUCCESS
Rank 23: DP init ERROR: trying to initialize the default process group twice!
"""