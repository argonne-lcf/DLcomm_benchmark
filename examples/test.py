#!/usr/bin/env python3
import os, socket, torch, torch.distributed as dist
from mpi4py import MPI

CYCLES = [{0:0, 1:1}, {0:2, 1:3}]     # cycle -> {rank: gpu}
BASE_PORT = 29500

comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
world  = comm.Get_size()
master = comm.bcast(socket.gethostname() if rank == 0 else None, root=0)

# Init a small "control" PG (gloo so GPUs can overlap safely here)
os.environ.update({"MASTER_ADDR": master, "MASTER_PORT": str(BASE_PORT),
                   "WORLD_SIZE": str(world), "RANK": str(rank)})
dist.init_process_group("nccl", init_method="env://", world_size=world, rank=rank)

log = open(f"rank{rank}.log", "w", buffering=1)
L   = lambda s: log.write(f"[r{rank}] {s}\n")

for i, mapping in enumerate(CYCLES):
    gpu = mapping[rank]
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    grp_ranks = list(mapping.keys())
    grp = dist.new_group(ranks=grp_ranks, backend="nccl")   # new communicator for this cycle

    L(f"cycle {i}: rank->{rank} gpu->{gpu} group->{grp_ranks}")

    t = torch.arange(8, device=device, dtype=torch.float32) + rank * 100
    dist.all_reduce(t, group=grp)
    L(f"cycle {i}: all_reduce sum -> {t.tolist()}")

    dist.destroy_process_group(grp)
    #torch.cuda.empty_cache()
    comm.Barrier()

log.close()
dist.destroy_process_group()
if rank == 0:
    print("Done. See rank0.log, rank1.log", flush=True)
