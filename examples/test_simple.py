#!/usr/bin/env python3
import os
import time
import json
import socket
import datetime
from pathlib import Path
from mpi4py import MPI
import torch
import torch.distributed as dist

# ============================================================
# Config knobs
# ============================================================
NUM_CYCLES = 2                # how many re-init/destroy cycles
BASE_PORT  = 29500            # first port, incremented each cycle
PRINT_TO_STDOUT = True        # set False to silence stdout (files only)

# ============================================================
# Logging utilities
# ============================================================
class RankLogger:
    def __init__(self, rank, log_dir):
        self.rank = rank
        self.dir = Path(log_dir)
        if rank == 0:
            self.dir.mkdir(parents=True, exist_ok=True)
        MPI.COMM_WORLD.Barrier()  # ensure dir exists before others open
        self.path = self.dir / f"rank{rank}.log"
        # line-buffered text file
        self.f = open(self.path, "w", buffering=1)

    def log(self, **fields):
        """
        Write one log line with ISO timestamp and fields as key=value.
        """
        ts = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        # basic flatten: msg is str, others to str
        parts = [f"time={ts}", f"g_rank={self.rank}"] + [f"{k}={v}" for k, v in fields.items()]
        line = " | ".join(parts)
        self.f.write(line + "\n")
        if PRINT_TO_STDOUT:
            # minimal mirror to stdout so you can see progress live
            print(f"[G{self.rank}] {fields.get('msg','')}", flush=True)

    def close(self):
        self.f.close()

def make_log_dir():
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"dist_logs_{stamp}"

def combine_logs(log_dir, world_size):
    """
    Rank 0 merges all rank logs and sorts by 'time=' field.
    """
    combined_path = Path(log_dir) / "combined.log"
    lines = []
    for r in range(world_size):
        rp = Path(log_dir) / f"rank{r}.log"
        if rp.exists():
            with rp.open("r") as f:
                lines.extend(f.readlines())
    # Parse timestamp (first key after 'time=')
    def key_fn(line):
        # line starts with time=2025-07-25T20:33:11.123Z | g_rank=...
        try:
            first = line.split("|", 1)[0].strip()  # "time=..."
            tstr = first.split("=", 1)[1]
            return tstr
        except Exception:
            return line  # fallback
    lines.sort(key=key_fn)
    with combined_path.open("w") as out:
        out.writelines(lines)

# ============================================================
# Device policy (edit to your taste)
# ============================================================
def pick_device_for_cycle(local_rank_node, local_gpu_count, cycle_id):
    """
    Example policy:
      cycle 0: device = local_rank_node
      cycle 1: device = reverse(local_rank_node)
    """
    if local_gpu_count == 0:
        return None
    if cycle_id % 2 == 0:
        return local_rank_node % local_gpu_count
    else:
        return (local_gpu_count - 1 - (local_rank_node % local_gpu_count)) % local_gpu_count

# ============================================================
# Main test
# ============================================================
def test_init_destroy_cycle():
    # MPICH quirk
    os.environ["MPICH_GPU_SUPPORT_ENABLED"] = os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0")

    comm_world = MPI.COMM_WORLD
    mpi_rank   = comm_world.Get_rank()
    mpi_size   = comm_world.Get_size()

    log_dir = make_log_dir()
    logger = RankLogger(mpi_rank, log_dir)

    # Minimal world print
    if mpi_rank == 0 and PRINT_TO_STDOUT:
        print(f"[World] size={mpi_size}", flush=True)

    # Master address once
    master_addr = comm_world.bcast(socket.gethostname() if mpi_rank == 0 else None, root=0)
    os.environ["MASTER_ADDR"] = master_addr

    # Device discovery
    if torch.cuda.is_available():
        dev_type      = "cuda"
        per_proc_devs = torch.cuda.device_count()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        dev_type      = "xpu"
        per_proc_devs = torch.xpu.device_count()
    else:
        dev_type      = "cpu"
        per_proc_devs = 1

    # Node-local rank for per-node GPU mapping
    node_comm        = comm_world.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank_node  = node_comm.Get_rank()
    local_size_node  = node_comm.Get_size()
    node_comm.Free()

    if mpi_rank == 0:
        logger.log(msg=f"Devices per process: {per_proc_devs} ({dev_type})",
                   step="device_discovery")

    # Define cycles (all ranks active each time here)
    group_configs = [
        {"name": f"Group {i+1}", "active_ranks": list(range(mpi_size))}
        for i in range(NUM_CYCLES)
    ]

    for cycle_id, cfg in enumerate(group_configs):
        # -------- Cycle header --------
        if mpi_rank == 0:
            logger.log(msg=f"=== {cfg['name']} start === Active: {cfg['active_ranks']}",
                       step="cycle_start",
                       cycle=cycle_id)

        in_group = mpi_rank in cfg["active_ranks"]
        color    = 1 if in_group else MPI.UNDEFINED
        subcomm  = comm_world.Split(color=color, key=mpi_rank)

        if not in_group:
            # Ranks not participating just wait
            comm_world.Barrier()
            continue

        # Rank within subgroup
        local_rank_pg = cfg["active_ranks"].index(mpi_rank)
        world_size_pg = len(cfg["active_ranks"])

        # Pick unique device for this cycle
        dev_id = pick_device_for_cycle(local_rank_node,
                                       per_proc_devs if dev_type != "cpu" else 0,
                                       cycle_id)

        # Set env for this PG
        os.environ["WORLD_SIZE"]  = str(world_size_pg)
        os.environ["RANK"]        = str(local_rank_pg)
        os.environ["MASTER_PORT"] = str(BASE_PORT + cycle_id)
        os.environ["LOCAL_RANK"]  = str(local_rank_node)

        # Set device before init
        if dev_type == "cuda" and dev_id is not None:
            torch.cuda.set_device(dev_id)
            device = torch.device(f"cuda:{dev_id}")
        elif dev_type == "xpu" and dev_id is not None:
            device = torch.device(f"xpu:{dev_id}")
        else:
            device = torch.device("cpu")

        logger.log(step="assign_device",
                   cycle=cycle_id,
                   msg=f"PG {local_rank_pg}/{world_size_pg}, nodeLocal={local_rank_node}, dev={device}",
                   dev=str(device),
                   pg_rank=local_rank_pg,
                   pg_world=world_size_pg,
                   node_local_rank=local_rank_node)

        subcomm.Barrier()
        logger.log(step="after_subcomm_barrier", cycle=cycle_id,
                   msg="ready to init PG",
                   env=json.dumps({
                       "RANK": os.environ["RANK"],
                       "WORLD_SIZE": os.environ["WORLD_SIZE"],
                       "MASTER_ADDR": os.environ["MASTER_ADDR"],
                       "MASTER_PORT": os.environ["MASTER_PORT"]
                   }))

        backend = "nccl" if dev_type == "cuda" else "gloo"
        logger.log(step="init_process_group", cycle=cycle_id,
                   msg=f"calling init (backend={backend})")

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size_pg,
            rank=local_rank_pg,
            timeout=datetime.timedelta(seconds=120),
        )

        logger.log(step="pg_init_ok", cycle=cycle_id, msg="PG initialized")

        # ---- Collective test ----
        tensor = torch.ones(3, device=device, dtype=torch.float32) * local_rank_pg
        logger.log(step="before_allreduce", cycle=cycle_id,
                   msg=f"tensor={tensor.tolist()}")

        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        logger.log(step="allreduce_launched", cycle=cycle_id, msg="waiting...")
        work.wait()
        logger.log(step="allreduce_done", cycle=cycle_id,
                   msg=f"tensor={tensor.tolist()}")

        expected = sum(range(world_size_pg))
        check = torch.full_like(tensor, expected)
        if not torch.allclose(tensor, check):
            logger.log(step="allreduce_check_fail", cycle=cycle_id,
                       msg=f"Mismatch got {tensor.tolist()} expected {[expected]*3}")

        # ---- Tear down ----
        dist.destroy_process_group()
        if dev_type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.log(step="pg_destroyed", cycle=cycle_id, msg="PG destroyed")

        subcomm.Free()
        comm_world.Barrier()
        time.sleep(0.2)

        if mpi_rank == 0:
            logger.log(step="cycle_end", cycle=cycle_id,
                       msg=f"=== {cfg['name']} end ===")

    # --------- Combine logs on rank0 ---------
    comm_world.Barrier()
    logger.close()
    if mpi_rank == 0:
        combine_logs(log_dir, mpi_size)
        if PRINT_TO_STDOUT:
            print(f"\nLogs written to: {log_dir}\nCombined log: {log_dir}/combined.log", flush=True)


if __name__ == "__main__":
    # Modern torch debug toggles (optional)
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("NCCL_DEBUG", "INFO")

    test_init_destroy_cycle()
