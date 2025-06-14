# IMPORTS
# ----------------------------------------------------------------------------
import os
import re
import hydra
import sys
import json
import datetime
import time
from pathlib import Path
from time import perf_counter
from omegaconf import DictConfig, OmegaConf
from mpi4py import MPI


import torch

from dl_comm.utils.utility import DLIOLogger, Profile
from dl_comm.collectives import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from dl_comm.timer import timer, print_all_times, print_all_bandwidths
from dl_comm.helpers import report_ccl_selection,filter_logs_post_run, finalize_logs

log = DLIOLogger.get_instance()
dlp = Profile("DL_COMM")

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def parse_buffer_size(size_str: str) -> int:
    """Parse buffer size string (e.g., '1MB', '512KB') to bytes."""
    s = size_str.strip().upper()
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024)
    elif s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    elif s.endswith("B"):
        return int(float(s[:-1]))
    else:
        raise ValueError(f"payload.size='{size_str}' has unknown format. Use '1MB', '512KB' etc")

# ----------------------------------------------------------------------------
# VALIDATION
# ----------------------------------------------------------------------------

class ConfigValidator:
    def __init__(self, spec: dict):
        self.spec = spec

    def validate(self, cfg: DictConfig):
        errors = []

        # framework
        framework = cfg.framework
        if framework not in self.spec["framework"]:
            errors.append(
                f"Invalid framework '{framework}'. Valid options: {self.spec['framework']}"
            )

        # ccl_backend
        backend = getattr(cfg, "ccl_backend", None)
        valid_backends = self.spec["backend"].get(framework, [])
        if backend not in valid_backends:
            errors.append(
                f"Invalid ccl_backend '{backend}' for framework '{framework}'. "
                f"Valid: {valid_backends}"
            )

        # collective.name
        collective = cfg.collective.name
        if collective not in self.spec["collective"]:
            errors.append(
                f"Invalid collective '{collective}'. Valid: {self.spec['collective']}"
            )

        # collective.op
        op = cfg.collective.op
        valid_ops = self.spec["op"].get(collective, [])
        if op not in valid_ops:
            errors.append(
                f"Invalid op '{op}' for collective '{collective}'. Valid: {valid_ops}"
            )

        # collective.algo
        algo = cfg.collective.algo
        valid_algos = self.spec["algo"].get(collective, [])
        if algo not in valid_algos:
            errors.append(
                f"Invalid algo '{algo}' for collective '{collective}'. Valid: {valid_algos}"
            )

        # dtype 
        dtype = cfg.collective.payload.dtype
        if dtype not in self.spec["dtype"]:
            errors.append(
                f"Invalid dtype '{dtype}'. Valid: {self.spec['dtype']}"
            )

        # buffer_size
        try:
            buffer_bytes = parse_buffer_size(cfg.collective.payload.buffer_size)
        except ValueError as ve:
            errors.append(str(ve))

        if errors:
            raise ValueError("ALL ERRORS:\n" + "\n".join(errors))

        return buffer_bytes

# ----------------------------------------------------------------------------
# SETUP FUNCTIONS
# ----------------------------------------------------------------------------

def setup_environment(cfg: DictConfig):

    # CCL environment variables
    os.environ["CCL_ATL_TRANSPORT"] = "mpi"
    os.environ["CCL_ATL_SHM"] = "0"
    os.environ["CCL_LOG_LEVEL"] = "debug"
    os.environ["CCL_PROCESS_LAUNCHER"] = "pmix"

    os.environ["TORCH_CPP_LOG_LEVEL"] = "error"


def get_default_device(rank: int):
   
    if torch.xpu.is_available():
        return torch.device(f"xpu:{rank % torch.xpu.device_count()}")
    else:
        return torch.device("cpu")

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

   
    if mpi_rank == 0:
        log.info("-------------------------------------------------------------------------")
        log.info("[CONFIG] Loading schema and validating user YAML")
    
    # Load and validate config spec
    config_spec_path = Path(__file__).parent / "config" / "config_spec.json"
    with open(config_spec_path, "r") as f:
        spec = json.load(f)
    
    validator = ConfigValidator(spec)
    buffer_in_bytes = validator.validate(cfg)
    
   
    setup_environment(cfg)
    
    # Import framework-specific modules after setting env vars
    if cfg.framework == "pytorch" and cfg.ccl_backend == "xccl":
        with timer("import time"):
            import intel_extension_for_pytorch
            import oneccl_bindings_for_pytorch
            import torch.nn.parallel
            import torch.distributed as dist
    
    # Extract configuration values
    framework   = cfg.framework
    coll_name   = cfg.collective.name
    op_name     = cfg.collective.op
    dtype_str   = cfg.collective.payload.dtype
    iters       = cfg.collective.iterations
    tp_size     = cfg.horizontal.tp_degree
    dp_size     = cfg.vertical.dp_degree
    flatview    = cfg.get("flatview", False)
    
    # Verify world size matches config
    expected_world_size = tp_size * dp_size
    """    if mpi_size != expected_world_size:
        if mpi_rank == 0:
            log.error(f"World size mismatch: MPI size={mpi_size}, expected={expected_world_size}")
        sys.exit(1)"""
    
    
    if mpi_rank == 0:
        log.info("[CONFIG] Final validated settings\n")
        log.info(f"  • framework           = {framework}")
        log.info(f"  • backend             = {cfg.ccl_backend}")
        log.info(f"  • collective_name     = {coll_name}")
        log.info(f"  • op                  = {op_name}")
        log.info(f"  • algo                = {cfg.collective.algo}")
        log.info(f"  • buffer_size         = {cfg.collective.payload.buffer_size} ({buffer_in_bytes} bytes)")
        log.info(f"  • dtype               = {dtype_str}")
        log.info(f"  • horizontal.num_gpus = {tp_size}")
        log.info(f"  • vertical.num_nodes  = {dp_size}")
        log.info(f"  • use_unitrace        = {cfg.get('use_unitrace', False)}")
        log.info("-------------------------------------------------------------------------")
    
   
    torch_dtype, elem_size = DTYPES[dtype_str]
    run_collective = COLLECTIVES[coll_name]
    op_obj = OP_MAP[op_name] if coll_name in OPS_NEED_REDUCE else None


    if mpi_rank == 0:
        log.info("[MPI] Computing rank counts")
    
        num_nodes = cfg.vertical.dp_degree
        ranks_per_node = cfg.horizontal.tp_degree
        total_ranks = num_nodes * ranks_per_node
        log.info(f"[MPI] num_nodes       = {num_nodes}")
        log.info(f"[MPI] ranks_per_node  = {ranks_per_node}")
        log.info(f"[MPI] total_ranks     = {total_ranks}")
        log.info(f"\n")

    if mpi_rank == 0:
        import socket
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2319
    else:
        MASTER_ADDR = None
        MASTER_PORT = None
    
    MASTER_ADDR = mpi_comm.bcast(MASTER_ADDR, root=0)
    MASTER_PORT = mpi_comm.bcast(MASTER_PORT, root=0)
    
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    
     
    mpi_comm.Barrier()
    with timer("init time"):
        dist.init_process_group(
            backend="ccl",
            init_method='env://',
            world_size=mpi_size,
            rank=mpi_rank,
            timeout=datetime.timedelta(seconds=3600)
        )
    mpi_comm.Barrier()
    
    # Build TP and DP groups
    tp_rank = mpi_rank % tp_size
    dp_rank = mpi_rank // tp_size
    
    # Tensor-parallel ranks
    tp_ranks = [n * tp_size + tp_rank for n in range(dp_size)]
    
    # Data-parallel ranks
    dp_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
    
    
    tp_group = dist.new_group(ranks=tp_ranks)
    dp_group = dist.new_group(ranks=dp_ranks)
    world_group = dist.group.WORLD
    
    # Get device and prepare buffer
    device = get_default_device(mpi_rank)
    num_elems = buffer_in_bytes // elem_size
    
    if mpi_rank == 0:
        log.info("")
        log.info("[MPI][SETUP] ------------------------------------------------------")
        log.info(f"[MPI][SETUP] Framework      : {framework}")
        log.info(f"[MPI][SETUP] Collective     : {coll_name}")
        log.info(f"[MPI][SETUP] Operation      : {op_name if op_obj else 'N/A'}")
        log.info(f"[MPI][SETUP] DType          : {dtype_str}")
        log.info(f"[MPI][SETUP] Buffer Size    : {buffer_in_bytes}")
        log.info(f"[MPI][SETUP] Iterations     : {iters}")
        log.info(f"[MPI][SETUP] World Size     : {mpi_size}")
        log.info("[MPI][SETUP] ------------------------------------------------------")
        log.info("")
        log.info("[MPI] Launching profiling job")


    for _ in range(iters):
        x = torch.ones(num_elems, dtype=torch_dtype).to(device, non_blocking=True)
        
        if not flatview:
            with timer("Latencies (s) (TP)"):
                run_collective(x, op_obj, group=tp_group)
                mpi_comm.Barrier()
            
            with timer("Latencies (s) (DP)"):
                run_collective(x, op_obj, group=dp_group)
                mpi_comm.Barrier()
        else:
            with timer("Latencies (s) (Flatview)"):
                run_collective(x, op_obj, group=world_group)
                mpi_comm.Barrier()
    
 
    if mpi_rank == 0:
        print_all_times()
        print_all_bandwidths(buffer_in_bytes, coll_name)
        
      
        log.info("-------------------------------------------------------------------------")
        log.info("[MPI] Job complete")
        log.info("-------------------------------------------------------------------------")
        
         
        log.info("[LOG_FILTER] Starting log separation...")
        ccl_log_path, terminal_log_path, output_dir = filter_logs_post_run(log)

        log.info("Parsing selection")

        report_ccl_selection(ccl_log_path, cfg.collective.name, log)
        log.info("-------------------------------------------------------------------------")
        log.info("[EXIT] All Done.")
        log.info("-------------------------------------------------------------------------")



  
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        
       
        time.sleep(0.5)
        

         
        finalize_logs(output_dir, log)



if __name__ == "__main__":
    main()