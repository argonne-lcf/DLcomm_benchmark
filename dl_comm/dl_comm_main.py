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

from dl_comm.utils.utility import DLCOMMLogger, Profile
from dl_comm.collectives import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from dl_comm.timer import timer, print_all_times, print_all_bandwidths
from dl_comm.helpers import report_ccl_selection

 
 

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def parse_buffer_size(size_str: str) -> int:
    """Parse buffer size string (e.g., '1GB', '1MB', '512KB') to bytes."""
    s = size_str.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1024 * 1024 * 1024)
    elif s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024)
    elif s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    elif s.endswith("B"):
        return int(float(s[:-1]))
    else:
        raise ValueError(f"payload.size='{size_str}' has unknown format. Use '1GB', '1MB', '512KB' etc")

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
    os.environ["FI_MR_CACHE_MONITOR"] = "userfaultfd"

def get_default_device(rank: int, gpu_ids: list = None):
    if torch.xpu.is_available():
        if gpu_ids and rank < len(gpu_ids):
            return torch.device(f"xpu:{gpu_ids[rank]}")
        else:
            return torch.device(f"xpu:{rank % torch.xpu.device_count()}")
    else:
        return torch.device("cpu")

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

   
    if mpi_rank == 0:
        # Always create timestamped log directory
        if "DL_COMM_LOG_DIR" in os.environ:
            # Use directory provided by jobscript (already timestamped)
            log_dir = os.environ["DL_COMM_LOG_DIR"]
        else:
            # Create timestamped directory for manual runs
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_dir = f"logs/run_{timestamp}"
        
        os.makedirs(log_dir, exist_ok=True)
        log = DLCOMMLogger.get_instance(log_file="dlcomm.log", log_dir=log_dir)
        log.info("-------------------------------------------------------------------------")
      
        log.info("[CONFIG] Loading schema and validating user YAML")
    else:
         
        class DummyLogger:
            def info(self, msg): pass
            def debug(self, msg): pass
            def warning(self, msg): pass
            def error(self, msg): pass
            def output(self, msg): pass
        log = DummyLogger()

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

    # NEW: Extract comm_group configuration
    comm_mode = cfg.collective.comm_group.mode
    

    if comm_mode == "hierarchical":
        within_size = cfg.collective.comm_group.hierarchical.within_node.num_gpus
        across_size = mpi_size // within_size  # Calculate from world size
        num_nodes = across_size
        num_gpus = within_size
    elif comm_mode == "flatview":
        within_size = mpi_size
        across_size = 1
        num_nodes = 1
        num_gpus = mpi_size
    else:
        raise ValueError(f"Unknown comm_mode: '{comm_mode}'. Valid options: 'within-node', 'across-node', 'hierarchical', 'flatview'") 
        
    # Verify world size matches config
    expected_world_size = within_size * across_size
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
        log.info(f"  • count               = {cfg.collective.payload.count}")
        log.info(f"  • comm_mode           = {comm_mode}")
        log.info(f"  • num_nodes           = {num_nodes}")
        log.info(f"  • num_gpus            = {num_gpus}")
        log.info(f"  • use_profiler        = {cfg.get('use_profiler', 'none')}")
        log.info("-------------------------------------------------------------------------")
    
    torch_dtype, elem_size = DTYPES[dtype_str]
    run_collective = COLLECTIVES[coll_name]
    #run_collective=dist.allreduce()
    op_obj = OP_MAP[op_name] if coll_name in OPS_NEED_REDUCE else None

 
    if mpi_rank == 0:
        import socket
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2314
    else:
        MASTER_ADDR = None
        MASTER_PORT = None
    
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    MASTER_PORT = MPI.COMM_WORLD.bcast(MASTER_PORT, root=0)
    
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    
     
    MPI.COMM_WORLD.Barrier()
    with timer("init time"):
        dist.init_process_group(
            backend="ccl",
            init_method='env://',
            world_size=mpi_size,
            rank=mpi_rank,
            timeout=datetime.timedelta(seconds=3600)
        )
   
   
    
    # Build within and across groups
    within_rank = mpi_rank % within_size
    across_rank = mpi_rank // within_size
    
    
    if comm_mode == "hierarchical":
        # Create within groups (intra-node): [0-11], [12-23] 
        within_groups = []
        for i in range(across_size):
            within_group_ranks = list(range(i * within_size, (i + 1) * within_size))
            within_group = dist.new_group(within_group_ranks)
            within_groups.append(within_group)
            if mpi_rank == 0:  # Only rank 0 logs group creation to avoid duplicates
                log.info(f"[GROUP CREATION] Within group {i}: ranks {within_group_ranks}")
        my_within_group = within_groups[across_rank]
        
        # Create across groups (inter-node): [0,12], [1,13], [2,14], etc.
        across_groups = []
        for i in range(within_size):
            across_group_ranks = list(range(i, mpi_size, within_size))
            across_group = dist.new_group(across_group_ranks)
            across_groups.append(across_group)
            if mpi_rank == 0:  # Only rank 0 logs group creation to avoid duplicates
                log.info(f"[GROUP CREATION] Across group {i}: ranks {across_group_ranks}")
        my_across_group = across_groups[within_rank]
        
        within_group = my_within_group
        across_group = my_across_group
        
        log.info(f"[RANK {mpi_rank}] within_rank={within_rank}, across_rank={across_rank}, my_within_group_id={across_rank}, my_across_group_id={within_rank}")
        log.info(f"[RANK {mpi_rank}] within_group={within_group}, across_group={across_group}")
    elif comm_mode == "flatview":
        # For flatview mode, use world_group 
        world_group = dist.group.WORLD
    else:
        raise ValueError(f"Unknown comm_mode: '{comm_mode}'. Valid options: 'hierarchical', 'flatview'")
    
    # Get device and prepare buffer
    if comm_mode == "hierarchical":
        # For hierarchical mode, use across_node GPU IDs for device mapping
        across_gpu_ids = cfg.collective.comm_group.hierarchical.across_node.gpu_ids_per_node
        device_id = across_gpu_ids[within_rank % len(across_gpu_ids)]
        device = torch.device(f"xpu:{device_id}") if torch.xpu.is_available() else torch.device('cpu')
        log.info(f"[RANK {mpi_rank}] Device assignment: within_rank={within_rank}, device_id={device_id}, device={device}")
    elif comm_mode == "flatview":
        device = get_default_device(mpi_rank)
        log.info(f"[RANK {mpi_rank}] Flatview device assignment: device={device}")
    else:
        raise ValueError(f"Unknown comm_mode: '{comm_mode}'. Valid options: 'hierarchical', 'flatview'")
        
        
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
        
        if comm_mode == "hierarchical":
            with timer("Total i(Within→Across)"):
                with timer("Latencies (s) (Within)"):
                    run_collective(x, op_obj, group=within_group)
                    MPI.COMM_WORLD.Barrier()
                
                with timer("(Across)"):
                    run_collective(x, op_obj, group=across_group)
                    MPI.COMM_WORLD.Barrier()

        elif comm_mode == "flatview":
            with timer("(Flatview)"):
                run_collective(x, op_obj, group=world_group)
                MPI.COMM_WORLD.Barrier()
        else:
            raise ValueError(f"Unknown comm_mode: '{comm_mode}'. Valid options: 'hierarchical', 'flatview'")
    
 
    if mpi_rank == 0:
        print_all_times(log)
        print_all_bandwidths(log, buffer_in_bytes, coll_name)
        
      
        log.info("-------------------------------------------------------------------------")
        log.info("[MPI] Job complete")
        log.info("-------------------------------------------------------------------------")
        
         
       

        log.info("Querying Default Table selection")

  
        terminal_log_path = os.path.join(log_dir, "terminal_output.log")
        if os.path.exists(terminal_log_path):
            report_ccl_selection(terminal_log_path, cfg.collective.name, log)
        else:
            log.info(f"[SELECTION] Terminal output log not found: {terminal_log_path}")


        log.info("-------------------------------------------------------------------------")
        log.info("[EXIT] All Done.")
        log.info("-------------------------------------------------------------------------")



  
        
    
        DLCOMMLogger.flush()
        DLCOMMLogger.reset()
            
       
       



if __name__ == "__main__":
    main()