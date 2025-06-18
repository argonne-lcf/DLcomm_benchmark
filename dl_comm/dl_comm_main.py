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
import pytz

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

        # collective.scale_up_algorithm
        scale_up_algo = cfg.collective.scale_up_algorithm
        valid_algos = self.spec["algo"].get(collective, [])
        if scale_up_algo not in valid_algos:
            errors.append(
                f"Invalid scale_up_algorithm '{scale_up_algo}' for collective '{collective}'. Valid: {valid_algos}"
            )
            
        # collective.scale_out_algorithm  
        scale_out_algo = cfg.collective.scale_out_algorithm
        if scale_out_algo not in valid_algos:
            errors.append(
                f"Invalid scale_out_algorithm '{scale_out_algo}' for collective '{collective}'. Valid: {valid_algos}"
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

        # comm_group validation
        comm_group = cfg.collective.comm_group
        comm_mode = comm_group.mode
        valid_modes = ["within_node", "across_node", "combined", "flatview"]
        
        if comm_mode not in valid_modes:
            errors.append(f"Invalid comm_mode '{comm_mode}'. Valid: {valid_modes}")
        
        # Mode-specific validation
        if comm_mode == "within_node":
            if not hasattr(comm_group, 'within_node'):
                errors.append("comm_mode 'within_node' requires 'within_node' configuration")
            else:
                within_config = comm_group.within_node
                if not hasattr(within_config, 'num_gpus_per_node') or not hasattr(within_config, 'gpu_ids_per_node'):
                    errors.append("within_node config requires 'num_gpus_per_node' and 'gpu_ids_per_node'")
        
        elif comm_mode == "across_node":
            if not hasattr(comm_group, 'across_node'):
                errors.append("comm_mode 'across_node' requires 'across_node' configuration")
            else:
                across_config = comm_group.across_node
                if not hasattr(across_config, 'num_compute_nodes') or not hasattr(across_config, 'num_gpus_per_node') or not hasattr(across_config, 'gpu_ids_per_node'):
                    errors.append("across_node config requires 'num_compute_nodes', 'num_gpus_per_node' and 'gpu_ids_per_node'")
        
        elif comm_mode == "combined":
            if not hasattr(comm_group, 'combined'):
                errors.append("comm_mode 'combined' requires 'combined' configuration")
            else:
                combined_config = comm_group.combined
                if not hasattr(combined_config, 'within_node') or not hasattr(combined_config, 'across_node'):
                    errors.append("combined config requires both 'within_node' and 'across_node' sub-configurations")

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
    os.environ["CCL_LOG_LEVEL"] ="debug"
    os.environ["CCL_PROCESS_LAUNCHER"] = "pmix"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "error"
    os.environ["FI_MR_CACHE_MONITOR"] = "userfaultfd"

    scale_up_override = f"CCL_{cfg.collective.name.upper()}"
    os.environ[scale_up_override]=cfg.collective.scale_up_algorithm

    scale_out_override = f"CCL_{cfg.collective.name.upper()}_SCALEOUT"
    os.environ[scale_out_override]=cfg.collective.scale_out_algorithm



def get_default_device(rank: int):
    if torch.xpu.is_available():
        return torch.device(f"xpu:{rank % torch.xpu.device_count()}")
    else:
        return torch.device("cpu")

def setup_communication_groups(cfg: DictConfig, mpi_rank: int, mpi_size: int, log, dist=None):
 
    
    comm_config = cfg.collective.comm_group
    comm_mode = comm_config.mode
    
  
    my_within_group = None
    my_across_group = None
    world_group = None
    device = None
    within_size = None
    across_size = None
    
    if mpi_rank == 0:
        log.info(f"[COMM] Setting up communication groups for mode: {comm_mode}")
    
    if comm_mode == "within_node":
 
        
     
        within_config = comm_config.within_node
        num_gpus = within_config.num_gpus_per_node
        num_nodes=within_config.num_compute_nodes
        gpu_ids_per_node = within_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM] Within-node config: {num_gpus} GPUs, IDs: {gpu_ids_per_node}")
        
  
        within_groups = []
        for node in range(num_nodes):
            group_ranks = []
            for gpu in range(num_gpus):
                rank = node * num_gpus + gpu
                group_ranks.append(rank)
            within_groups.append(dist.new_group(ranks=group_ranks))
        
        node_id = mpi_rank // num_gpus
        my_within_group = within_groups[node_id] 

        
        
        gpu_idx = mpi_rank % num_gpus
        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[gpu_idx]
            device = torch.device(f"xpu:{device_id}")
 
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")
        
        within_size = num_gpus
        across_size = num_nodes  
        
        if mpi_rank == 0:
            log.info(f"[COMM] Within group size: {within_size}, using world group for communication")
        
    elif comm_mode == "across_node":
        
        if mpi_rank == 0:
            log.info("[COMM] Setting up across-node groups")
        
        across_config = comm_config.across_node
        num_nodes = across_config.num_compute_nodes
        num_gpus = across_config.num_gpus_per_node
        gpu_ids_per_node = across_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM] Across-node config: {num_nodes} nodes, {num_gpus} GPUs per node, IDs: {gpu_ids_per_node}")
 
        
        across_groups = []
        for i in range(num_gpus):
            group_ranks = []
            for node in range(num_nodes):
                rank = node * num_gpus + i
                group_ranks.append(rank)
            across_groups.append(dist.new_group(ranks=group_ranks))
        
      
        gpu_index = mpi_rank % num_gpus
        node_id = mpi_rank // num_gpus
        my_across_group = across_groups[gpu_index]
        
         
        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[gpu_index]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")
        
        within_size = 1  
        across_size = num_nodes
        
        if mpi_rank == 0:
            log.info(f"[COMM] Across group size: {across_size}, created {len(across_groups)} across-node groups")
        
    elif comm_mode == "combined":
       
        if mpi_rank == 0:
            log.info("[COMM] Setting up combined (within + across) groups")
        
       
        within_config = comm_config.combined.within_node
        across_config = comm_config.combined.across_node
        
        num_gpus = within_config.num_gpus_per_node
        gpu_ids_per_node = within_config.gpu_ids_per_node
        num_nodes = across_config.num_compute_nodes
        across_gpu_ids = across_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM] Combined config - Within: {num_gpus} GPUs {gpu_ids_per_node}, Across: {num_nodes} nodes {across_gpu_ids}")
        
         
        
        
        within_groups = []
        for node in range(num_nodes):
            group_ranks = []
            for gpu in range(num_gpus):
                rank = node * num_gpus + gpu
                group_ranks.append(rank)
            within_groups.append(dist.new_group(ranks=group_ranks))
        
     
        node_id = mpi_rank // num_gpus
        gpu_index = mpi_rank % num_gpus
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
        
   

        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[gpu_index]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")
        
        within_size = num_gpus
        across_size = num_nodes if my_across_group is not None else 1
        
        if mpi_rank == 0:
            log.info(f"[COMM] Created {len(within_groups)} within-node groups and {len(across_groups)} across-node groups")
            log.info(f"[COMM] Within group size: {within_size}, Across group size: {across_size}")
        
    elif comm_mode == "flatview":
         
        if mpi_rank == 0:
            log.info("[COMM] Using flatview (world group)")
            
        world_group = None  # Use default world group
        device = get_default_device(mpi_rank)
        within_size = mpi_size
        across_size = 1
        
    else:
        raise ValueError(f"Unknown comm_mode: '{comm_mode}'. Valid: within, across, combined, flatview")
    
    return {
        'my_within_group': my_within_group,
        'my_across_group': my_across_group, 
        'world_group': world_group,
        'device': device,
        'within_size': within_size,
        'across_size': across_size
    }

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

 
    if mpi_rank == 0:      
        if "DL_COMM_LOG_DIR" in os.environ:
            
            log_dir = os.environ["DL_COMM_LOG_DIR"]
        else:
            
            
            chicago_tz = pytz.timezone('America/Chicago')
            timestamp = datetime.datetime.now(chicago_tz).strftime("%Y%m%d_%H%M%S_%f")
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
    
    if mpi_rank == 0:
        log.info("-------------------------------------------------------------------------")
        log.info("[CONFIG] Loading schema and validating user YAML")

    config_spec_path = Path(__file__).parent / "config" / "config_spec.json"
    with open(config_spec_path, "r") as f:
        spec = json.load(f)
    
    validator = ConfigValidator(spec)
    buffer_in_bytes = validator.validate(cfg)
    
   
    setup_environment(cfg)
    
 
    if cfg.framework == "pytorch" and cfg.ccl_backend == "xccl":
        with timer("import time"):
            import intel_extension_for_pytorch
            import oneccl_bindings_for_pytorch
            import torch.nn.parallel
            import torch.distributed as dist
    
    
    framework           = cfg.framework
    coll_name           = cfg.collective.name
    op_name             = cfg.collective.op
    dtype_str           = cfg.collective.payload.dtype
    iters               = cfg.collective.iterations
    enable_correctness  = cfg.collective.verify_correctness 
    comm_mode           = cfg.collective.comm_group.mode
    
     
 

    
    if mpi_rank == 0:
        log.info("[CONFIG] Final validated settings\n")
        log.info(f"  • framework           = {framework}")
        log.info(f"  • backend             = {cfg.ccl_backend}")
        log.info(f"  • collective_name     = {coll_name}")
        log.info(f"  • op                  = {op_name}")
        log.info(f"  • scale_up_algo       = {cfg.collective.scale_up_algorithm}")
        log.info(f"  • scale_out_algo      = {cfg.collective.scale_out_algorithm}")
        log.info(f"  • buffer_size         = {cfg.collective.payload.buffer_size} ({buffer_in_bytes} bytes)")
        log.info(f"  • dtype               = {dtype_str}")
        log.info(f"  • count               = {cfg.collective.payload.count}")
        log.info(f"  • comm_mode           = {comm_mode}")
        log.info(f"  • use_profiler        = {cfg.get('use_profiler', 'none')}")
        log.info("-------------------------------------------------------------------------")
    #coll_name   = cfg.collective.name
    torch_dtype, elem_size = DTYPES[dtype_str]
    run_collective = COLLECTIVES[coll_name]
    #run_collective=dist.allreduce()
    op_obj = OP_MAP[op_name] if coll_name in OPS_NEED_REDUCE else None

 
    if mpi_rank == 0:
        import socket
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2357
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
    
    
    comm_info = setup_communication_groups(cfg, mpi_rank, mpi_size, log, dist)
    my_within_group = comm_info['my_within_group']
    my_across_group = comm_info['my_across_group'] 
    world_group = comm_info['world_group']
    device = comm_info['device']
    within_size = comm_info['within_size']
    across_size = comm_info['across_size']
   
   
    
 
 
        
        
    num_elems = buffer_in_bytes // elem_size
    
   
    MPI.COMM_WORLD.Barrier()
    
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
        log.info("")
        log.info("[MPI][SETUP] ------------------------------------------------------")
        log.info("")
        log.info("[MPI] Launching profiling job")


    for i in range(iters):
        x = torch.ones(num_elems, dtype=torch_dtype).to(device, non_blocking=True)
        
        if mpi_rank == 0 and i==0 and enable_correctness:
            log.info(f"[CORRECTNESS CHECK] Before collective {x.sum()}")
        

        if comm_mode == "flatview":
            with timer("(Flatview)"):
                run_collective(x, op_obj, group=world_group)
                MPI.COMM_WORLD.Barrier()



        elif comm_mode == "within_node":
            with timer("(Within)"):
                run_collective(x, op_obj, group=my_within_group)
                MPI.COMM_WORLD.Barrier()



        elif comm_mode == "across_node":
            with timer("(Across)"):
                run_collective(x, op_obj, group=my_across_group)
                MPI.COMM_WORLD.Barrier()




        elif comm_mode == "combined":
            with timer("Total (Within→Across)"):
                with timer("(Within)"):
                    run_collective(x, op_obj, group=my_within_group)
                    MPI.COMM_WORLD.Barrier()
                
                with timer("(Across)"):
                    if my_across_group:
                        run_collective(x, op_obj, group=my_across_group)
                    MPI.COMM_WORLD.Barrier()

 

        else:
            raise ValueError(f"Unknown comm_mode: '{comm_mode}'. Valid options: 'within_node', 'across_node', 'combined', 'flatview'")
        
        if mpi_rank == 0 and i==0 and enable_correctness:
            log.info(f"[CORRECTNESS CHECK] After collective {x.sum()}")
    
 
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

        #destroy_process_group()
        # pytorch/torch/distributed/distributed_c10d.py  line 2094
            
       
       



if __name__ == "__main__":
    main()