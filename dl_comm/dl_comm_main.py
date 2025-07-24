# ----------------------------------------------------------------------------
# OVERALL STRUCTURE
# ----------------------------------------------------------------------------

# dl_comm/
# ├── dl_comm_main.py    # main(), setup_environment()
# ├── analysis/          # CCL parsing + bandwidth analysis
# │   ├── ccl_parser.py     # parse_ccl_selection(), report_ccl_selection()
# │   └── bandwidth.py      # bytes_per_rank(), bytes_per_coll(), print_all_bandwidths()
# ├── comm/             
# │   ├── comm_setup.py     # setup_communication_groups()
# │   └── collectives.py    # COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
# ├── config/          
# │   └── validation.py     # ConfigValidator, parse_buffer_size()
# ├── timer/           
# │   └── timer.py          # timer(), print_all_times()
# └── utils/            
#     └── utility.py        # DLCOMMLogger, Profile

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------

import os
import re
import sys
import json
import time
import pytz
import torch
import hydra
import socket
import datetime
from mpi4py import MPI
from pathlib import Path
from time import perf_counter
from omegaconf import DictConfig, OmegaConf
# dl_comm packages
from dl_comm.comm import setup_communication_groups
from dl_comm.utils.utility import DLCOMMLogger, Profile
from dl_comm.analysis.correctness import check_collective_correctness
from dl_comm.comm import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from dl_comm.analysis import report_ccl_selection, print_all_bandwidths 
from dl_comm.timer import timer, print_all_times, gather_and_print_all_times, reset_times
from dl_comm.config import ConfigValidator, parse_buffer_size, print_system_info, 
from dl_comm.config import adjust_buffer_size_for_group_divisibility, validate_mpi_configuration

# ----------------------------------------------------------------------------
# SETUP FUNCTIONS
# ----------------------------------------------------------------------------

def setup_collective_algorithms(cfg: DictConfig, coll_cfg, comm_mode: str):

    if cfg.ccl_debug:
        os.environ["CCL_LOG_LEVEL"] = "debug"
    scale_up_override = f"CCL_{coll_cfg.name.upper()}"
    os.environ[scale_up_override] = coll_cfg.scale_up_algorithm

    scale_out_override = f"CCL_{coll_cfg.name.upper()}_SCALEOUT"
    os.environ[scale_out_override] = coll_cfg.scale_out_algorithm

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: DictConfig):

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    

    # ----------------------------------------------------------------------------
    # EXTRACT CONFIG VALUES (before logging)
    # ----------------------------------------------------------------------------

    framework       = cfg.framework
    ccl_backend     = cfg.ccl_backend
    
    # Extract implementations from new config format
    raw_implementations = cfg.order_of_run
    if isinstance(raw_implementations, (list, tuple)) or (hasattr(raw_implementations, '__iter__') and not isinstance(raw_implementations, str)):
        implementations_to_run = list(raw_implementations)
    else:
        implementations_to_run = [raw_implementations]
    
    barrier_enabled = cfg.barrier

    # ----------------------------------------------------------------------------
    # LOGGER INITIALIZATION
    # ----------------------------------------------------------------------------

    if mpi_rank == 0:      
        if "DL_COMM_LOG_DIR" in os.environ:
            log_dir = os.environ["DL_COMM_LOG_DIR"]
        else:
            chicago_tz = pytz.timezone('America/Chicago')
            timestamp = datetime.datetime.now(chicago_tz).strftime("%Y%m%d_%H%M%S_%f")
            log_dir = f"logs/run_{timestamp}"

        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = None
    
    
    log_dir = MPI.COMM_WORLD.bcast(log_dir, root=0)
    log = DLCOMMLogger.get_instance(log_file="dlcomm.log", log_dir=log_dir)
    
    if mpi_rank == 0:
        log.info("-------------------------------------------------------------------------")
        log.info("[CONFIG] Loading schema and validating user YAML")
        

        log.info(f"[DEBUG] Current working directory: {os.getcwd()}")
        log.info(f"[DEBUG] Script location: {os.path.dirname(os.path.abspath(__file__))}")
        log.info(f"[DEBUG] Implementations to run: {implementations_to_run}")
        log.info(f"[DEBUG] Config loaded successfully")

    # ----------------------------------------------------------------------------
    # FRAMEWORK-SPECIFIC IMPORTS (once per execution)
    # ----------------------------------------------------------------------------
    if cfg.framework == "pytorch":
        # timer func defined in ./timer/timer.py
        with timer("import time"):
            import torch.nn.parallel
            import torch.distributed as dist
            
            # Intel-specific imports for CCL backends
            if ccl_backend in ["xccl", "ccl"]:
                import intel_extension_for_pytorch
                import oneccl_bindings_for_pytorch

    # Define barrier function for timing synchronization
    def time_barrier(group=None, device=None):
        if barrier_enabled:
            if group is not None and device is not None:
                dist.barrier(group=group,device_ids=[device.index])
            else:
                MPI.COMM_WORLD.Barrier()
    
    # ----------------------------------------------------------------------------
    # SYSTEM INFORMATION LOGGING (once per execution)
    # ----------------------------------------------------------------------------

    # print_system_info defined in ./config/system_info.py
    print_system_info(log, mpi_rank)
    
    # ----------------------------------------------------------------------------
    # MPI RANK COORDINATION (once per execution) 
    # ----------------------------------------------------------------------------
 
    if mpi_rank == 0:
       
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2254
    else:
        MASTER_ADDR = None
        MASTER_PORT = None
    
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    
    MASTER_PORT = MPI.COMM_WORLD.bcast(MASTER_PORT, root=0)
    
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    # ----------------------------------------------------------------------------
    # TORCH DISTRIBUTED INIT (once per execution)
    # ----------------------------------------------------------------------------
    
    max_mpi_size_needed = validate_mpi_configuration(cfg, mpi_size, mpi_rank, log)
     
    MPI.COMM_WORLD.Barrier()
    with timer("init time"):
        dist.init_process_group(
            backend=ccl_backend,
            init_method='env://',
            world_size=mpi_size,
            rank=mpi_rank,
            timeout=datetime.timedelta(seconds=3600)
        )

    # ----------------------------------------------------------------------------
    # GLOBAL DEVICE ALLOCATION (once per execution)
    # ----------------------------------------------------------------------------
    
    # Assign each rank to a unique device before any mode execution
    if cfg.ccl_backend == "nccl" and torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
        device_id = mpi_rank % available_devices
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
 
    elif cfg.ccl_backend in ["ccl", "xccl"] and torch.xpu.is_available():
        available_devices = torch.xpu.device_count()
        device_id = mpi_rank % available_devices
        device = torch.device(f"xpu:{device_id}")

        if mpi_rank==0:
            log.info(f"available device {available_devices}")
        log.info(f"my rank{mpi_rank}, device id {device_id}, device {device}")
 
    else:
        device = torch.device('cpu')
   
    MPI.COMM_WORLD.Barrier()
    
    # Create validator once for entire run to prevent repeated backend warnings
    config_spec_path = Path(__file__).parent / "config" / "config_spec.json"
    with open(config_spec_path, "r") as f:
        spec = json.load(f)
    
    validator = ConfigValidator(spec)
    
    # Start multi-implementation execution loop
    for impl_index, impl_name in enumerate(implementations_to_run):
        if mpi_rank == 0 and len(implementations_to_run) > 1:
            log.info("")
            log.info("=" * 80)
            log.info(f"[IMPLEMENTATION {impl_index + 1}/{len(implementations_to_run)}] ==================== {impl_name.upper()} ====================")
            log.info("=" * 80)
            log.info("")

        # Get the implementation configuration
        implementation_config = None
        for impl_config in cfg.implementations:
            if hasattr(impl_config, 'name') and impl_config.name == impl_name:
                implementation_config = impl_config
                break
        
        if implementation_config is None:
            if mpi_rank == 0:
                log.error(f"[CONFIG] Implementation '{impl_name}' not found in configuration")
            continue
        
        # Get communication groups for this implementation
        comm_groups = implementation_config.comm_groups
        available_modes = []
        if hasattr(comm_groups, 'within_node'):
            available_modes.append('within_node')
        if hasattr(comm_groups, 'across_node'):
            available_modes.append('across_node')
        if hasattr(comm_groups, 'flatview'):
            available_modes.append('flatview')
        

        
        # Execute each available mode in this implementation
        for mode_index, comm_mode in enumerate(available_modes):
            if mpi_rank == 0:
                log.info("")
                log.info(f"[MODE {mode_index + 1}/{len(available_modes)}] ---------- {comm_mode.upper()} ----------")
                log.info("")

            # Reset timers for each mode (except the first one to preserve setup times)
            if impl_index > 0 or mode_index > 0:
                reset_times()

            # 1) Pick the right config block based on comm_mode from current implementation
            if comm_mode == "flatview":
                coll_cfg = comm_groups.flatview.collective
                mode_cfg = comm_groups.flatview

            elif comm_mode == "within_node":
                coll_cfg = comm_groups.within_node.collective
                mode_cfg = comm_groups.within_node

            elif comm_mode == "across_node":
                coll_cfg = comm_groups.across_node.collective
                mode_cfg = comm_groups.across_node

            else:
                raise ValueError(f"Unknown comm_mode: {comm_mode}")

            # Extract configuration
            coll_name          = coll_cfg.name
            op_name            = coll_cfg.op
            dtype_str          = coll_cfg.payload.dtype
            iters              = coll_cfg.iterations
            enable_correctness = mode_cfg.verify_correctness

            # compute buffer/count
            buffer_in_bytes = parse_buffer_size(coll_cfg.payload.buffer_size)
            torch_dtype, elem_size = DTYPES[dtype_str]
            
            # Calculate group size for buffer adjustment
            if comm_mode == "flatview":
                group_size = mode_cfg.num_gpus_per_node*mode_cfg.num_compute_nodes
            elif comm_mode == "within_node":
                group_size = mode_cfg.num_gpus_per_node
            elif comm_mode == "across_node":
                group_size = mode_cfg.num_compute_nodes
            else:
                raise ValueError (f"Unkown problem occured in group size assignment")
            
            # Adjust buffer size for operations requiring group divisibility
            buffer_in_bytes, adjustment_msg = adjust_buffer_size_for_group_divisibility(buffer_in_bytes, group_size, coll_name, elem_size, log, mpi_rank)
            num_elems = buffer_in_bytes // elem_size

            # lookup collective fn and op
            run_collective = COLLECTIVES[coll_name]
            op_obj         = OP_MAP[op_name] if coll_name in OPS_NEED_REDUCE else None

            
            # ----------------------------------------------------------------------------
            # CONFIG VALIDATION 
            # ----------------------------------------------------------------------------
            
            # ConfigValidator and spec loaded once per implementation above
            config_valid, validation_buffer_bytes = validator.validate(cfg, implementation_config, comm_mode, mpi_rank, log)
            
            if not config_valid:
                if mpi_rank == 0:
                    log.error("[EXIT] Exiting due to configuration validation errors")
                continue
            
            # Validation for MPI and hardware setup
            if not validator.validate_runtime(cfg, mode_cfg, comm_mode, mpi_size, mpi_rank, log):
                if mpi_rank == 0:
                    log.error("[EXIT] Exiting due to runtime validation errors")
                continue
            
            if mpi_rank == 0:
                log.info("")
                log.info("[CONFIG] Setup")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info(f"[CONFIG] Implementation       : {impl_name}")
                log.info(f"[CONFIG] Framework            : {framework}")
                log.info(f"[CONFIG] Backend              : {cfg.ccl_backend}")
                log.info(f"[CONFIG] Use Profiler         : {cfg.get('use_profiler', 'none')}")
                log.info(f"[CONFIG] Barrier Enabled      : {cfg.barrier}")
                log.info(f"[CONFIG] World Size           : {mpi_size}")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info("")
                
                log.info("[CONFIG] Communication Group")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info(f"[CONFIG] Mode                 : {comm_mode}")
                nodes = mode_cfg.num_compute_nodes
                gpus = mode_cfg.num_gpus_per_node
                log.info(f"[CONFIG] Topology             : {nodes} nodes x {gpus} GPUs")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info("")
                    
                log.info("[CONFIG] Communication Group Details")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info(f"[CONFIG] Collective Name      : {coll_name}")
                log.info(f"[CONFIG] Operation            : {op_name if op_obj else 'N/A'}")
                log.info(f"[CONFIG] Scale Up Algorithm   : {coll_cfg.scale_up_algorithm}")
                log.info(f"[CONFIG] Scale Out Algorithm  : {coll_cfg.scale_out_algorithm}")
                log.info(f"[CONFIG] Data Type            : {dtype_str}")
                log.info(f"[CONFIG] Element Count        : {coll_cfg.payload.count}")
                log.info(f"[CONFIG] Buffer Size          : {coll_cfg.payload.buffer_size} ({buffer_in_bytes} bytes)")
                log.info(f"[CONFIG] Iterations           : {iters}")
                log.info(f"[CONFIG] Verify Correctness   : {enable_correctness}")
                log.info("[CONFIG] ------------------------------------------------------")
                if adjustment_msg:
                    log.info(adjustment_msg)
                log.info("")
            

            # ----------------------------------------------------------------------------
            # ENVIRONMENT SETUP
            # ----------------------------------------------------------------------------
            
            
            setup_collective_algorithms(cfg, coll_cfg, comm_mode)

            # ----------------------------------------------------------------------------
            # COMMUNICATION GROUP SETUP
            # ----------------------------------------------------------------------------

            # setup_communication_groups defined in ./comm/comm_setup.py
            # Pass the current mode as force_mode for multi-mode support and pre-allocated device
            comm_info = setup_communication_groups(mode_cfg, mpi_rank, log, dist, force_mode=comm_mode)
            my_within_group = comm_info['my_within_group']
            my_across_group = comm_info['my_across_group'] 
            flat_group = comm_info['flat_group']
            # device is assigned at the begining
            within_group_id = comm_info['within_group_id']
            across_group_id = comm_info['across_group_id']
            ranks_responsible_for_logging = comm_info['ranks_responsible_for_logging']
            
            
            MPI.COMM_WORLD.Barrier()
            
            # Print setup times (import, init) before launching profiling job
            gather_and_print_all_times(log, ranks_responsible_for_logging, barrier_enabled, "[TIMERS - SETUP]", "setup")
            
            if mpi_rank == 0:
                log.output("")
                log.output("[MPI] Launching profiling job")

            # ----------------------------------------------------------------------------
            #  COLLECTIVE OP EXECUTION
            # ----------------------------------------------------------------------------
            

            # Collective execution for all modes
            for i in range(iters):
                x = torch.ones(num_elems, dtype=torch_dtype).to(device, non_blocking=True)
                context = {'mpi_rank': mpi_rank, 'cfg': cfg,'log': log, 'iteration': i}

                if comm_mode == "flatview":
                    if flat_group is not None:
                        time_barrier(group=flat_group, device=device)
                        with timer("(Flatview)"):
                            result = run_collective(x, op_obj, group=flat_group, dist=dist)
                            time_barrier(group=flat_group, device=device)
                        check_collective_correctness(context, x, coll_name, op=op_obj, group=flat_group, result_data=result, group_type="Flatview", group_id="All")

                elif comm_mode == "within_node":
                    if my_within_group is not None:
                        time_barrier(group=my_within_group, device=device)
                        with timer(f"(Within-Group-{within_group_id})"):
                            result = run_collective(x, op_obj, group=my_within_group, dist=dist, log=log)
                            time_barrier(group=my_within_group, device=device)
                        check_collective_correctness(context, x, coll_name, op=op_obj, group=my_within_group, result_data=result, group_type="Within", group_id=within_group_id)
             
                elif comm_mode == "across_node":
                    if my_across_group is not None:
                        time_barrier(group=my_across_group , device=device)
                        with timer(f"(Across-Group-{across_group_id})"):
                            result = run_collective(x, op_obj, group=my_across_group, dist=dist, log=log)
                            time_barrier(group=my_across_group,  device=device)
                        check_collective_correctness(context, x, coll_name, op=op_obj, group=my_across_group, result_data=result, group_type="Across", group_id=across_group_id)
            

            # ----------------------------------------------------------------------------
            #  REPORTING (FOR SINGLE-PHASE MODES ONLY)
            # ----------------------------------------------------------------------------

            # Gather all timer data from responsible ranks and let rank 0 print organized output
            gather_and_print_all_times(log, ranks_responsible_for_logging, barrier_enabled, "[TIMERS]", None, coll_name)
            
            # Gather bandwidth data from responsible ranks and let rank 0 print organized output
            if comm_mode == "flatview":
                adjusted_buffer_sizes_single = {'flatview': buffer_in_bytes}
            elif comm_mode == "within_node":
                adjusted_buffer_sizes_single = {'within': buffer_in_bytes}
            elif comm_mode == "across_node":
                adjusted_buffer_sizes_single = {'across': buffer_in_bytes}
            else:
                adjusted_buffer_sizes_single = None
            print_all_bandwidths(log, cfg, mpi_size, ranks_responsible_for_logging, None, adjusted_buffer_sizes_single, comm_mode, mode_cfg)
            
            # Only rank 0 prints remaining analysis
            if mpi_rank == 0:

                log.info("-------------------------------------------------------------------------")
                log.info("[MPI] Job complete")
                log.info("-------------------------------------------------------------------------")
                
                if cfg.ccl_debug:
                    log.info("Querying Default Table selection")

                    terminal_log_path = os.path.join(log_dir, "terminal_output.log")
                    if os.path.exists(terminal_log_path):
                        # report_ccl_selection func defined in ./analysis/ccl_parser.py
                        report_ccl_selection(terminal_log_path, coll_name, log)
                    else:
                        log.info(f"[SELECTION] Terminal output log not found: {terminal_log_path}")

                log.info("-------------------------------------------------------------------------")
                if len(available_modes) > 1:
                    log.info(f"[EXIT] Mode {mode_index + 1}/{len(available_modes)} ({comm_mode.upper()}) completed.")
                else:
                    log.info("[EXIT] All Done.")
                log.info("-------------------------------------------------------------------------")

            # Inter-mode cleanup (if not the last mode)
            if mode_index < len(available_modes) - 1:

                
                # Clean up process groups for next mode
                if 'my_within_group' in locals() and my_within_group is not None:
                    del my_within_group
                if 'my_across_group' in locals() and my_across_group is not None:
                    del my_across_group
                
                # Barrier to synchronize before next mode
                MPI.COMM_WORLD.Barrier()

    # End of multi-implementation execution loop
    
    if mpi_rank == 0 and len(implementations_to_run) > 1:
        log.info("")
        log.info("=" * 80)
        log.info(f"[FINAL] All {len(implementations_to_run)} implementations completed successfully!")
        log.info("=" * 80)
        log.info("")

    # ----------------------------------------------------------------------------
    #  CLEAN UP
    # ----------------------------------------------------------------------------

    DLCOMMLogger.flush()
    DLCOMMLogger.reset()
    MPI.COMM_WORLD.Barrier()   
    dist.destroy_process_group()
    reset_times()
    
if __name__ == "__main__":
    main()