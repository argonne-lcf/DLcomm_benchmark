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
import datetime
from mpi4py import MPI
from pathlib import Path
from time import perf_counter
from omegaconf import DictConfig, OmegaConf
# dl_comm packages
from dl_comm.comm import setup_communication_groups
from dl_comm.utils.utility import DLCOMMLogger, Profile
from dl_comm.config import ConfigValidator, parse_buffer_size, print_system_info, adjust_buffer_size_for_group_divisibility
from dl_comm.comm import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from dl_comm.timer import timer, print_all_times, gather_and_print_all_times, reset_times
from dl_comm.analysis import report_ccl_selection, print_all_bandwidths 
from dl_comm.analysis.correctness import check_collective_correctness
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
        log.info(f"[DEBUG] Config mode: {cfg.comm_group.mode}")
        if hasattr(cfg.comm_group, 'flatview') and cfg.comm_group.mode == 'flatview':
            log.info(f"[DEBUG] Flatview collective: {cfg.comm_group.flatview.collective.name}")
        log.info(f"[DEBUG] Config loaded successfully")

    # ----------------------------------------------------------------------------
    # EXTRACT CONFIG VALUES
    # ----------------------------------------------------------------------------

    framework       = cfg.framework
    ccl_backend     = cfg.ccl_backend
    comm_mode       = cfg.comm_group.mode
    barrier_enabled = cfg.barrier

    # 1) Pick the right config block (or blocks) based on comm_mode
    if comm_mode == "flatview":
        coll_cfg = cfg.comm_group.flatview.collective

    elif comm_mode == "within_node":
        coll_cfg = cfg.comm_group.within_node.collective

    elif comm_mode == "across_node":
        coll_cfg = cfg.comm_group.across_node.collective

    elif comm_mode == "combined":
        # Unpack both within- and across-node configs
        coll_within_cfg = cfg.comm_group.combined.within_node.collective
        coll_across_cfg = cfg.comm_group.combined.across_node.collective

    else:
        raise ValueError(f"Unknown comm_group.mode: {comm_mode}")

    # 2) Single-phase modes: extract once
    if comm_mode != "combined":
        mode_cfg = getattr(cfg.comm_group, comm_mode)
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
            group_size = mpi_size
        elif comm_mode == "within_node":
            group_size = mode_cfg.num_gpus_per_node
        elif comm_mode == "across_node":
            group_size = mode_cfg.num_compute_nodes
        else:
            group_size = mpi_size
            
        # Adjust buffer size for operations requiring group divisibility
        buffer_in_bytes, adjustment_msg = adjust_buffer_size_for_group_divisibility(buffer_in_bytes, group_size, coll_name, elem_size, log, mpi_rank)
        num_elems = buffer_in_bytes // elem_size

        # lookup collective fn and op
        run_collective = COLLECTIVES[coll_name]
        op_obj         = OP_MAP[op_name] if coll_name in OPS_NEED_REDUCE else None

    # 3) Combined mode: extract for both phases
    else:
        # ─── Within-node phase unpack ───────────────────────────────────────────
        within_mode_cfg = cfg.comm_group.combined.within_node
        coll_name_within          = coll_within_cfg.name
        op_name_within            = coll_within_cfg.op
        dtype_str_within          = coll_within_cfg.payload.dtype
        iters_within              = coll_within_cfg.iterations
        enable_correctness_within = within_mode_cfg.verify_correctness

        buffer_within_bytes = parse_buffer_size(coll_within_cfg.payload.buffer_size)
        torch_dtype_within, elem_size_within = DTYPES[dtype_str_within]
        
        # Calculate within-node group size for buffer adjustment
        within_group_size = within_mode_cfg.num_gpus_per_node
        
        # Adjust buffer size for operations requiring group divisibility
        buffer_within_bytes, adjustment_msg_within = adjust_buffer_size_for_group_divisibility(buffer_within_bytes, within_group_size, coll_name_within, elem_size_within, log, mpi_rank)
        num_elems_within   = buffer_within_bytes // elem_size_within

        run_within = COLLECTIVES[coll_name_within]
        op_within  = OP_MAP[op_name_within] if coll_name_within in OPS_NEED_REDUCE else None

        # ─── Across-node phase unpack ───────────────────────────────────────────
        across_mode_cfg = cfg.comm_group.combined.across_node
        coll_name_across          = coll_across_cfg.name
        op_name_across            = coll_across_cfg.op
        dtype_str_across          = coll_across_cfg.payload.dtype
        iters_across              = coll_across_cfg.iterations
        enable_correctness_across = across_mode_cfg.verify_correctness

        buffer_across_bytes = parse_buffer_size(coll_across_cfg.payload.buffer_size)
        torch_dtype_across, elem_size_across = DTYPES[dtype_str_across]
        
        # Calculate across-node group size for buffer adjustment
        across_group_size = across_mode_cfg.num_compute_nodes
        
        # Adjust buffer size for operations requiring group divisibility
        buffer_across_bytes, adjustment_msg_across = adjust_buffer_size_for_group_divisibility(buffer_across_bytes, across_group_size, coll_name_across, elem_size_across, log, mpi_rank)
        num_elems_across   = buffer_across_bytes // elem_size_across

        run_across = COLLECTIVES[coll_name_across]
        op_across  = OP_MAP[op_name_across] if coll_name_across in OPS_NEED_REDUCE else None
    
    # ----------------------------------------------------------------------------
    # CONFIG VALIDATION 
    # ----------------------------------------------------------------------------
    
    config_spec_path = Path(__file__).parent / "config" / "config_spec.json"
    with open(config_spec_path, "r") as f:
        spec = json.load(f)
    
    # ConfigValidator and parse_buffer_size funcs defined in ./config/validation.py
    validator = ConfigValidator(spec)
    config_valid, validation_buffer_bytes = validator.validate(cfg, mpi_rank, log)
    
    if not config_valid:
        if mpi_rank == 0:
            log.error("[EXIT] Exiting due to configuration validation errors")
        return
    
    # Validation for MPI and hardware setup
    if not validator.validate_runtime(cfg, mpi_size, mpi_rank, log):
        if mpi_rank == 0:
            log.error("[EXIT] Exiting due to runtime validation errors")
        return
    
    
    
    # ----------------------------------------------------------------------------
    # FRAMEWORK-SPECIFIC IMPORTS
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
    def time_barrier(group=None):
        if barrier_enabled:
            if group is not None:
                dist.barrier(group=group)
            else:
                dist.barrier()
    
    # ----------------------------------------------------------------------------
    # SYSTEM INFORMATION LOGGING
    # ----------------------------------------------------------------------------

    # print_system_info defined in ./config/system_info.py
    print_system_info(log, mpi_rank)
    
    if mpi_rank == 0:
        log.info("")
        log.info("[CONFIG] Setup")
        log.info("[CONFIG] ------------------------------------------------------")
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
        if comm_mode == "combined":
            within_nodes = cfg.comm_group.combined.within_node.num_compute_nodes
            within_gpus = cfg.comm_group.combined.within_node.num_gpus_per_node
            across_nodes = cfg.comm_group.combined.across_node.num_compute_nodes  
            across_gpus = cfg.comm_group.combined.across_node.num_gpus_per_node
            log.info(f"[CONFIG] Within               : {within_nodes} nodes x {within_gpus} GPUs")
            log.info(f"[CONFIG] Across               : {across_nodes} nodes x {across_gpus} GPUs")
            log.info("[CONFIG] ------------------------------------------------------")
            log.info("")
        else:
            mode_cfg = getattr(cfg.comm_group, comm_mode)
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
    # MPI RANK COORDINATION
    # ----------------------------------------------------------------------------
 
    if mpi_rank == 0:
        import socket
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2219
    else:
        MASTER_ADDR = None
        MASTER_PORT = None
    
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    MASTER_PORT = MPI.COMM_WORLD.bcast(MASTER_PORT, root=0)
    
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    # ----------------------------------------------------------------------------
    # TORCH DISTRIBUTED INIT
    # ----------------------------------------------------------------------------
     
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
    # ENVIRONMENT SETUP
    # ----------------------------------------------------------------------------
    
    
    if comm_mode != "combined":
        setup_collective_algorithms(cfg, coll_cfg, comm_mode)

    # ----------------------------------------------------------------------------
    # COMMUNICATION GROUP SETUP
    # ----------------------------------------------------------------------------

    # setup_communication_groups defined in ./comm/comm_setup.py
    comm_info = setup_communication_groups(cfg, mpi_rank, log, dist)
    my_within_group = comm_info['my_within_group']
    my_across_group = comm_info['my_across_group'] 
    world_group = comm_info['world_group']
    device = comm_info['device']
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

 

    # Single-phase (flatview / within_node / across_node)
    if comm_mode != "combined":
        for i in range(iters):
           
            x = torch.ones(num_elems, dtype=torch_dtype).to(device, non_blocking=True)
            context = {'mpi_rank': mpi_rank, 'cfg': cfg,'log': log, 'iteration': i}

            if comm_mode == "flatview":
                time_barrier()
                with timer("(Flatview)"):
                    result = run_collective(x, op_obj, group=world_group, dist=dist)
                    time_barrier()
                check_collective_correctness(context, x, coll_name, op=op_obj, group=world_group, result_data=result, group_type="Flatview", group_id="All")

            elif comm_mode == "within_node":
                time_barrier()
                with timer(f"(Within-Group-{within_group_id})"):
                    result = run_collective(x, op_obj, group=my_within_group, dist=dist, log=log)
                    time_barrier()
                check_collective_correctness(context, x, coll_name, op=op_obj, group=my_within_group, result_data=result, group_type="Within", group_id=within_group_id)

            elif comm_mode == "across_node":
                time_barrier()
                with timer(f"(Across-Group-{across_group_id})"):
                    result = run_collective(x, op_obj, group=my_across_group, dist=dist, log=log)
                    time_barrier()
                check_collective_correctness(context, x, coll_name, op=op_obj, group=my_across_group, result_data=result, group_type="Across", group_id=across_group_id)
    
    else:
        # ═══════════════════════════════════════════════════════════════════════════
        
        if mpi_rank == 0:
            log.info("")
            log.info("[CONFIG] ═══════════════════════════════════════════════════════")
            log.info("[CONFIG] WITHIN-NODE PHASE")
            log.info(f"[CONFIG] Collective Name      : {coll_name_within}")
            log.info(f"[CONFIG] Operation            : {op_name_within if op_within else 'N/A'}")
            log.info(f"[CONFIG] Scale Up Algorithm   : {coll_within_cfg.scale_up_algorithm}")
            log.info(f"[CONFIG] Scale Out Algorithm  : {coll_within_cfg.scale_out_algorithm}")
            log.info(f"[CONFIG] Data Type            : {dtype_str_within}")
            log.info(f"[CONFIG] Element Count        : {coll_within_cfg.payload.count}")
            log.info(f"[CONFIG] Buffer Size          : {coll_within_cfg.payload.buffer_size} ({buffer_within_bytes} bytes)")
            log.info(f"[CONFIG] Iterations           : {iters_within}")
            log.info(f"[CONFIG] Verify Correctness   : {enable_correctness_within}")
            log.info("[CONFIG] ═══════════════════════════════════════════════════════")
            if adjustment_msg_within:
                log.info(adjustment_msg_within)
            log.info("")

        # ─── Within-node phase iterations ───────────────────────────
        setup_collective_algorithms(cfg, coll_within_cfg, "within_node")
 
        for i in range(iters_within):
            context = {'mpi_rank': mpi_rank, 'cfg': cfg,'log': log, 'iteration': i}
            x = torch.ones(num_elems_within, dtype=torch_dtype_within).to(device, non_blocking=True)
            
            time_barrier()
            with timer(f"(Within-Group-{within_group_id})"):
                result = run_within(x, op_within, group=my_within_group, dist=dist, log=log)
                time_barrier()
            check_collective_correctness(context, x, coll_name_within, op=op_within, group=my_within_group, result_data=result, group_type="Within", group_id=within_group_id)

        # ─── Within-node phase reporting ───────────────────────────
        gather_and_print_all_times(log, ranks_responsible_for_logging, barrier_enabled, "[TIMERS]", "within", coll_name_within)
        adjusted_buffer_sizes_within = {'within': buffer_within_bytes}
        print_all_bandwidths(log, cfg, mpi_size, ranks_responsible_for_logging, "within", adjusted_buffer_sizes_within)

 

        # ═══════════════════════════════════════════════════════════════════════════

        if mpi_rank == 0:
            log.info("")
            log.info("[CONFIG] ═══════════════════════════════════════════════════════")
            log.info("[CONFIG] ACROSS-NODE PHASE")
            log.info(f"[CONFIG] Collective Name      : {coll_name_across}")
            log.info(f"[CONFIG] Operation            : {op_name_across if op_across else 'N/A'}")
            log.info(f"[CONFIG] Scale Up Algorithm   : {coll_across_cfg.scale_up_algorithm}")
            log.info(f"[CONFIG] Scale Out Algorithm  : {coll_across_cfg.scale_out_algorithm}")
            log.info(f"[CONFIG] Data Type            : {dtype_str_across}")
            log.info(f"[CONFIG] Element Count        : {coll_across_cfg.payload.count}")
            log.info(f"[CONFIG] Buffer Size          : {coll_across_cfg.payload.buffer_size} ({buffer_across_bytes} bytes)")
            log.info(f"[CONFIG] Iterations           : {iters_across}")
            log.info(f"[CONFIG] Verify Correctness   : {enable_correctness_across}")
            log.info("[CONFIG] ═══════════════════════════════════════════════════════")
            if adjustment_msg_across:
                log.info(adjustment_msg_across)
            log.info("")


        # ─── Across-node phase iterations ───────────────────────────
        setup_collective_algorithms(cfg, coll_across_cfg, "across_node")
 
        for i in range(iters_across):
            context = {'mpi_rank': mpi_rank, 'cfg': cfg,'log': log, 'iteration': i}
            x = torch.ones(num_elems_across, dtype=torch_dtype_across).to(device, non_blocking=True)
            
            if my_across_group:
                time_barrier(group=my_across_group)
                with timer(f"(Across-Group-{across_group_id})"):
                    result = run_across(x, op_across, group=my_across_group, dist=dist, log=log)
                    time_barrier(group=my_across_group)
                check_collective_correctness(context, x, coll_name_across, op=op_across, group=my_across_group, result_data=result, group_type="Across", group_id=across_group_id)
            else:
                result = None

        # ─── Across-node phase reporting ───────────────────────────
        gather_and_print_all_times(log, ranks_responsible_for_logging, barrier_enabled, "[TIMERS]", "across", coll_name_across)
        adjusted_buffer_sizes_across = {'across': buffer_across_bytes}
        print_all_bandwidths(log, cfg, mpi_size, ranks_responsible_for_logging, "across", adjusted_buffer_sizes_across)

        time_barrier()

    # ----------------------------------------------------------------------------
    #  REPORTING (FOR SINGLE-PHASE MODES ONLY)
    # ----------------------------------------------------------------------------
  
    # Only single-phase modes need final reporting (combined mode already reported)
    if comm_mode != "combined":
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
        print_all_bandwidths(log, cfg, mpi_size, ranks_responsible_for_logging, None, adjusted_buffer_sizes_single)
    
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
                if comm_mode != "combined":
                    report_ccl_selection(terminal_log_path, coll_name, log)
                else:
                    report_ccl_selection(terminal_log_path, coll_name_within, log)
                    log.info("")
                    report_ccl_selection(terminal_log_path, coll_name_across, log)
            else:
                log.info(f"[SELECTION] Terminal output log not found: {terminal_log_path}")

        log.info("-------------------------------------------------------------------------")
        log.info("[EXIT] All Done.")
        log.info("-------------------------------------------------------------------------")

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