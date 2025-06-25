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
from pathlib import Path
from time import perf_counter
from omegaconf import DictConfig, OmegaConf
from mpi4py import MPI
# dl_comm packages
from dl_comm.utils.utility import DLCOMMLogger, Profile
from dl_comm.comm import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from dl_comm.timer import timer, print_all_times, gather_and_print_all_times
from dl_comm.analysis import report_ccl_selection, print_all_bandwidths, check_group_correctness
from dl_comm.comm import setup_communication_groups
from dl_comm.config import ConfigValidator, parse_buffer_size
 
# ----------------------------------------------------------------------------
# SETUP FUNCTIONS
# ----------------------------------------------------------------------------

def setup_environment_with_collective(cfg: DictConfig, coll_cfg):
    # CCL environment variables
    os.environ["CCL_ATL_TRANSPORT"] = "mpi"
    os.environ["CCL_ATL_SHM"] = "0"
    if cfg.ccl_debug:
        os.environ["CCL_LOG_LEVEL"] = "debug"
    os.environ["CCL_PROCESS_LAUNCHER"] = "pmix"
    os.environ["TORCH_CPP_LOG_LEVEL"] = "error"
    os.environ["FI_MR_CACHE_MONITOR"] = "userfaultfd"

    scale_up_override = f"CCL_{coll_cfg.name.upper()}"
    os.environ[scale_up_override] = coll_cfg.scale_up_algorithm

    scale_out_override = f"CCL_{coll_cfg.name.upper()}_SCALEOUT"
    os.environ[scale_out_override] = coll_cfg.scale_out_algorithm

# ----------------------------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------------------------

@hydra.main(config_path="config", config_name="config", version_base=None)
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
    
    if mpi_rank == 0:
        log.info("-------------------------------------------------------------------------")
        log.info("[CONFIG] Loading schema and validating user YAML")

    # ----------------------------------------------------------------------------
    # EXTRACT CONFIG VALUES
    # ----------------------------------------------------------------------------

    framework       = cfg.framework
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
        num_elems_across   = buffer_across_bytes // elem_size_across

        run_across = COLLECTIVES[coll_name_across]
        op_across  = OP_MAP[op_name_across] if coll_name_across in OPS_NEED_REDUCE else None
    
    # ----------------------------------------------------------------------------
    # CONFIG VALIDATION & ENVIRONMENT SETUP
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
    
    # setup_environment func defined in current file
    if comm_mode != "combined":
        setup_environment_with_collective(cfg, coll_cfg)
    else:
        # For combined mode, set up environment with within-node config first
        setup_environment_with_collective(cfg, coll_within_cfg)
    
    # ----------------------------------------------------------------------------
    # FRAMEWORK-SPECIFIC IMPORTS
    # ----------------------------------------------------------------------------

    if cfg.framework == "pytorch" and cfg.ccl_backend == "xccl":
        # timer func defined in ./timer/timer.py
        with timer("import time"):
            import intel_extension_for_pytorch
            import oneccl_bindings_for_pytorch
            import torch.nn.parallel
            import torch.distributed as dist

    # Define barrier function for timing synchronization
    def time_barrier():
        if barrier_enabled:
            MPI.COMM_WORLD.Barrier()
    
    if mpi_rank == 0:
        log.info("[CONFIG] Final validated settings\n")
        log.info(f"  • framework           = {framework}")
        log.info(f"  • backend             = {cfg.ccl_backend}")
        log.info(f"  • comm_mode           = {comm_mode}")
        log.info(f"  • use_profiler        = {cfg.get('use_profiler', 'none')}")
        
        if comm_mode != "combined":
            log.info(f"  • collective_name     = {coll_name}")
            log.info(f"  • op                  = {op_name}")
            log.info(f"  • scale_up_algo       = {coll_cfg.scale_up_algorithm}")
            log.info(f"  • scale_out_algo      = {coll_cfg.scale_out_algorithm}")
            log.info(f"  • buffer_size         = {coll_cfg.payload.buffer_size} ({buffer_in_bytes} bytes)")
            log.info(f"  • dtype               = {dtype_str}")
            log.info(f"  • count               = {coll_cfg.payload.count}")
        else:
            log.info(f"  • within_collective   = {coll_name_within}")
            log.info(f"  • within_op           = {op_name_within}")
            log.info(f"  • within_scale_up     = {coll_within_cfg.scale_up_algorithm}")
            log.info(f"  • within_buffer_size  = {coll_within_cfg.payload.buffer_size} ({buffer_within_bytes} bytes)")
            log.info(f"  • within_dtype        = {dtype_str_within}")
            log.info(f"  • across_collective   = {coll_name_across}")
            log.info(f"  • across_op           = {op_name_across}")
            log.info(f"  • across_scale_up     = {coll_across_cfg.scale_up_algorithm}")
            log.info(f"  • across_buffer_size  = {coll_across_cfg.payload.buffer_size} ({buffer_across_bytes} bytes)")
            log.info(f"  • across_dtype        = {dtype_str_across}")
        
        log.info("-------------------------------------------------------------------------")
   
    # ----------------------------------------------------------------------------
    # MPI RANK COORDINATION
    # ----------------------------------------------------------------------------
 
    if mpi_rank == 0:
        import socket
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2257
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
            backend="ccl",
            init_method='env://',
            world_size=mpi_size,
            rank=mpi_rank,
            timeout=datetime.timedelta(seconds=3600)
        )

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
    
    if mpi_rank == 0:
        log.info("")
        log.info("[MPI][SETUP] ------------------------------------------------------")
        log.info(f"[MPI][SETUP] Framework      : {framework}")
        log.info(f"[MPI][SETUP] World Size     : {mpi_size}")
        
        if comm_mode != "combined":
            log.info(f"[MPI][SETUP] Collective     : {coll_name}")
            log.info(f"[MPI][SETUP] Operation      : {op_name if op_obj else 'N/A'}")
            log.info(f"[MPI][SETUP] DType          : {dtype_str}")
            log.info(f"[MPI][SETUP] Count          : {coll_cfg.payload.count}")
            log.info(f"[MPI][SETUP] Buffer Size    : {buffer_in_bytes}")
            log.info(f"[MPI][SETUP] Iterations     : {iters}")
        else:
            log.info(f"[MPI][SETUP] Within Collective : {coll_name_within}")
            log.info(f"[MPI][SETUP] Within Operation  : {op_name_within if op_within else 'N/A'}")
            log.info(f"[MPI][SETUP] Within DType      : {dtype_str_within}")
            log.info(f"[MPI][SETUP] Within Buffer Size: {buffer_within_bytes}")
            log.info(f"[MPI][SETUP] Across Collective : {coll_name_across}")
            log.info(f"[MPI][SETUP] Across Operation  : {op_name_across if op_across else 'N/A'}")
            log.info(f"[MPI][SETUP] Across DType      : {dtype_str_across}")
            log.info(f"[MPI][SETUP] Across Buffer Size: {buffer_across_bytes}")
            log.info(f"[MPI][SETUP] Iterations        : {iters_within}")
        
        log.info("[MPI][SETUP] ------------------------------------------------------")
        log.info("")
        log.info("[MPI] Launching profiling job")

    # ----------------------------------------------------------------------------
    #  COLLECTIVE OP EXECUTION
    # ----------------------------------------------------------------------------

 

    # Single-phase (flatview / within_node / across_node)
    if comm_mode != "combined":
        for i in range(iters):
            # fresh tensor for this iteration
            x = torch.ones(num_elems, dtype=torch_dtype) \
                    .to(device, non_blocking=True)
            context = {'mpi_rank': mpi_rank, 'cfg': cfg,
                    'log': log, 'iteration': i}

            if comm_mode == "flatview":
                check_group_correctness(context, x, "flatview", "before")
                time_barrier()
                with timer("(Flatview)"):
                    run_collective(x, op_obj, group=world_group)
                    time_barrier()
                check_group_correctness(context, x, "flatview", "after")

            elif comm_mode == "within_node":
                check_group_correctness(context, x, "within", "before")
                time_barrier()
                with timer(f"(Within-Group-{within_group_id})"):
                    run_collective(x, op_obj, group=my_within_group)
                    time_barrier()
                check_group_correctness(context, x, "within", "after")

            else:  # "across_node"
                check_group_correctness(context, x, "across", "before")
                time_barrier()
                with timer(f"(Across-Group-{across_group_id})"):
                    run_collective(x, op_obj, group=my_across_group)
                    time_barrier()
                check_group_correctness(context, x, "across", "after")


    # Combined (within → across in each iteration)
    else:
        for i in range(iters_within):
            context = {'mpi_rank': mpi_rank, 'cfg': cfg,
                    'log': log, 'iteration': i}

            time_barrier()
            with timer("Total (Within→Across)"):
                # ─── Within-node phase ───────────────────────────
                x = torch.ones(num_elems_within,
                            dtype=torch_dtype_within) \
                        .to(device, non_blocking=True)
                check_group_correctness(context, x, "within", "before")
                time_barrier()
                with timer(f"(Within-Group-{within_group_id})"):
                    run_within(x, op_within, group=my_within_group)
                    time_barrier()
                check_group_correctness(context, x, "within", "after")

                # ─── Across-node phase ───────────────────────────
                x = torch.ones(num_elems_across,
                            dtype=torch_dtype_across) \
                        .to(device, non_blocking=True)
                check_group_correctness(context, x, "across", "before")
                if my_across_group:
                    time_barrier()
                    with timer(f"(Across-Group-{across_group_id})"):
                        run_across(x, op_across, group=my_across_group)
                        time_barrier()
                check_group_correctness(context, x, "across", "after")

            time_barrier()

    # ----------------------------------------------------------------------------
    #  REPORTING
    # ----------------------------------------------------------------------------
  
    # Gather all timer data from responsible ranks and let rank 0 print organized output
    gather_and_print_all_times(log, ranks_responsible_for_logging, barrier_enabled)
    
    # Only rank 0 prints bandwidth analysis
    if mpi_rank == 0:
        # print_all_bandwidths func defined in ./analysis/bandwidth.py
        if comm_mode != "combined":
            print_all_bandwidths(log, buffer_in_bytes, coll_name)
        else:
            # For combined mode, report both buffer sizes
            log.info("[BANDWIDTH] Combined mode - reporting within-node bandwidth:")
            print_all_bandwidths(log, buffer_within_bytes, coll_name_within)
            log.info("[BANDWIDTH] Combined mode - reporting across-node bandwidth:")
            print_all_bandwidths(log, buffer_across_bytes, coll_name_across)

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
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()