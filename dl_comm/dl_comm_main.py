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
import hydra
import socket
import datetime
from mpi4py import MPI
from pathlib import Path
from time import perf_counter
from omegaconf import DictConfig, OmegaConf
# dl_comm packages
from dl_comm.comm import setup_communication_groups
from dl_comm.utils.utility import DLCOMMLogger, Profile, dummy_mxm_compute
from dl_comm.analysis.correctness import check_collective_correctness
from dl_comm.comm import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES
from dl_comm.comm.collectives import init_framework_constants
from dl_comm.analysis import report_ccl_selection, report_nccl_selection, gather_and_print_all_bandwidths 
from dl_comm.timer import timer, print_all_times, gather_and_print_all_times, reset_times
from dl_comm.config import ConfigValidator, parse_buffer_size, validate_and_calculate_buffer_size, print_system_info
from dl_comm.config import adjust_buffer_size_for_group_divisibility, validate_mpi_configuration
from dl_comm.config import setup_algorithm_overrides, setup_collective_algorithms_ccl

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

    framework       = cfg.framework.lower()
    ccl_backend     = cfg.ccl_backend
    device_type     = cfg.device_type.lower()
    memory_source   = cfg.memory_source.lower()
    
    # Initialize framework-specific constants
    init_framework_constants(framework)
    
    # Extract task names from order_of_run
    raw_tasks = cfg.order_of_run
    if isinstance(raw_tasks, (list, tuple)) or (hasattr(raw_tasks, '__iter__') and not isinstance(raw_tasks, str)):
        tasks_to_run = list(raw_tasks)
    else:
        tasks_to_run = [raw_tasks]
    
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
        log.info(f"[DEBUG] Tasks to run: {tasks_to_run}")
        log.info(f"[DEBUG] Config loaded successfully")

    # ----------------------------------------------------------------------------
    # FRAMEWORK-SPECIFIC IMPORTS (once per execution)
    # ----------------------------------------------------------------------------
    if framework == "pytorch":
        # timer func defined in ./timer/timer.py
        with timer("import time"):
            import torch
            import torch.nn.parallel
            import torch.distributed as dist
            
            # Intel-specific imports for CCL backends
            if ccl_backend in ["xccl", "ccl"]:
                import intel_extension_for_pytorch
                import oneccl_bindings_for_pytorch
    elif framework == "jax":
        with timer("import time"):
            import jax
            import jax.numpy as jnp
            import jax.distributed as jdist

    # Define barrier function for timing synchronization
    def time_barrier(group=None, device=None):
        if barrier_enabled:
            if framework == "pytorch":
                if device_type == 'cpu':
                    if group is not None:
                        dist.barrier(group=group)
                elif device_type == 'gpu':
                    if group is not None and device is not None:
                        dist.barrier(group=group, device_ids=[device.index])
            elif framework == "jax":
                if device_type == 'cpu':
                    if group is not None:
                        jdist.barrier(group=group)
                elif device_type == 'gpu':
                    if group is not None:
                        jdist.barrier(group=group)
          
    # ----------------------------------------------------------------------------
    # SYSTEM INFORMATION LOGGING (once per execution)
    # ----------------------------------------------------------------------------

    # print_system_info defined in ./config/system_info.py
    print_system_info(log, mpi_rank, framework)
    
    # ----------------------------------------------------------------------------
    # MPI RANK COORDINATION (once per execution) 
    # ----------------------------------------------------------------------------
 
    if mpi_rank == 0:
       
        MASTER_ADDR = socket.gethostname()
        MASTER_PORT = 2268
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
    
    max_mpi_size_needed, mpi_validation_errors = validate_mpi_configuration(cfg, mpi_size, mpi_rank, log)
    
    # ----------------------------------------------------------------------------
    # ALGORITHM SETUP (before distributed init)
    # ----------------------------------------------------------------------------
    
    setup_algorithm_overrides(cfg, log)
     
    MPI.COMM_WORLD.Barrier()
    with timer("init time"):
        if framework == "pytorch":
            dist.init_process_group(
                backend=ccl_backend,
                init_method='env://',
                world_size=mpi_size,
                rank=mpi_rank,
                timeout=datetime.timedelta(seconds=3600)
            )
        elif framework == "jax":
            jdist.initialize(coordinator_address="env://", num_processes=mpi_size, process_id=mpi_rank)


    # ----------------------------------------------------------------------------
    # DEVICE ALLOCATION - Moved inside implementation loop for sequential assignment
    # ----------------------------------------------------------------------------
    
    # Create validator once for entire run to prevent repeated backend warnings
    config_spec_path = Path(__file__).parent / "config" / "config_spec.json"
    with open(config_spec_path, "r") as f:
        spec = json.load(f)
    
    validator = ConfigValidator(spec)
    
    # Validate task names are unique (tasks_to_run should have unique names)
    if len(tasks_to_run) != len(set(tasks_to_run)):
        if mpi_rank == 0:
            log.error("[EXIT] Exiting due to duplicate task names in order_of_run")
        sys.exit(1)
    
    # Start multi-task execution loop
    for task_index, task_name in enumerate(tasks_to_run):
        if mpi_rank == 0 and len(tasks_to_run) > 1:
            log.info("")
            log.info("=" * 80)
            log.info(f"[TASK {task_index + 1}/{len(tasks_to_run)}] ==================== {task_name.upper()} ====================")
            log.info("=" * 80)
            log.info("")

        # Get the task configuration
        if not hasattr(cfg, task_name):
            if mpi_rank == 0:
                log.error(f"[CONFIG] Task '{task_name}' not found in configuration")
            continue
        
        task_config = getattr(cfg, task_name)
        
        # Get communication mode for this task
        comm_mode = task_config.comm_group
        available_modes = [comm_mode]  # Single mode per task now
        

        
        # Execute the single mode for this task
        for mode_index, current_mode in enumerate(available_modes):
            if mpi_rank == 0 and len(available_modes) > 1:
                log.info("")
                log.info(f"[MODE {mode_index + 1}/{len(available_modes)}] ---------- {current_mode.upper()} ----------")
                log.info("")

            # Reset timers for each task (except the first one to preserve setup times)
            if task_index > 0 or mode_index > 0:
                reset_times()

            # Get collective and mode configuration directly from task
            coll_cfg = task_config.collective
            mode_cfg = task_config
            comm_mode = current_mode  # Use the current mode from the loop

            # Check if we have enough ranks for this mode
            if comm_mode == "flatview":
                required_ranks = mode_cfg.num_devices_per_node * mode_cfg.num_compute_nodes
            elif comm_mode == "within_node":
                required_ranks = mode_cfg.num_devices_per_node * mode_cfg.num_compute_nodes
            elif comm_mode == "across_node":
                required_ranks = mode_cfg.num_devices_per_node * mode_cfg.num_compute_nodes
            
            if mpi_size < required_ranks:
                if mpi_rank == 0:
                    log.warning(f"[SKIP] {task_name}_{comm_mode} requires {required_ranks} ranks but only {mpi_size} available - skipping")
                continue

            # Extract configuration
            coll_name          = coll_cfg.collective_name
            op_name            = coll_cfg.collective_op
            dtype_str          = coll_cfg.payload.dtype
            iters              = coll_cfg.iterations
            warmup_iters       = getattr(coll_cfg, 'warmup_iterations', 0)  # Default to 0 if not specified
            add_mxm_compute    = getattr(coll_cfg, 'add_mxm_compute', False)  # Default to False if not specified
            enable_correctness = mode_cfg.verify_correctness

            # Validate operation is provided for collectives that need it
            if coll_name in OPS_NEED_REDUCE:
                if not op_name or (isinstance(op_name, str) and op_name.strip() == ''):
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] {task_name}_{comm_mode}: Collective '{coll_name}' requires an operation (op). Valid operations: {list(OP_MAP.keys())}")
                    continue  # Skip this task
                elif op_name not in OP_MAP:
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] {task_name}_{comm_mode}: Invalid operation '{op_name}' for collective '{coll_name}'. Valid operations: {list(OP_MAP.keys())}")
                    continue  # Skip this task

            # compute buffer/count using new validation function
            buffer_in_bytes, num_elems, buffer_errors = validate_and_calculate_buffer_size(coll_cfg.payload, f"{task_name}_{comm_mode}", log, mpi_rank)
            if buffer_errors:
                continue  # Skip this task due to validation errors
             
            _dtype, elem_size = DTYPES[dtype_str]
 
            # Calculate group size for buffer adjustment
            if comm_mode == "flatview":
                group_size = mode_cfg.num_devices_per_node*mode_cfg.num_compute_nodes
            elif comm_mode == "within_node":
                group_size = mode_cfg.num_devices_per_node
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
            
            # ConfigValidator and spec loaded once per task above
            config_valid, validation_buffer_bytes = validator.validate(cfg, task_config, comm_mode, mpi_rank, log)
            
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
                log.info(f"[CONFIG] Task Name            : {task_name}")
                log.info(f"[CONFIG] Framework            : {framework}")
                log.info(f"[CONFIG] Backend              : {cfg.ccl_backend}")
                log.info(f"[CONFIG] Extended Logging     : {cfg.extended_logging}")
                log.info(f"[CONFIG] Barrier Enabled      : {cfg.barrier}")
                log.info(f"[CONFIG] World Size           : {mpi_size}")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info("")
                
                log.info("[CONFIG] Communication Group")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info(f"[CONFIG] Mode                 : {comm_mode}")
                nodes = mode_cfg.num_compute_nodes
                devices = mode_cfg.num_devices_per_node
                log.info(f"[CONFIG] Topology             : {nodes} nodes x {devices} devices")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info("")
                    
                log.info("[CONFIG] Communication Group Details")
                log.info("[CONFIG] ------------------------------------------------------")
                log.info(f"[CONFIG] Collective Name      : {coll_name}")
                log.info(f"[CONFIG] Operation            : {op_name if op_obj else 'N/A'}")
                log.info(f"[CONFIG] Scale Up Algorithm   : {coll_cfg.scale_up_algorithm}")
                log.info(f"[CONFIG] Scale Out Algorithm  : {coll_cfg.scale_out_algorithm}")
                log.info(f"[CONFIG] Data Type            : {dtype_str}")
                log.info(f"[CONFIG] Element Count        : {num_elems}")
                # Show original config and final calculated values
                if hasattr(coll_cfg.payload, 'buffer_size') and coll_cfg.payload.buffer_size:
                    log.info(f"[CONFIG] Buffer Size          : {coll_cfg.payload.buffer_size} ({buffer_in_bytes} bytes)")
                elif hasattr(coll_cfg.payload, 'count') and coll_cfg.payload.count:
                    log.info(f"[CONFIG] Count                : {coll_cfg.payload.count} elements ({buffer_in_bytes} bytes)")
                log.info(f"[CONFIG] Iterations           : {iters}")
                log.info(f"[CONFIG] Verify Correctness   : {enable_correctness}")
                log.info("[CONFIG] ------------------------------------------------------")
                if adjustment_msg:
                    log.info(adjustment_msg)
                log.info("")
            

            # ----------------------------------------------------------------------------
            # ENVIRONMENT SETUP
            # ----------------------------------------------------------------------------
            
            # All algorithms already set globally at startup

            # ----------------------------------------------------------------------------
            # COMMUNICATION GROUP SETUP
            # ----------------------------------------------------------------------------

            # setup_communication_groups defined in ./comm/comm_setup.py
            # Pass the current mode as force_mode for multi-mode support and pre-allocated device
            comm_info = setup_communication_groups(mode_cfg, mpi_rank, log, dist, force_mode=comm_mode, full_cfg=cfg)
            my_within_group = comm_info['my_within_group']
            my_across_group = comm_info['my_across_group'] 
            flat_group = comm_info['flat_group']
            device = comm_info['device']  # Device assigned based on group membership
    
            within_group_id = comm_info['within_group_id']
            across_group_id = comm_info['across_group_id']
            ranks_responsible_for_logging = comm_info['ranks_responsible_for_logging']
            
            
            # ----------------------------------------------------------------------------
            #  HOST TO DEVICE TRANSFER TEST
            # ----------------------------------------------------------------------------
            
            if framework == "pytorch":
                if memory_source == "host" and device_type == "gpu":
                    x_test = torch.ones(num_elems, dtype=_dtype, device="cpu")
                    with timer("Host to Device Transfer Time"):
                        x_test = x_test.to(device, non_blocking=True)
                else:
                    x_test = torch.ones(num_elems, dtype=_dtype).to(device, non_blocking=True)


            elif framework== "jax":
                import numpy as np
                if memory_source == "host" and device_type == "gpu":
                    x_host = np.ones(num_elems, dtype=_dtype)   
                    with timer("Host to Device Transfer Time"):
                        x_test = jax.device_put(x_host, device)
                else:
                    x_test = jax.device_put(jnp.ones(num_elems, dtype=_dtype), device)

            MPI.COMM_WORLD.Barrier()
            # ----------------------------------------------------------------------------
            #  MxM COMPUTE SECTION 
            # ----------------------------------------------------------------------------
            if add_mxm_compute:
                mxm_size=1024
                with timer(f"MxM Compute Time, m={mxm_size}"):
                    dummy_mxm_compute(device, _dtype, size=mxm_size, framework=framework) # defined in ./utils/utility.py)
                if mpi_rank == 0:
                    log.output("")
                    log.output(f"[MxM COMPUTE] Matrix multiplication compute completed.")
                    log.output("")
            MPI.COMM_WORLD.Barrier()



            # Print setup times (import, init, host to device) before launching profiling job
            gather_and_print_all_times(log, ranks_responsible_for_logging, barrier_enabled, "[TIMERS - SETUP]", "setup")
            
            if mpi_rank == 0:
                log.output("")
                log.output("[MPI] Launching profiling job")
            # ----------------------------------------------------------------------------
            #  WARMUP ITERATIONS
            # ----------------------------------------------------------------------------
            
            if warmup_iters > 0:
                if mpi_rank == 0:
                    log.info("")
                    log.info(f"  [WARMUP] Running {warmup_iters} warmup iterations...")
                
                for i in range(warmup_iters):
                    if framework == "pytorch":
                        x = torch.ones(num_elems, dtype=_dtype).to(device, non_blocking=True)
                    elif framework == "jax":
                        x = jax.device_put(jnp.ones(num_elems, dtype=_dtype), device)
                    

                    
                    if comm_mode == "flatview":
                        if flat_group is not None:
                            result = run_collective(x, op_obj, group=flat_group, dist=dist, framework=framework)
                    
                    elif comm_mode == "within_node":
                        if my_within_group is not None:
                            result = run_collective(x, op_obj, group=my_within_group, dist=dist, log=log, framework=framework)
                    
                    elif comm_mode == "across_node":
                        if my_across_group is not None:
                            result = run_collective(x, op_obj, group=my_across_group, dist=dist, log=log, framework=framework)
                
                MPI.COMM_WORLD.Barrier()
                if mpi_rank == 0:
                    log.info(f"  [WARMUP] Warmup completed, starting timed iterations...")
                    log.info("")
            

 
    
            # ----------------------------------------------------------------------------
            #  COLLECTIVE OP EXECUTION (TIMED)
            # ----------------------------------------------------------------------------
            

            # Collective execution for all modes
            for i in range(iters):
                if framework == "pytorch":
                    x = torch.ones(num_elems, dtype=_dtype).to(device, non_blocking=True)
                elif framework == "jax":
                    x = jax.device_put(jnp.ones(num_elems, dtype=_dtype), device)
                context = {'mpi_rank': mpi_rank, 'cfg': cfg,'log': log, 'iteration': i}
 

                    
                if comm_mode == "flatview":
                    if flat_group is not None:
                        time_barrier(group=flat_group, device=device)
                        with timer("(Flatview)"):
                            result = run_collective(x, op_obj, group=flat_group, dist=dist, framework=framework)
                            time_barrier(group=flat_group, device=device)
                        if enable_correctness:
                            check_collective_correctness(context, x, coll_name, op=op_obj, group=flat_group, result_data=result, group_type="Flatview", group_id="All")

                elif comm_mode == "within_node":
                    if my_within_group is not None:
                        time_barrier(group=my_within_group, device=device)
                        with timer(f"(Within-Group-{within_group_id})"):
                            result = run_collective(x, op_obj, group=my_within_group, dist=dist, log=log, framework=framework)
                            time_barrier(group=my_within_group, device=device)
                        if enable_correctness:
                            check_collective_correctness(context, x, coll_name, op=op_obj, group=my_within_group, result_data=result, group_type="Within", group_id=within_group_id)
             
                elif comm_mode == "across_node":
                    if my_across_group is not None:
                        time_barrier(group=my_across_group , device=device)
                        with timer(f"(Across-Group-{across_group_id})"):
                            result = run_collective(x, op_obj, group=my_across_group, dist=dist, log=log, framework=framework)
                            time_barrier(group=my_across_group,  device=device)
                        if enable_correctness:
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
            gather_and_print_all_bandwidths(log, cfg, mpi_size, ranks_responsible_for_logging, "[BANDWIDTH]", adjusted_buffer_sizes_single, comm_mode, mode_cfg, coll_name)
            
            # Only rank 0 prints remaining analysis
            if mpi_rank == 0:

                log.info("-------------------------------------------------------------------------")
                log.info("[MPI] Job complete")
                log.info("-------------------------------------------------------------------------")
                
                if cfg.extended_logging:
                    log.info("Querying Default Table selection")

                    terminal_log_path = os.path.join(log_dir, "terminal_output.log")
                    if os.path.exists(terminal_log_path):
                        if ccl_backend in ["nccl", "rccl"]:
                            scale_up_alg = getattr(coll_cfg, 'scale_up_algorithm', None)
                            scale_out_alg = getattr(coll_cfg, 'scale_out_algorithm', None)
                            report_nccl_selection(terminal_log_path, coll_name, log, scale_up_alg, scale_out_alg)
                        else:
                            # Get user's configured algorithms for display (preserve original values)
                            scale_up_alg = getattr(coll_cfg, 'scale_up_algorithm', None)
                            scale_out_alg = getattr(coll_cfg, 'scale_out_algorithm', None)
                            report_ccl_selection(terminal_log_path, coll_name, log, scale_up_alg, scale_out_alg)
                    else:
                        log.info(f"[SELECTION] Terminal output log not found: {terminal_log_path}")

                log.info("-------------------------------------------------------------------------")
                if len(available_modes) > 1:
                    log.info(f"[EXIT] Mode {mode_index + 1}/{len(available_modes)} ({comm_mode.upper()}) completed.")
                else:
                    log.info("[EXIT] All Done.")
                log.info("-------------------------------------------------------------------------")
 
    
    if mpi_rank == 0 and len(tasks_to_run) > 1:
        log.info("")
        log.info("=" * 80)
        log.info(f"[FINAL] All {len(tasks_to_run)} tasks completed successfully!")
        log.info("=" * 80)
        
    # ----------------------------------------------------------------------------
    #  CLEAN UP
    # ----------------------------------------------------------------------------

    DLCOMMLogger.flush()
    DLCOMMLogger.reset()
    MPI.COMM_WORLD.Barrier()
    if framework == "pytorch":
        dist.destroy_process_group()
    elif framework == "jax":
        jdist.shutdown()
    reset_times()
    
if __name__ == "__main__":
    main()