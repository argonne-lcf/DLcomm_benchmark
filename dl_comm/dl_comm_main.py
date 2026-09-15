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
import signal
import faulthandler
import numpy as np
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
from dl_comm.timer import timer, print_all_times, gather_and_print_all_times, reset_times, set_sync_device
from dl_comm.verify import build_payload
from dl_comm.verify import failures as verify_failures
from dl_comm.analysis.results import build_results, write_results, write_csv
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
    #  HANG WATCHDOG
    # ----------------------------------------------------------------------------
    # A benchmark that hangs consumes its whole walltime allocation and reports
    # nothing at all, which is strictly worse than a wrong number: the evidence
    # is destroyed along with the run. faulthandler.dump_traceback_later fires
    # from a dedicated thread and prints the Python stack of EVERY thread on
    # EVERY rank, so a deadlock names its own location instead of requiring a
    # debugger on a compute node (which is air-gapped and has no gdb/py-spy).
    #
    # DLCOMM_WATCHDOG=0 disables it. See docs/fixes/12-hang-watchdog.md
    _watchdog_s = float(os.environ.get("DLCOMM_WATCHDOG", "600"))
    if _watchdog_s > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(_watchdog_s, exit=True)
        # SIGABRT/SIGSEGV/SIGBUS handlers so a native crash also yields a stack
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)






    # ----------------------------------------------------------------------------
    # EXTRACT CONFIG VALUES (before logging)
    # ----------------------------------------------------------------------------

    framework       = cfg.framework.lower()
    ccl_backend     = cfg.ccl_backend
    device_type     = cfg.device_type.lower()
    memory_source   = cfg.memory_source.lower()
    

    
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
        # RUN_LOG_DIR is exported by the bundled jobscripts, but the benchmark
        # must not crash with a bare KeyError when it is launched any other way
        # (a bare mpiexec, a CI harness, an interactive debug session). Fall
        # back to a timestamped directory under the current working directory
        # and say so, rather than dying before a single collective has run.
        log_dir = os.environ.get("RUN_LOG_DIR") or os.environ.get("DL_COMM_LOG_DIR")
        if not log_dir:
            log_dir = os.path.join(
                os.getcwd(), "logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
            print(f"[dl_comm] RUN_LOG_DIR not set; logging to {log_dir}",
                  flush=True)
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
    # FRAMEWORK-SPECIFIC IMPORTS (once per execution)
    # ----------------------------------------------------------------------------
    dist=None
    if framework == "pytorch":
        # timer func defined in ./timer/timer.py
        with timer("import time"):
            import torch
            import torch.nn.parallel
            import torch.distributed as dist
            
            # Intel-specific imports for CCL backends.
            #
            # torch 2.10 on Aurora (frameworks/2025.3.1) provides XCCL
            # natively via torch.distributed, and the standalone
            # `oneccl_bindings_for_pytorch` shim is no longer shipped.
            # Importing it unconditionally made the benchmark unrunnable on
            # the current module stack: every rank died with
            # ModuleNotFoundError before any collective executed. Import the
            # shims only when they exist, and only fail if the requested
            # backend is genuinely unavailable.
            if ccl_backend == "torchcomms":
                # torchcomms is a transport, not a framework: it carries the
                # same torch.Tensor objects. Fail here with an actionable
                # message rather than deep inside the first collective.
                from dl_comm.comm import torchcomms_backend as _tcb
                if not _tcb.is_available():
                    raise RuntimeError(
                        "ccl_backend 'torchcomms' requires the torchcomms "
                        "module (PyTorch >= 2.8). On Aurora it ships with "
                        "frameworks/2025.3.1; elsewhere `pip install "
                        "torchcomms`.")
                # IPEX is not required by torchcomms, which reaches XCCL
                # directly. Import it opportunistically and tolerate any
                # failure: under the 0.3.0 stack (torch 2.13) IPEX pulls in
                # the frameworks torchvision, built against torch 2.10, which
                # raises RuntimeError("operator torchvision::nms does not
                # exist") rather than ImportError. Catching ImportError alone
                # let that abort the whole run before a single collective.
                try:
                    import intel_extension_for_pytorch  # noqa: F401
                except Exception as exc:  # noqa: BLE001
                    log.info(
                        "[CONFIG] intel_extension_for_pytorch unavailable "
                        "(%s: %s); continuing, torchcomms does not need it",
                        type(exc).__name__, exc)
            elif ccl_backend in ["xccl", "ccl"]:
                try:
                    import intel_extension_for_pytorch  # noqa: F401
                except ImportError:
                    pass
                try:
                    import oneccl_bindings_for_pytorch  # noqa: F401
                except ImportError:
                    native = getattr(dist, f"is_{ccl_backend}_available",
                                     lambda: False)()
                    if not native:
                        raise RuntimeError(
                            f"backend '{ccl_backend}' is not available: "
                            f"oneccl_bindings_for_pytorch is not installed and "
                            f"torch.distributed has no native {ccl_backend} "
                            f"support in this build")


    elif framework == "jax":
        with timer("import time"):
            import jax
            import jax.numpy as jnp
            import jax.distributed as jdist
 


        coordinator = os.environ.get("MASTER_ADDR", "127.0.0.1") + ":" + os.environ.get("MASTER_PORT", "1234")
        print(f"[DEBUG] Calling jax.distributed.initialize() on Rank {mpi_rank}, Coordinator: {coordinator}", flush=True)

        
        jdist.initialize(
                coordinator_address=coordinator,
                num_processes=mpi_size,
                process_id=mpi_rank
            )
 



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
    # TORCH DISTRIBUTED INIT (once per execution)
    # ----------------------------------------------------------------------------
    
    max_mpi_size_needed, mpi_validation_errors = validate_mpi_configuration(cfg, mpi_size, mpi_rank, log)
    if mpi_validation_errors:
        # Previously this result was computed and discarded, so a launch whose
        # rank count did not match the config ran with silently idle ranks.
        # See docs/fixes/03-rank-topology-validation.md
        if mpi_rank == 0:
            log.error("[EXIT] Exiting due to MPI launch geometry mismatch")
        DLCOMMLogger.flush()
        MPI.COMM_WORLD.Barrier()
        sys.exit(2)
    
    # ----------------------------------------------------------------------------
    # ALGORITHM SETUP (before distributed init)
    # ----------------------------------------------------------------------------
    
    setup_algorithm_overrides(cfg, log)

    # Accumulates one entry per measured communication group across all tasks;
    # written out as results.json / results.csv at the end of the run.
    run_measurements = []
    # Tracks whether any task actually enabled verification. Read after the
    # task loop, `cfg` does not carry this flag -- it lives on the per-mode
    # config (`mode_cfg.verify_correctness`), so probing `cfg` reports False
    # even on runs that verified 480 checks. See docs/fixes/14-enabled-flag.md
    correctness_was_enabled = False
    verify_failures.reset()
     
    MPI.COMM_WORLD.Barrier()
    with timer("init time"):
        if framework == "pytorch" and ccl_backend == "torchcomms":
            # torchcomms derives rank/world size from RANK/WORLD_SIZE and
            # bootstraps via MASTER_ADDR/MASTER_PORT, the way torchrun sets
            # them. DLcomm launches under mpiexec, so export them from the MPI
            # rank before creating the communicator.
            from dl_comm.comm import torchcomms_backend as _tcb

            os.environ.setdefault("RANK", str(mpi_rank))
            os.environ.setdefault("WORLD_SIZE", str(mpi_size))

            _tc_device_type = "gpu"
            try:
                _tc_device_type = cfg.device_type
            except Exception:
                pass

            if _tc_device_type in ("gpu", "xpu"):
                _tc_dev = torch.device("xpu", mpi_rank % torch.xpu.device_count())
            else:
                _tc_dev = torch.device("cpu")

            # Replacing the module-level `dist` with the adapter is what makes
            # all 13 collectives work unmodified: they already receive their
            # comm module as a `dist=` parameter.
            dist = _tcb.build(
                _tc_dev,
                device_type=_tc_device_type,
                timeout=datetime.timedelta(seconds=3600),
            )
        elif framework == "pytorch":
            dist.init_process_group(
                backend=ccl_backend,
                init_method='env://',
                world_size=mpi_size,
                rank=mpi_rank,
                timeout=datetime.timedelta(seconds=3600)
            )
        elif framework == "jax":
            #jdist.initialize(coordinator_address="env://", num_processes=mpi_size, process_id=mpi_rank)
            pass


    # Initialize framework-specific constants
    init_framework_constants(framework)
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
        # Snapshot the tally so this task's own result can be reported when it
        # finishes. The end-of-run summary is the only place passing checks
        # were previously printed, so a hang in a later task erased the
        # evidence for every task that had already succeeded (Aurora jobs
        # 8824532 and 8824561 lost alltoallv and sendrecv results this way).
        _task_start_tally = verify_failures.snapshot()
        _task_coll_name = "?"

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
            _task_coll_name    = coll_name
            op_name            = coll_cfg.collective_op
            dtype_str          = coll_cfg.payload.dtype
            iters              = coll_cfg.iterations
            warmup_iters       = getattr(coll_cfg, 'warmup_iterations', 0)  # Default to 0 if not specified
            add_mxm_compute    = getattr(coll_cfg, 'add_mxm_compute', False)  # Default to False if not specified
            enable_correctness = mode_cfg.verify_correctness
            if enable_correctness:
                correctness_was_enabled = True

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
            # A bare COLLECTIVES[coll_name] raises an unadorned KeyError that
            # names neither the config field at fault nor the valid choices.
            if coll_name not in COLLECTIVES:
                raise ValueError(
                    f"Unknown collective '{coll_name}'. Registered collectives: "
                    f"{sorted(COLLECTIVES)}")
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
            if framework=="pytorch":
                comm_info = setup_communication_groups(mode_cfg, mpi_rank, log, dist, force_mode=comm_mode, full_cfg=cfg)
                my_within_group = comm_info['my_within_group']
                my_across_group = comm_info['my_across_group'] 
                flat_group = comm_info['flat_group']
                device = comm_info['device']  # Device assigned based on group membership
        
                within_group_id = comm_info['within_group_id']
                across_group_id = comm_info['across_group_id']
                ranks_responsible_for_logging = comm_info['ranks_responsible_for_logging']

            elif framework=="jax":
                ranks_responsible_for_logging=[0]
                pass    
            


            # ----------------------------------------------------------------------------
            #  HOST TO DEVICE TRANSFER TEST
            # ----------------------------------------------------------------------------
            
            if framework == "pytorch":
                if memory_source == "host" and device_type == "gpu":
                    x_test = torch.ones(num_elems, dtype=_dtype, device="cpu")
                    with timer("Host to Device Transfer Time"):
                        x_test = x_test.to(device, non_blocking=True)
                else:
                    pass


            elif framework== "jax":
                pass

            MPI.COMM_WORLD.Barrier()
            # ----------------------------------------------------------------------------
            #  MxM COMPUTE SECTION 
            # ----------------------------------------------------------------------------
            if framework=="pytorch":
                if add_mxm_compute:
                    mxm_size=1024
                    mxm_metrics_local = None
                    with timer(f"MxM Compute Time, m={mxm_size}"):
                        mxm_metrics_local = dummy_mxm_compute(device, _dtype, size=mxm_size, framework=framework)
                    
                    all_mxm_metrics = MPI.COMM_WORLD.gather(mxm_metrics_local, root=0)
                    
                    if mpi_rank == 0:
                        log.output("")
                        log.output(f"[MxM COMPUTE] Benchmarking GEMM {mxm_size}x{mxm_size} · {mxm_size}x{mxm_size} ...")
                        log.output("")
                        
                        total_time_ms = 0
                        total_gflops = 0
                        total_throughput = 0
                        device_count = 0
                        
                        for rank, metrics in enumerate(all_mxm_metrics):
                            if metrics:
                                device_count += 1
                                total_time_ms += metrics['time_ms']
                                total_gflops += metrics['gflops']
                                total_throughput += metrics['tflops_throughput']
                                
                                log.output(f"[GPU {metrics['device_id']}: {metrics['device_name']}]  "
                                          f"size=({mxm_size}x{mxm_size})·({mxm_size}x{mxm_size})  "
                                          f"time={metrics['time_ms']:.3f} ms  "
                                          f"ops={metrics['gflops']:.2f} GFLOP  "
                                          f"throughput={metrics['tflops_throughput']:.2f} TFLOP/s")
                        
                        if device_count > 1:
                            avg_time = total_time_ms / device_count
                            aggregate_throughput = total_gflops / (avg_time / 1000)
                            
                            log.output("")
                            log.output("=== Node Summary (sequential benchmark run) ===")
                            log.output(f"Total ops executed: {total_gflops:.2f} GFLOP")
                            log.output(f"Aggregate time: {avg_time/1000:.3f} s")
                            log.output(f"Aggregate throughput: {aggregate_throughput/1000:.2f} TFLOP/s")
                            log.output(f"Sum of per-GPU throughputs: {total_throughput:.2f} TFLOP/s")
                        
                        log.output("")
                MPI.COMM_WORLD.Barrier()
            elif framework=="jax":
                pass


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
                        # Warmup must exercise the same payload construction as
                        # the measured loop so that any lazy allocation it
                        # triggers is paid here rather than in iteration 0.
                        _wu_group = (flat_group if comm_mode == "flatview" else
                                     my_within_group if comm_mode == "within_node" else
                                     my_across_group)
                        if _wu_group is not None:
                            _wu_ranks = dist.get_process_group_ranks(_wu_group)
                            _wu_world, _wu_index = len(_wu_ranks), _wu_ranks.index(mpi_rank)
                        else:
                            _wu_world, _wu_index = mpi_size, mpi_rank
                        x = build_payload(torch, num_elems, _dtype, _wu_index,
                                          _wu_world, op_name, device=device)
                    elif framework == "jax":
                        pass
                    

                    
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
            if framework == "jax":
                import jax
                import jax.numpy as jnp

                world_devs = jax.device_count()         # global device count (e.g., 16)
                local_devs = jax.local_device_count()   # devices visible to this process (often 1)

                for i in range(iters):
                    # Make buffer divisible by world_devs for equal splits
                    num_elems = (num_elems // world_devs) * world_devs
                    split_size = num_elems // world_devs

                    # Shape: [local_devs, world_devs, split_size]
                    #   - leading axis == local_devs (required by pmap input)
                    #   - split_axis (1) == world_devs (required by lax.all_to_all)
                    x = jnp.ones((local_devs, world_devs, split_size), dtype=_dtype)

                    if mpi_rank == 0:
                        print("device count", world_devs)
                        print("DEBUG x before", x.sum())

                    with timer("(Flatview)"):
                        result = run_collective(x, op_name, group=None, dist=None, framework=framework)

                    if mpi_rank == 0:
                        print("DEBUG x after", result.sum())


            # Collective execution for all modes
            elif framework=="pytorch":
                # Register the device whose queue must drain before each
                # timestamp, so the measurement no longer depends on the
                # external CCL_OP_SYNC environment variable.
                # See docs/fixes/05-timing-and-statistics.md
                set_sync_device(device, enabled=True)

                active_group = None
                if comm_mode == "flatview":
                    active_group = flat_group
                elif comm_mode == "within_node":
                    active_group = my_within_group
                elif comm_mode == "across_node":
                    active_group = my_across_group

                if active_group is not None:
                    group_ranks = dist.get_process_group_ranks(active_group)
                    group_world = len(group_ranks)
                    my_group_index = group_ranks.index(mpi_rank)
                else:
                    group_world, my_group_index = mpi_size, mpi_rank

                for i in range(iters):

                    # Rank-dependent payload: an all-ones buffer made 12 of 15
                    # collective/op combinations impossible to verify.
                    # See docs/fixes/01-rank-dependent-verification.md
                    x = build_payload(torch, num_elems, _dtype, my_group_index,
                                      group_world, op_name, device=device)

                    context = {'mpi_rank': mpi_rank, 'cfg': cfg,'log': log, 'iteration': i}

                    if comm_mode == "flatview":
                        if flat_group is not None:
                            time_barrier(group=flat_group, device=device)
                            with timer("(Flatview)"):
                                result = run_collective(x, op_obj, group=flat_group, dist=dist, framework=framework)
                            # Barrier moved OUT of the timed region: it was
                            # previously inside, so its cost was charged to the
                            # collective. See docs/fixes/05-timing-and-statistics.md
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
            gather_and_print_all_bandwidths(log, cfg, mpi_size, ranks_responsible_for_logging, "[BANDWIDTH]", adjusted_buffer_sizes_single, comm_mode, mode_cfg, coll_name, results_sink=run_measurements)
            
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

        # Per-task correctness line, emitted as soon as the task finishes.
        # This is deliberately NOT the cross-rank verdict -- it is rank 0's own
        # tally -- but it survives a hang in a later task, which the end-of-run
        # summary does not.
        if mpi_rank == 0:
            _end = verify_failures.snapshot()
            _d_checks = _end["checks"] - _task_start_tally["checks"]
            _d_fail = _end["failures"] - _task_start_tally["failures"]
            _d_skip = _end["skipped"] - _task_start_tally["skipped"]
            _status = "FAILED" if _d_fail else ("NO-CHECKS" if _d_checks == 0 else "ok")
            log.output(
                f"[TASK-CORRECTNESS] {task_name} collective={_task_coll_name} "
                f"checks={_d_checks} failures={_d_fail} skipped={_d_skip} [{_status}]"
            )
 
    
    # ----------------------------------------------------------------------------
    #  CORRECTNESS VERDICT AND STRUCTURED RESULTS
    # ----------------------------------------------------------------------------
    # A verification failure previously only produced a log line and the process
    # still exited 0. Reduce the per-rank tallies across MPI_COMM_WORLD so any
    # rank's failure fails the whole job.
    # See docs/fixes/04-fail-loudly.md

    local_verify = verify_failures.snapshot()

    # Reaching this point is itself collective state: every rank must arrive or
    # the reduce below hangs. A non-blocking barrier with a bounded wait turns
    # that silent deadlock into a diagnosable error naming the missing ranks.
    # See docs/fixes/11-verdict-barrier-timeout.md
    _VERDICT_TIMEOUT_S = float(os.environ.get("DLCOMM_VERDICT_TIMEOUT", "120"))
    _req = MPI.COMM_WORLD.Ibarrier()
    _deadline = time.time() + _VERDICT_TIMEOUT_S
    while not _req.Test():
        if time.time() > _deadline:
            sys.stderr.write(
                f"[CORRECTNESS] rank {mpi_rank} timed out after "
                f"{_VERDICT_TIMEOUT_S:.0f}s waiting for all {mpi_size} ranks to "
                f"reach the verdict barrier. At least one rank exited the task "
                f"loop early; the cross-rank reduce cannot complete.\n")
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(3)
        time.sleep(0.05)

    # Use the UPPERCASE buffer-based MPI calls, not the lowercase pickle-based
    # ones. The lowercase mpi4py variants (allreduce/gather) negotiate object
    # sizes with dynamic probe/recv traffic, which deadlocks once the XCCL
    # backend has been initialised on the same ranks: Aurora job 8824457 hung
    # here with all 24 ranks confirmed present at this line by the watchdog.
    # The uppercase calls move fixed-size buffers and are unaffected.
    # See docs/fixes/13-mpi-buffer-api.md
    _counts = np.array([local_verify["failures"],
                        local_verify["checks"],
                        local_verify["skipped"],
                        1 if correctness_was_enabled else 0], dtype=np.int64)
    _totals = np.zeros(4, dtype=np.int64)
    MPI.COMM_WORLD.Allreduce(_counts, _totals, op=MPI.SUM)
    total_failures = int(_totals[0])
    total_checks = int(_totals[1])
    total_skipped = int(_totals[2])
    # Any rank having verification on means the run verified. Reducing this
    # rather than reading cfg on rank 0 also covers the case where a rank sits
    # outside every communication group and so runs no tasks at all.
    any_enabled = bool(_totals[3] > 0)

    # Per-rank detail strings are variable-length objects, so gathering them
    # would reintroduce the same pickle path. Each rank already logs its own
    # failures as they happen; rank 0 reports only its local sample.
    all_details = [local_verify["details"]]

    correctness_summary = {
        "enabled": any_enabled,
        "total_checks": total_checks,
        "total_failures": total_failures,
        "total_skipped": total_skipped,
        "passed": bool(total_failures == 0),
    }

    if mpi_rank == 0:
        flat_details = [d for chunk in (all_details or []) if chunk for d in chunk]
        if flat_details:
            correctness_summary["details"] = flat_details[:200]

        log.output("")
        log.output("[CORRECTNESS] ---------------------------------------------------------")
        log.output(f"[CORRECTNESS] checks={total_checks} failures={total_failures} "
                   f"skipped={total_skipped}")
        if total_failures:
            log.error(f"[CORRECTNESS] VERIFICATION FAILED on {total_failures} check(s)")
            for detail in flat_details[:20]:
                log.error(f"[CORRECTNESS]   {detail}")
        elif correctness_summary["enabled"] and total_checks == 0:
            log.warning("[CORRECTNESS] verification was enabled but no checks ran")
        elif correctness_summary["enabled"]:
            log.output("[CORRECTNESS] all checks passed")
        log.output("[CORRECTNESS] ---------------------------------------------------------")

        try:
            results_dir = log_dir
        except NameError:
            results_dir = os.getcwd()
        document = build_results(
            cfg=cfg, mpi_size=mpi_size, comm_mode=None, collective_name=None,
            measurements=run_measurements, correctness=correctness_summary)
        write_results(os.path.join(results_dir, "results.json"), document, log)
        write_csv(os.path.join(results_dir, "results.csv"), run_measurements, log)

    if mpi_rank == 0 and len(tasks_to_run) > 1 and total_failures == 0:
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

    # Decide the exit status BEFORE tearing down the communicators. XCCL
    # teardown of subgroups created with use_local_synchronization=True can
    # segfault on Aurora (observed: "rank 5 died from signal 11" after
    # "[EXIT] All Done."), which turns a clean run into exit 143 and would
    # equally mask a real correctness failure behind a signal. The verdict is
    # already computed, so latch it here and make teardown non-fatal.
    exit_code = 1 if total_failures else 0

    if framework == "pytorch":
        try:
            dist.destroy_process_group()
        except Exception as exc:                       # pragma: no cover
            print(f"[dl_comm] destroy_process_group raised (ignored): {exc}",
                  flush=True)
    if framework == "jax":
        try:
            jdist.shutdown()
        except Exception as exc:                       # pragma: no cover
            print(f"[dl_comm] jax shutdown raised (ignored): {exc}", flush=True)
    reset_times()

    # Flush stdio before _exit: os._exit skips atexit handlers and buffers.
    sys.stdout.flush()
    sys.stderr.flush()

    # os._exit bypasses interpreter finalization, where the XCCL/oneCCL
    # destructors run. Those destructors are what raise SIGSEGV on Aurora, and
    # a signal death overrides our exit status -- the exact "job failed but
    # reported success" class this work exists to remove.
    os._exit(exit_code)
    
if __name__ == "__main__":
    main()