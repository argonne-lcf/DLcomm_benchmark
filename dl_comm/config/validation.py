from omegaconf import DictConfig



def parse_buffer_size(size_str: str) -> int:
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


def validate_and_calculate_buffer_size(payload_config, mode_name: str, log=None, mpi_rank: int = 0) -> tuple[int, int, bool]:
    from dl_comm.comm import DTYPES
    has_errors = False
    
    if not hasattr(payload_config, 'dtype'):
        if mpi_rank == 0 and log:
            log.error(f"[VALIDATION] {mode_name}: 'dtype' is required in payload configuration")
        has_errors = True
        return 0, 0, has_errors
    
    dtype_str = payload_config.dtype
    if dtype_str not in DTYPES:
        if mpi_rank == 0 and log:
            log.error(f"[VALIDATION] {mode_name}: Unsupported dtype '{dtype_str}'. Supported types: {list(DTYPES.keys())}")
        has_errors = True
        return 0, 0, has_errors
    
    # Framework-specific dtype handling will be added later
    torch_dtype, elem_size_bytes = DTYPES[dtype_str]
    
    has_buffer_size = hasattr(payload_config, 'buffer_size') and payload_config.buffer_size is not None
    has_count = hasattr(payload_config, 'count') and payload_config.count is not None
    
    if has_buffer_size and has_count:
        if mpi_rank == 0 and log:
            log.error(f"[VALIDATION] {mode_name}: Cannot specify both 'buffer_size' and 'count'. Use either 'buffer_size' OR 'count' + 'dtype'")
        has_errors = True
        return 0, 0, has_errors
    
    if not has_buffer_size and not has_count:
        if mpi_rank == 0 and log:
            log.error(f"[VALIDATION] {mode_name}: Must specify either 'buffer_size' OR 'count' in payload configuration")
        has_errors = True
        return 0, 0, has_errors
    
    if has_buffer_size:
        try:
            buffer_size_bytes = parse_buffer_size(payload_config.buffer_size)
            if buffer_size_bytes % elem_size_bytes != 0:
                if mpi_rank == 0 and log:
                    log.error(f"[VALIDATION] {mode_name}: buffer_size ({buffer_size_bytes} bytes) must be divisible by dtype size ({elem_size_bytes} bytes for {dtype_str})")
                has_errors = True
                return 0, 0, has_errors
            num_elements = buffer_size_bytes // elem_size_bytes
        except ValueError as e:
            if mpi_rank == 0 and log:
                log.error(f"[VALIDATION] {mode_name}: {e}")
            has_errors = True
            return 0, 0, has_errors
        
    elif has_count:
        try:
            num_elements = int(payload_config.count)
            if num_elements <= 0:
                if mpi_rank == 0 and log:
                    log.error(f"[VALIDATION] {mode_name}: 'count' must be a positive integer, got {num_elements}")
                has_errors = True
                return 0, 0, has_errors
            buffer_size_bytes = num_elements * elem_size_bytes
        except (ValueError, TypeError) as e:
            if mpi_rank == 0 and log:
                log.error(f"[VALIDATION] {mode_name}: Invalid count value: {e}")
            has_errors = True
            return 0, 0, has_errors
    
    return buffer_size_bytes, num_elements, has_errors


def adjust_buffer_size_for_group_divisibility(buffer_bytes: int, group_size: int, collective_name: str, elem_size: int, log=None, mpi_rank: int = 0) -> tuple[int, str]:
    
    collectives_needing_divisibility = ["alltoallsingle"]
    

    if collective_name.lower() not in collectives_needing_divisibility:
        return buffer_bytes, ""
    
    num_elems = buffer_bytes // elem_size
    
    if num_elems % group_size != 0:
        remainder = num_elems % group_size
        
        adjusted_elems_up = num_elems + (group_size - remainder)
        adjusted_bytes_up = adjusted_elems_up * elem_size
        
        adjusted_elems_down = num_elems - remainder
        adjusted_bytes_down = adjusted_elems_down * elem_size
        
        diff_up = abs(adjusted_bytes_up - buffer_bytes)
        diff_down = abs(adjusted_bytes_down - buffer_bytes)
        
        if diff_up <= diff_down and adjusted_elems_up > 0:
            adjusted_bytes = adjusted_bytes_up
            adjusted_elems = adjusted_elems_up
            direction = "up"
        elif adjusted_elems_down > 0:
            adjusted_bytes = adjusted_bytes_down
            adjusted_elems = adjusted_elems_down
            direction = "down"
        else:
            adjusted_bytes = adjusted_bytes_up
            adjusted_elems = adjusted_elems_up
            direction = "up"
        
        adjustment_msg = f"[BUFFER ADJUSTMENT] {collective_name}: Adjusted buffer size from {buffer_bytes} bytes ({num_elems} elements) to {adjusted_bytes} bytes ({adjusted_elems} elements) - rounded {direction} to be divisible by group size {group_size}"
        
        return adjusted_bytes, adjustment_msg
    
    return buffer_bytes, ""




class ConfigValidator:
    def __init__(self, spec: dict):
        self.spec = spec
        self.backend_warning_shown = False

    def validate_implementation_names(self, cfg: DictConfig, mpi_rank: int, log):
        """Validate that implementation names are unique"""
        has_errors = False
        implementation_names = []
        
        for impl_config in cfg.implementations:
            if hasattr(impl_config, 'task_name'):
                impl_name = impl_config.task_name
                if impl_name in implementation_names:
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] Duplicate implementation name '{impl_name}' found. Implementation names must be unique.")
                    has_errors = True
                else:
                    implementation_names.append(impl_name)
            else:
                if mpi_rank == 0:
                    log.error(f"[VALIDATION] Implementation missing 'task_name' field")
                has_errors = True
        
        return has_errors

    def validate(self, cfg: DictConfig, implementation_config, comm_mode: str, mpi_rank: int, log):
     
        has_errors = False
        buffer_bytes = None

        # framework
        framework = cfg.framework
        if framework not in self.spec["framework"]:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] Invalid framework '{framework}'. Valid options: {self.spec['framework']}")
            has_errors = True

        # ccl_backend
        backend = getattr(cfg, "ccl_backend", None)
        valid_backends = self.spec["backend"].get(framework, [])
        if backend not in valid_backends:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] Invalid ccl_backend '{backend}' for framework '{framework}'. Valid: {valid_backends}")
            has_errors = True

        # Buffer size validation - extract from active communication mode
        buffer_bytes = None

        # comm_group validation using new implementation structure
        comm_groups = implementation_config.comm_group
        valid_modes = ["within_node", "across_node", "flatview"]
        
        # Validate the specific mode passed to this function
        if comm_mode not in valid_modes:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] Invalid comm_mode '{comm_mode}'. Valid: {valid_modes}")
            has_errors = True
        
        # Mode-specific validation
        if comm_mode == "within_node":
            # In old structure, the implementation_config itself is the within_node config
            within_config = implementation_config
            if not hasattr(within_config, 'num_devices_per_node') or not hasattr(within_config, 'device_ids_per_node'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] within_node config requires 'num_devices_per_node' and 'device_ids_per_node'")
                has_errors = True
            else:
                # Validate that num_devices_per_node matches length of device_ids_per_node
                if within_config.num_devices_per_node != len(within_config.device_ids_per_node):
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] within_node: num_devices_per_node ({within_config.num_devices_per_node}) must equal length of device_ids_per_node ({len(within_config.device_ids_per_node)})")
                    has_errors = True
                
                # Validate payload configuration
                if hasattr(within_config, 'collective') and hasattr(within_config.collective, 'payload'):
                    buffer_bytes, num_elements, payload_errors = validate_and_calculate_buffer_size(within_config.collective.payload, "within_node", log, mpi_rank)
                    if payload_errors:
                        has_errors = True
        
        elif comm_mode == "across_node":
            # In old structure, the implementation_config itself is the across_node config
            across_config = implementation_config
            if not hasattr(across_config, 'num_compute_nodes') or not hasattr(across_config, 'num_devices_per_node') or not hasattr(across_config, 'device_ids_per_node'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] across_node config requires 'num_compute_nodes', 'num_devices_per_node' and 'device_ids_per_node'")
                has_errors = True
            else:
                # Validate that num_devices_per_node matches length of device_ids_per_node
                if across_config.num_devices_per_node != len(across_config.device_ids_per_node):
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] across_node: num_devices_per_node ({across_config.num_devices_per_node}) must equal length of device_ids_per_node ({len(across_config.device_ids_per_node)})")
                    has_errors = True
                
                # Validate payload configuration
                if hasattr(across_config, 'collective') and hasattr(across_config.collective, 'payload'):
                    buffer_bytes, num_elements, payload_errors = validate_and_calculate_buffer_size(across_config.collective.payload, "across_node", log, mpi_rank)
                    if payload_errors:
                        has_errors = True
        
        elif comm_mode == "flatview":
            # In old structure, the implementation_config itself is the flatview config
            flatview_config = implementation_config
            if not hasattr(flatview_config, 'num_compute_nodes') or not hasattr(flatview_config, 'num_devices_per_node') or not hasattr(flatview_config, 'device_ids_per_node'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] flatview config requires 'num_compute_nodes', 'num_devices_per_node' and 'device_ids_per_node'")
                has_errors = True
            else:
                # Validate that num_devices_per_node matches length of device_ids_per_node
                if flatview_config.num_devices_per_node != len(flatview_config.device_ids_per_node):
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] flatview: num_devices_per_node ({flatview_config.num_devices_per_node}) must equal length of device_ids_per_node ({len(flatview_config.device_ids_per_node)})")
                    has_errors = True
                
                # Validate payload configuration
                if hasattr(flatview_config, 'collective') and hasattr(flatview_config.collective, 'payload'):
                    buffer_bytes, num_elements, payload_errors = validate_and_calculate_buffer_size(flatview_config.collective.payload, "flatview", log, mpi_rank)
                    if payload_errors:
                        has_errors = True

        # Validate operation is provided for collectives that need it
        if not has_errors:
            from dl_comm.comm import OPS_NEED_REDUCE, OP_MAP
            
            # Get the collective config for this mode - in old structure it's directly in implementation_config
            coll_cfg = implementation_config.collective
            
            collective_name = coll_cfg.collective_name.lower()
            op_name = getattr(coll_cfg, 'collective_op', None)
            
            if collective_name in OPS_NEED_REDUCE:
                if not op_name or op_name.strip() == '':
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] {comm_mode}: Collective '{collective_name}' requires an operation (op). Valid operations: {list(OP_MAP.keys())}")
                    has_errors = True
                elif op_name not in OP_MAP:
                    if mpi_rank == 0:
                        log.error(f"[VALIDATION] {comm_mode}: Invalid operation '{op_name}' for collective '{collective_name}'. Valid operations: {list(OP_MAP.keys())}")
                    has_errors = True

        # Ensure buffer_bytes is set
        if buffer_bytes is None and not has_errors:
            if mpi_rank == 0:
                log.error("[VALIDATION] Could not extract buffer size from configuration")
            has_errors = True

        if has_errors:
            if mpi_rank == 0:
                log.error("[VALIDATION] Configuration validation failed - please check configuration")
            return (False, None)

        return (True, buffer_bytes)

    def validate_runtime(self, cfg: DictConfig, mode_cfg, comm_mode: str, mpi_size: int, mpi_rank: int, log):
         
        has_errors = False
        
        # Validate distributed backend availability
        backend = cfg.ccl_backend.lower()
        
        if cfg.framework == "pytorch":
            import torch
            import torch.distributed as dist
            
            if backend == "nccl":
                try:
                    if not dist.is_nccl_available():
                        if mpi_rank == 0:
                            log.error("[VALIDATION] NCCL backend requested but not available")
                        has_errors = True
                except AttributeError:
                    if mpi_rank == 0:
                        log.warning("[VALIDATION] Cannot check NCCL availability (API not available)")
            
            elif backend == "mpi":
                try:
                    if not dist.is_mpi_available():
                        if mpi_rank == 0:
                            log.error("[VALIDATION] MPI backend requested but not available")
                        has_errors = True
                except AttributeError:
                    if mpi_rank == 0:
                        log.warning("[VALIDATION] Cannot check MPI availability (API not available)")
            
            elif backend in ["ccl", "xccl"]:
                try: 
                    if not torch.distributed.distributed_c10d.is_xccl_available():
                        if mpi_rank == 0:
                            log.error("[VALIDATION] CCL/XCCL backend requested but not available")
                        has_errors = True
                except (AttributeError, ImportError):
                    if mpi_rank == 0 and not self.backend_warning_shown:
                        log.warning("[VALIDATION] Cannot check CCL/XCCL availability (API not available)")
                        self.backend_warning_shown = True
        
        elif cfg.framework == "jax":
            pass

                    
 
        if cfg.framework == "pytorch":
            import torch
            if torch.cuda.is_available():
                available_devices = torch.cuda.device_count()
            elif torch.xpu.is_available():
                available_devices = torch.xpu.device_count()
            else:
                available_devices = 1
        elif cfg.framework == "jax":
            import jax
            available_devices = jax.device_count()   
        
        def validate_basic_config(config_section, mode_name): 
            nonlocal has_errors
            num_gpus = config_section.num_devices_per_node
            num_nodes = config_section.num_compute_nodes
            
             
            expected_total_ranks = num_nodes * num_gpus
            # Disabled rank count validation
            # if expected_total_ranks != mpi_size:
            #     if mpi_rank == 0:
            #         log.error(f"[VALIDATION] {mode_name}: Expected {expected_total_ranks} total ranks but got {mpi_size}")
            #     has_errors = True
            
   
            if available_devices < num_gpus:
                if mpi_rank == 0:
                    log.error(f"[VALIDATION] {mode_name}: Need {num_gpus} GPUs per node but only {available_devices} available")
                has_errors = True
        
        # Use the passed mode_cfg directly instead of extracting from cfg
        if comm_mode == "within_node":
            validate_basic_config(mode_cfg, "Within-node mode")
            
        elif comm_mode == "across_node":
            validate_basic_config(mode_cfg, "Across-node mode")
            
        
         
        
        if has_errors:
            if mpi_rank == 0:
                log.error("[VALIDATION] Runtime validation failed - please check configuration")
            return False
        
        return True