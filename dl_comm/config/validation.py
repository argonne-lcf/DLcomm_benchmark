import torch
import torch.distributed as dist
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
        comm_groups = implementation_config.comm_groups
        valid_modes = ["within_node", "across_node", "flatview"]
        
        # Validate the specific mode passed to this function
        if comm_mode not in valid_modes:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] Invalid comm_mode '{comm_mode}'. Valid: {valid_modes}")
            has_errors = True
        
        # Mode-specific validation
        if comm_mode == "within_node":
            if not hasattr(comm_groups, 'within_node'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'within_node' requires 'within_node' configuration")
                has_errors = True
            else:
                within_config = comm_groups.within_node
                if not hasattr(within_config, 'num_gpus_per_node') or not hasattr(within_config, 'gpu_ids_per_node'):
                    if mpi_rank == 0:
                        log.error("[VALIDATION] within_node config requires 'num_gpus_per_node' and 'gpu_ids_per_node'")
                    has_errors = True
                else:
                    # Validate that num_gpus_per_node matches length of gpu_ids_per_node
                    if within_config.num_gpus_per_node != len(within_config.gpu_ids_per_node):
                        if mpi_rank == 0:
                            log.error(f"[VALIDATION] within_node: num_gpus_per_node ({within_config.num_gpus_per_node}) must equal length of gpu_ids_per_node ({len(within_config.gpu_ids_per_node)})")
                        has_errors = True
                
                # Validate buffer size
                if hasattr(within_config, 'collective'):
                    if hasattr(within_config.collective, 'payload') and hasattr(within_config.collective.payload, 'buffer_size'):
                        try:
                            buffer_bytes = parse_buffer_size(within_config.collective.payload.buffer_size)
                        except ValueError as e:
                            if mpi_rank == 0:
                                log.error(f"[VALIDATION] Within-node: {e}")
                            has_errors = True
        
        elif comm_mode == "across_node":
            if not hasattr(comm_groups, 'across_node'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'across_node' requires 'across_node' configuration")
                has_errors = True
            else:
                across_config = comm_groups.across_node
                if not hasattr(across_config, 'num_compute_nodes') or not hasattr(across_config, 'num_gpus_per_node') or not hasattr(across_config, 'gpu_ids_per_node'):
                    if mpi_rank == 0:
                        log.error("[VALIDATION] across_node config requires 'num_compute_nodes', 'num_gpus_per_node' and 'gpu_ids_per_node'")
                    has_errors = True
                else:
                    # Validate that num_gpus_per_node matches length of gpu_ids_per_node
                    if across_config.num_gpus_per_node != len(across_config.gpu_ids_per_node):
                        if mpi_rank == 0:
                            log.error(f"[VALIDATION] across_node: num_gpus_per_node ({across_config.num_gpus_per_node}) must equal length of gpu_ids_per_node ({len(across_config.gpu_ids_per_node)})")
                        has_errors = True
                
                # Validate buffer size
                if hasattr(across_config, 'collective'):
                    if hasattr(across_config.collective, 'payload') and hasattr(across_config.collective.payload, 'buffer_size'):
                        try:
                            buffer_bytes = parse_buffer_size(across_config.collective.payload.buffer_size)
                        except ValueError as e:
                            if mpi_rank == 0:
                                log.error(f"[VALIDATION] Across-node: {e}")
                            has_errors = True
        
        elif comm_mode == "flatview":
            if not hasattr(comm_groups, 'flatview'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'flatview' requires 'flatview' configuration")
                has_errors = True
            else:
                flatview_config = comm_groups.flatview
                if not hasattr(flatview_config, 'num_compute_nodes') or not hasattr(flatview_config, 'num_gpus_per_node') or not hasattr(flatview_config, 'gpu_ids_per_node'):
                    if mpi_rank == 0:
                        log.error("[VALIDATION] flatview config requires 'num_compute_nodes', 'num_gpus_per_node' and 'gpu_ids_per_node'")
                    has_errors = True
                else:
                    # Validate that num_gpus_per_node matches length of gpu_ids_per_node
                    if flatview_config.num_gpus_per_node != len(flatview_config.gpu_ids_per_node):
                        if mpi_rank == 0:
                            log.error(f"[VALIDATION] flatview: num_gpus_per_node ({flatview_config.num_gpus_per_node}) must equal length of gpu_ids_per_node ({len(flatview_config.gpu_ids_per_node)})")
                        has_errors = True
                
                # Validate buffer size
                if hasattr(flatview_config, 'collective'):
                    if hasattr(flatview_config.collective, 'payload') and hasattr(flatview_config.collective.payload, 'buffer_size'):
                        try:
                            buffer_bytes = parse_buffer_size(flatview_config.collective.payload.buffer_size)
                        except ValueError as e:
                            if mpi_rank == 0:
                                log.error(f"[VALIDATION] Flatview: {e}")
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

                    
 
        if torch.cuda.is_available():
            available_devices = torch.cuda.device_count()
        elif torch.xpu.is_available():
            available_devices = torch.xpu.device_count()
        else:
            available_devices = 1   
        
        def validate_basic_config(config_section, mode_name): 
            nonlocal has_errors
            num_gpus = config_section.num_gpus_per_node
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