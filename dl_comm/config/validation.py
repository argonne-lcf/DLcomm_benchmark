import torch
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


class ConfigValidator:
    def __init__(self, spec: dict):
        self.spec = spec

    def validate(self, cfg: DictConfig, mpi_rank: int, log):
     
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

        # For now, skip collective validation as it's moved to mode-specific configs
        # TODO: Implement proper validation for new structure
        buffer_bytes = 1024  # Default for now

        # comm_group validation
        comm_group = cfg.comm_group
        comm_mode = comm_group.mode
        valid_modes = ["within_node", "across_node", "combined", "flatview"]
        
        if comm_mode not in valid_modes:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] Invalid comm_mode '{comm_mode}'. Valid: {valid_modes}")
            has_errors = True
        
        # Mode-specific validation
        if comm_mode == "within_node":
            if not hasattr(comm_group, 'within_node'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'within_node' requires 'within_node' configuration")
                has_errors = True
            else:
                within_config = comm_group.within_node
                if not hasattr(within_config, 'num_gpus_per_node') or not hasattr(within_config, 'gpu_ids_per_node'):
                    if mpi_rank == 0:
                        log.error("[VALIDATION] within_node config requires 'num_gpus_per_node' and 'gpu_ids_per_node'")
                    has_errors = True
        
        elif comm_mode == "across_node":
            if not hasattr(comm_group, 'across_node'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'across_node' requires 'across_node' configuration")
                has_errors = True
            else:
                across_config = comm_group.across_node
                if not hasattr(across_config, 'num_compute_nodes') or not hasattr(across_config, 'num_gpus_per_node') or not hasattr(across_config, 'gpu_ids_per_node'):
                    if mpi_rank == 0:
                        log.error("[VALIDATION] across_node config requires 'num_compute_nodes', 'num_gpus_per_node' and 'gpu_ids_per_node'")
                    has_errors = True
        
        elif comm_mode == "combined":
            if not hasattr(comm_group, 'combined'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'combined' requires 'combined' configuration")
                has_errors = True
            else:
                combined_config = comm_group.combined
                if not hasattr(combined_config, 'within_node') or not hasattr(combined_config, 'across_node'):
                    if mpi_rank == 0:
                        log.error("[VALIDATION] combined config requires both 'within_node' and 'across_node' sub-configurations")
                    has_errors = True

        if has_errors:
            if mpi_rank == 0:
                log.error("[VALIDATION] Configuration validation failed - please check configuration")
            return (False, None)

        return (True, buffer_bytes)

    def validate_runtime(self, cfg: DictConfig, mpi_size: int, mpi_rank: int, log):
         
        has_errors = False
        
 
        if torch.xpu.is_available():
            available_devices = torch.xpu.device_count()
        else:
            available_devices = 1   
        
        def validate_basic_config(config_section, mode_name): 
            nonlocal has_errors
            num_gpus = config_section.num_gpus_per_node
            num_nodes = config_section.num_compute_nodes
            
             
            expected_total_ranks = num_nodes * num_gpus
            if expected_total_ranks != mpi_size:
                if mpi_rank == 0:
                    log.error(f"[VALIDATION] {mode_name}: Expected {expected_total_ranks} total ranks but got {mpi_size}")
                has_errors = True
            
   
            if available_devices < num_gpus:
                if mpi_rank == 0:
                    log.error(f"[VALIDATION] {mode_name}: Need {num_gpus} GPUs per node but only {available_devices} available")
                has_errors = True
        
        comm_config = cfg.comm_group
        comm_mode = comm_config.mode
         
        if comm_mode == "within_node":
            validate_basic_config(comm_config.within_node, "Within-node mode")
            
        elif comm_mode == "across_node":
            validate_basic_config(comm_config.across_node, "Across-node mode")
            
        elif comm_mode == "combined":
            validate_basic_config(comm_config.combined.within_node, "Combined mode")
        
         
        
        if has_errors:
            if mpi_rank == 0:
                log.error("[VALIDATION] Runtime validation failed - please check configuration")
            return False
        
        return True