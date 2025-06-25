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

    def validate_collective_config(self, collective_cfg, mode_name, mpi_rank, log):
        """Validate a collective configuration block"""
        has_errors = False
        buffer_bytes = None
        
        # collective.name
        collective = collective_cfg.name
        if collective not in self.spec["collective"]:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] {mode_name}: Invalid collective '{collective}'. Valid: {self.spec['collective']}")
            has_errors = True

        # collective.op
        op = collective_cfg.op
        valid_ops = self.spec["op"].get(collective, [])
        if op not in valid_ops:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] {mode_name}: Invalid op '{op}' for collective '{collective}'. Valid: {valid_ops}")
            has_errors = True

        # collective.scale_up_algorithm
        scale_up_algo = collective_cfg.scale_up_algorithm
        valid_algos = self.spec["algo"].get(collective, [])
        if scale_up_algo not in valid_algos:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] {mode_name}: Invalid scale_up_algorithm '{scale_up_algo}' for collective '{collective}'. Valid: {valid_algos}")
            has_errors = True
            
        # collective.scale_out_algorithm  
        scale_out_algo = collective_cfg.scale_out_algorithm
        if scale_out_algo not in valid_algos:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] {mode_name}: Invalid scale_out_algorithm '{scale_out_algo}' for collective '{collective}'. Valid: {valid_algos}")
            has_errors = True

        # dtype 
        dtype = collective_cfg.payload.dtype
        if dtype not in self.spec["dtype"]:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] {mode_name}: Invalid dtype '{dtype}'. Valid: {self.spec['dtype']}")
            has_errors = True

        # buffer_size
        try:
            buffer_bytes = parse_buffer_size(collective_cfg.payload.buffer_size)
        except ValueError as ve:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] {mode_name}: {str(ve)}")
            has_errors = True
            
        return has_errors, buffer_bytes

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

        # comm_group validation
        comm_group = cfg.comm_group
        comm_mode = comm_group.mode
        valid_modes = ["within_node", "across_node", "combined", "flatview"]
        
        if comm_mode not in valid_modes:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] Invalid comm_mode '{comm_mode}'. Valid: {valid_modes}")
            has_errors = True
        
        # Mode-specific validation with collective validation
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
                
                # Validate collective config
                if hasattr(within_config, 'collective'):
                    collective_errors, buffer_bytes = self.validate_collective_config(
                        within_config.collective, "within_node", mpi_rank, log)
                    has_errors = has_errors or collective_errors
        
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
                
                # Validate collective config
                if hasattr(across_config, 'collective'):
                    collective_errors, buffer_bytes = self.validate_collective_config(
                        across_config.collective, "across_node", mpi_rank, log)
                    has_errors = has_errors or collective_errors
        
        elif comm_mode == "flatview":
            if not hasattr(comm_group, 'flatview'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'flatview' requires 'flatview' configuration")
                has_errors = True
            else:
                flatview_config = comm_group.flatview
                
                # Validate collective config
                if hasattr(flatview_config, 'collective'):
                    collective_errors, buffer_bytes = self.validate_collective_config(
                        flatview_config.collective, "flatview", mpi_rank, log)
                    has_errors = has_errors or collective_errors
        
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
                
                # Validate within_node collective config
                if hasattr(combined_config, 'within_node') and hasattr(combined_config.within_node, 'collective'):
                    collective_errors, within_buffer_bytes = self.validate_collective_config(
                        combined_config.within_node.collective, "combined.within_node", mpi_rank, log)
                    has_errors = has_errors or collective_errors
                    if buffer_bytes is None:
                        buffer_bytes = within_buffer_bytes
                
                # Validate across_node collective config
                if hasattr(combined_config, 'across_node') and hasattr(combined_config.across_node, 'collective'):
                    collective_errors, across_buffer_bytes = self.validate_collective_config(
                        combined_config.across_node.collective, "combined.across_node", mpi_rank, log)
                    has_errors = has_errors or collective_errors

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