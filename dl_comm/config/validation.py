import torch
from omegaconf import DictConfig

# CCL Algorithm validation based on Intel oneCCL documentation
CCL_SCALE_UP_ALGORITHMS = {
    "allgather": ["topo", "direct", "naive", "flat", "multi_bcast", "ring"],

    "allreduce": ["topo", "direct", "rabenseifner", "nreduce", "ring", "double_tree", "recursive_doubling", "2d"],
    "alltoall": ["topo", "direct", "naive", "scatter"],

    "barrier": ["direct", "ring"],
    "broadcast": ["topo", "direct", "ring", "double_tree", "naive"],

    "reduce": ["topo", "direct", "rabenseifner", "tree", "double_tree"],

    "reducescatter": ["topo", "direct", "naive", "ring"],
    "scatter": ["topo", "direct", "naive", "ring"],
    "gather": ["topo", "direct", "naive", "ring"]
}

CCL_SCALE_OUT_ALGORITHMS = {
    "allgather": ["topo","direct", "naive", "flat", "multi_bcast", "ring"],

    "allreduce": ["direct", "rabenseifner", "nreduce", "ring", "double_tree", "topo"],
    "alltoall": ["naive", "scatter","topo"],

    "reduce": ["direct", "rabenseifner", "tree", "double_tree","topo"],

    "reducescatter": ["direct", "naive", "ring","topo"],
    "scatter": ["direct", "naive", "ring","topo"],
    "gather": ["direct", "naive", "ring","topo"]
}

COLLECTIVES_WITH_SCALEOUT = [
    "allgather", "allreduce", "alltoall", 
    "reduce",  "reducescatter", "scatter", "gather"
]


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


def validate_ccl_algorithms(collective_config, mode_name, mpi_rank, log):
    has_errors = False
    
    collective_name = collective_config.name.lower()
    scale_up_algo = collective_config.scale_up_algorithm
    scale_out_algo = collective_config.scale_out_algorithm
    
    # Validate collective name
    if collective_name not in CCL_SCALE_UP_ALGORITHMS:
        if mpi_rank == 0:
            valid_collectives = list(CCL_SCALE_UP_ALGORITHMS.keys())
            log.error(f"[VALIDATION] {mode_name}: Unknown collective '{collective_name}'. Valid: {valid_collectives}")
        has_errors = True
        return has_errors
    
    # Validate scale-up algorithm
    valid_scale_up = CCL_SCALE_UP_ALGORITHMS[collective_name]
    if scale_up_algo not in valid_scale_up:
        if mpi_rank == 0:
            log.error(f"[VALIDATION] {mode_name}: Invalid scale_up_algorithm '{scale_up_algo}' for {collective_name}")
            log.error(f"[VALIDATION] {mode_name}: Valid scale-up algorithms for {collective_name}: {valid_scale_up}")
        has_errors = True
    
    # Validate scale-out algorithm (only if collective supports it)
    if collective_name in COLLECTIVES_WITH_SCALEOUT:
        valid_scale_out = CCL_SCALE_OUT_ALGORITHMS[collective_name]
        if scale_out_algo not in valid_scale_out:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] {mode_name}: Invalid scale_out_algorithm '{scale_out_algo}' for {collective_name}")
                log.error(f"[VALIDATION] {mode_name}: Valid scale-out algorithms for {collective_name}: {valid_scale_out}")
            has_errors = True
    else:
        # Some collectives don't support scale-out, warn if specified
        if mpi_rank == 0:
            log.info(f"[VALIDATION] {mode_name}: Note: {collective_name} does not use scale-out algorithms")
    
    return has_errors


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

        # Buffer size validation - extract from active communication mode
        buffer_bytes = None

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
                
                # Validate CCL algorithms for within_node
                if hasattr(within_config, 'collective'):
                    if validate_ccl_algorithms(within_config.collective, "Within-node", mpi_rank, log):
                        has_errors = True
                    
                    # Validate buffer size
                    if hasattr(within_config.collective, 'payload') and hasattr(within_config.collective.payload, 'buffer_size'):
                        try:
                            buffer_bytes = parse_buffer_size(within_config.collective.payload.buffer_size)
                        except ValueError as e:
                            if mpi_rank == 0:
                                log.error(f"[VALIDATION] Within-node: {e}")
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
                
                # Validate CCL algorithms for across_node
                if hasattr(across_config, 'collective'):
                    if validate_ccl_algorithms(across_config.collective, "Across-node", mpi_rank, log):
                        has_errors = True
                    
                    # Validate buffer size
                    if hasattr(across_config.collective, 'payload') and hasattr(across_config.collective.payload, 'buffer_size'):
                        try:
                            buffer_bytes = parse_buffer_size(across_config.collective.payload.buffer_size)
                        except ValueError as e:
                            if mpi_rank == 0:
                                log.error(f"[VALIDATION] Across-node: {e}")
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
                
                # Validate CCL algorithms for combined mode
                buffer_within_bytes = None
                buffer_across_bytes = None
                
                if hasattr(combined_config, 'within_node') and hasattr(combined_config.within_node, 'collective'):
                    if validate_ccl_algorithms(combined_config.within_node.collective, "Combined within-node", mpi_rank, log):
                        has_errors = True
                    
                    # Validate within-node buffer size
                    if hasattr(combined_config.within_node.collective, 'payload') and hasattr(combined_config.within_node.collective.payload, 'buffer_size'):
                        try:
                            buffer_within_bytes = parse_buffer_size(combined_config.within_node.collective.payload.buffer_size)
                        except ValueError as e:
                            if mpi_rank == 0:
                                log.error(f"[VALIDATION] Combined within-node: {e}")
                            has_errors = True
                
                if hasattr(combined_config, 'across_node') and hasattr(combined_config.across_node, 'collective'):
                    if validate_ccl_algorithms(combined_config.across_node.collective, "Combined across-node", mpi_rank, log):
                        has_errors = True
                    
                    # Validate across-node buffer size
                    if hasattr(combined_config.across_node.collective, 'payload') and hasattr(combined_config.across_node.collective.payload, 'buffer_size'):
                        try:
                            buffer_across_bytes = parse_buffer_size(combined_config.across_node.collective.payload.buffer_size)
                        except ValueError as e:
                            if mpi_rank == 0:
                                log.error(f"[VALIDATION] Combined across-node: {e}")
                            has_errors = True
                
                # For combined mode, use within_node buffer size as primary (main app re-parses both anyway)
                if buffer_within_bytes is not None:
                    buffer_bytes = buffer_within_bytes
                elif buffer_across_bytes is not None:
                    buffer_bytes = buffer_across_bytes
        
        elif comm_mode == "flatview":
            if not hasattr(comm_group, 'flatview'):
                if mpi_rank == 0:
                    log.error("[VALIDATION] comm_mode 'flatview' requires 'flatview' configuration")
                has_errors = True
            else:
                flatview_config = comm_group.flatview
                # Validate CCL algorithms for flatview
                if hasattr(flatview_config, 'collective'):
                    if validate_ccl_algorithms(flatview_config.collective, "Flatview", mpi_rank, log):
                        has_errors = True
                    
                    # Validate buffer size
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