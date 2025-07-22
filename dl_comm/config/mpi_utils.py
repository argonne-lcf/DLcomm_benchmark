def calculate_max_ranks_needed(cfg):
    mode_requirements = {}
  
    active_modes = cfg.comm_group.mode
    if isinstance(active_modes, str):
        active_modes = [active_modes]
    
 
    for mode_name in active_modes:
        if hasattr(cfg.comm_group, mode_name):
            mode_config = getattr(cfg.comm_group, mode_name)
            total_ranks = mode_config.num_compute_nodes * len(mode_config.gpu_ids_per_node)
            mode_requirements[mode_name] = total_ranks
    
    if not mode_requirements:
        return 1, None, {}
    
    max_ranks = max(mode_requirements.values())
    max_mode = max(mode_requirements, key=mode_requirements.get)
    
    return max_ranks, max_mode, mode_requirements


def validate_mpi_configuration(cfg, mpi_size, mpi_rank, log):
    max_ranks, max_mode, requirements = calculate_max_ranks_needed(cfg)
    
    if mpi_size < max_ranks:
        error_msg = f"MPI world size ({mpi_size}) insufficient for mode '{max_mode}' requiring {max_ranks} ranks. Mode requirements: {requirements}"
        if mpi_rank == 0:
            log.error(f"[MPI][VALIDATION] {error_msg}")
        raise ValueError(error_msg)
    
    if mpi_rank == 0:
        log.info(f"[MPI][VALIDATION] World size: {mpi_size}, Max required: {max_ranks} (mode: {max_mode})")
        log.info(f"[MPI][VALIDATION] Mode requirements: {requirements}")
    
    return max_ranks 