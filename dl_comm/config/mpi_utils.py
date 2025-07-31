def calculate_max_ranks_needed(cfg):
    mode_requirements = {}
  
    # Handle new implementations structure
    implementations_to_run = cfg.order_of_run
    if isinstance(implementations_to_run, str):
        implementations_to_run = [implementations_to_run]
    
    # Check all implementations and their modes
    for impl_name in implementations_to_run:
        # Find the implementation configuration
        implementation_config = None
        for impl_config in cfg.implementations:
            if hasattr(impl_config, 'name') and impl_config.name == impl_name:
                implementation_config = impl_config
                break
        
        if implementation_config:
            comm_groups = implementation_config.comm_groups
            # Check all available modes in this implementation
            for mode_name in ['within_node', 'across_node', 'flatview']:
                if hasattr(comm_groups, mode_name):
                    mode_config = getattr(comm_groups, mode_name)
                    total_ranks = mode_config.num_compute_nodes * len(mode_config.gpu_ids_per_node)
                    key = f"{impl_name}_{mode_name}"
                    mode_requirements[key] = total_ranks
    
    if not mode_requirements:
        return 1, None, {}
    
    max_ranks = max(mode_requirements.values())
    max_mode = max(mode_requirements, key=mode_requirements.get)
    
    return max_ranks, max_mode, mode_requirements


def validate_mpi_configuration(cfg, mpi_size, mpi_rank, log):
    max_ranks, max_mode, requirements = calculate_max_ranks_needed(cfg)
    has_errors = False
    
    if mpi_size < max_ranks:
        has_errors = True
    
    return mpi_size, has_errors 