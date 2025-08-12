def calculate_max_ranks_needed(cfg):
    mode_requirements = {}
  
    # Handle new task structure
    tasks_to_run = cfg.order_of_run
    if isinstance(tasks_to_run, str):
        tasks_to_run = [tasks_to_run]
    
    # Check all tasks
    for task_name in tasks_to_run:
        # Get the task configuration
        if hasattr(cfg, task_name):
            task_config = getattr(cfg, task_name)
            comm_mode = task_config.comm_group
            total_ranks = task_config.num_compute_nodes * len(task_config.device_ids_per_node)
            key = f"{task_name}_{comm_mode}"
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