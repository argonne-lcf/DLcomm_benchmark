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
    """Check that the job was launched with enough ranks for every task.

    Historically this function computed ``has_errors`` and the caller ignored
    the result, so an undersized launch proceeded silently. The caller now
    aborts on a False verdict; see docs/fixes/03-rank-topology-validation.md.
    """
    max_ranks, max_mode, requirements = calculate_max_ranks_needed(cfg)
    has_errors = False

    if mpi_size < max_ranks:
        has_errors = True
        if mpi_rank == 0:
            log.error(f"[VALIDATION] Job has {mpi_size} ranks but task '{max_mode}' "
                      f"requires {max_ranks}")
            for key, needed in sorted(requirements.items()):
                marker = "  <-- insufficient" if needed > mpi_size else ""
                log.error(f"[VALIDATION]   {key}: needs {needed} ranks{marker}")

    # An oversized launch is equally wrong: the surplus ranks join no group and
    # contribute nothing, while still appearing in MPI_COMM_WORLD collectives.
    for key, needed in sorted(requirements.items()):
        if mpi_size > needed:
            if mpi_rank == 0:
                log.error(f"[VALIDATION] Job has {mpi_size} ranks but task '{key}' "
                          f"describes only {needed}; {mpi_size - needed} rank(s) "
                          f"would be idle and unreported")
            has_errors = True

    return mpi_size, has_errors 