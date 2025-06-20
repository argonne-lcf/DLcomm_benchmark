def check_group_correctness(mpi_rank, cfg, log, x, comm_mode, group_type, stage, enable_correctness, iteration):

    if iteration != 0 or not enable_correctness:
        return
    
    group_id = None
    should_log = False
    
    if comm_mode == "within_node" and group_type == "within":
        node_id = mpi_rank // cfg.collective.comm_group.within_node.num_gpus_per_node
        gpu_idx = mpi_rank % cfg.collective.comm_group.within_node.num_gpus_per_node
        if gpu_idx == 0:  
            group_id = node_id
            should_log = True
            
    elif comm_mode == "across_node" and group_type == "across":
        gpu_idx = mpi_rank % cfg.collective.comm_group.across_node.num_gpus_per_node
        node_id = mpi_rank // cfg.collective.comm_group.across_node.num_gpus_per_node
        if node_id == 0:   
            group_id = gpu_idx
            should_log = True
            
    elif comm_mode == "combined" and group_type == "within":
        node_id = mpi_rank // cfg.collective.comm_group.combined.within_node.num_gpus_per_node
        gpu_idx = mpi_rank % cfg.collective.comm_group.combined.within_node.num_gpus_per_node
        if gpu_idx == 0:  
            group_id = node_id
            should_log = True
            
    elif comm_mode == "combined" and group_type == "across":
        gpu_idx = mpi_rank % cfg.collective.comm_group.combined.within_node.num_gpus_per_node
        current_gpu_id = cfg.collective.comm_group.combined.within_node.gpu_ids_per_node[gpu_idx]
        if current_gpu_id in cfg.collective.comm_group.combined.across_node.gpu_ids_per_node:
            across_group_idx = cfg.collective.comm_group.combined.across_node.gpu_ids_per_node.index(current_gpu_id)
            node_id = mpi_rank // cfg.collective.comm_group.combined.within_node.num_gpus_per_node
            if node_id == 0:   
                group_id = across_group_idx
                should_log = True
                
    elif comm_mode == "flatview" and group_type == "flatview":
        if mpi_rank == 0:
            group_id = "All"
            should_log = True
    
    if should_log and group_id is not None:
        group_label = f"{group_type.title()}-Group-{group_id}"
        log.info(f"[CORRECTNESS {group_label}] {stage.title()}: {x.sum()}")