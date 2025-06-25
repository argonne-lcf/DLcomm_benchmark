 
_before_values = {}

def check_group_correctness(context, x, group_type, phase):
 
    # Check verify_correctness based on comm_mode and group_type
    cfg = context['cfg']
    comm_mode = cfg.comm_group.mode
    
    verify_correctness = False
    if comm_mode == "flatview" and group_type == "flatview":
        verify_correctness = cfg.comm_group.flatview.verify_correctness
    elif comm_mode == "within_node" and group_type == "within":
        verify_correctness = cfg.comm_group.within_node.verify_correctness
    elif comm_mode == "across_node" and group_type == "across":
        verify_correctness = cfg.comm_group.across_node.verify_correctness
    elif comm_mode == "combined":
        if group_type == "within":
            verify_correctness = cfg.comm_group.combined.within_node.verify_correctness
        elif group_type == "across":
            verify_correctness = cfg.comm_group.combined.across_node.verify_correctness
    
    if context['iteration'] != 0 or not verify_correctness:
        return
    
  
    mpi_rank = context['mpi_rank']
    log = context['log']
    
    group_rank_id = None
    should_log = False
    
    if comm_mode == "within_node" and group_type == "within":
        node_id = mpi_rank // cfg.comm_group.within_node.num_gpus_per_node
        rank_id_per_node = mpi_rank % cfg.comm_group.within_node.num_gpus_per_node
        if rank_id_per_node == 0:  
            group_rank_id = node_id
            should_log = True
            
    elif comm_mode == "across_node" and group_type == "across":
        rank_id_per_node = mpi_rank % cfg.comm_group.across_node.num_gpus_per_node
        node_id = mpi_rank // cfg.comm_group.across_node.num_gpus_per_node
        if node_id == 0:   
            group_rank_id = rank_id_per_node
            should_log = True
            
    elif comm_mode == "combined" and group_type == "within":
        node_id = mpi_rank // cfg.comm_group.combined.within_node.num_gpus_per_node
        rank_id_per_node = mpi_rank % cfg.comm_group.combined.within_node.num_gpus_per_node
        if rank_id_per_node == 0:  
            group_rank_id = node_id
            should_log = True
            
    elif comm_mode == "combined" and group_type == "across":
        rank_id_per_node = mpi_rank % cfg.comm_group.combined.across_node.num_gpus_per_node
        node_id = mpi_rank // cfg.comm_group.combined.across_node.num_gpus_per_node
        if node_id == 0:   
            group_rank_id = rank_id_per_node
            should_log = True
                
    elif comm_mode == "flatview" and group_type == "flatview":
        if mpi_rank == 0:
            group_rank_id = "All"
            should_log = True
    
    if should_log and group_rank_id is not None:
        group_label = f"{group_type.title()}-Group-{group_rank_id}"
        tensor_sum = float(x.sum())
        
        if phase == "before":
             
            _before_values[group_label] = tensor_sum
            
        elif phase == "after":
             
            if group_label in _before_values:
                before_value = _before_values[group_label]
                after_value = tensor_sum
                
                log.output(f"[CORRECTNESS][{group_label}] Tensor sum before collective: {before_value} → after collective: {after_value}")
                
                 
                del _before_values[group_label]