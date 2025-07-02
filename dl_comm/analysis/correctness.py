from mpi4py import MPI

_before_values = {}

def check_group_correctness(context, x, group_type, phase, tensor_list=None, collective_name=None):
 
    # Get verify_correctness setting based on mode and group type
    cfg = context['cfg']
    comm_mode = cfg.comm_group.mode
    
    # Determine if correctness checking is enabled for this group type
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
    
    # Special handling for broadcast - allow all ranks to log their results
    if tensor_list is not None and isinstance(tensor_list, dict) and 'type' in tensor_list and collective_name == "broadcast":
        # For broadcast, we want to see all ranks, not just designated logging ranks
        if comm_mode == "within_node" and group_type == "within":
            node_id = mpi_rank // cfg.comm_group.within_node.num_gpus_per_node
            group_label = f"{group_type.title()}-Group-{node_id}"
        elif comm_mode == "across_node" and group_type == "across":
            rank_id_per_node = mpi_rank % cfg.comm_group.across_node.num_gpus_per_node
            group_label = f"{group_type.title()}-Group-{rank_id_per_node}"
        elif comm_mode == "combined" and group_type == "within":
            node_id = mpi_rank // cfg.comm_group.combined.within_node.num_gpus_per_node
            group_label = f"{group_type.title()}-Group-{node_id}"
        elif comm_mode == "combined" and group_type == "across":
            rank_id_per_node = mpi_rank % cfg.comm_group.combined.across_node.num_gpus_per_node
            group_label = f"{group_type.title()}-Group-{rank_id_per_node}"
        elif comm_mode == "flatview" and group_type == "flatview":
            group_label = f"{group_type.title()}-Group-All"
        
        # Log broadcast results for all ranks
        if tensor_list['type'] == 'source':
            sent_sum = tensor_list['sent_sum']
            global_rank = tensor_list['global_rank']
            log.output(f"[CORRECTNESS][{group_label}] Broadcast Source Rank {global_rank}: sent tensor sum = {sent_sum}")
        elif tensor_list['type'] == 'receiver':
            received_sum = tensor_list['received_sum']
            global_rank = tensor_list['global_rank']
            source_rank = tensor_list['source_rank']
            log.output(f"[CORRECTNESS][{group_label}] Broadcast Receiver Rank {global_rank}: received from rank {source_rank}, tensor sum = {received_sum}")
        
        return  # Exit early for broadcast special handling
    
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
                
                # Check list-based correctness for collectives that return data structures
                if tensor_list is not None and phase == "after":
                    if isinstance(tensor_list, list):
                        # List validation for AllGather or Gather
                        collective_label = collective_name.title() if collective_name else "List-based collective"
                        log.output(f"[CORRECTNESS][{group_label}] {collective_label} validation:")
                        log.output(f"[CORRECTNESS][{group_label}]   - List size: {len(tensor_list)} tensors")
                        for i, tensor in enumerate(tensor_list):
                            tensor_sum = float(tensor.sum())
                            log.output(f"[CORRECTNESS][{group_label}]   - Tensor[{i}] sum: {tensor_sum}")
                    elif isinstance(tensor_list, dict):
                        if 'type' in tensor_list and tensor_list['type'] == 'source' and 'sent_tensor' in tensor_list:
                            # Broadcast validation - source rank
                            log.output(f"[CORRECTNESS][{group_label}] Broadcast validation (Source):")
                            global_rank = tensor_list['global_rank']
                            group_rank = tensor_list['group_rank']
                            source_rank = tensor_list['source_rank']
                            sent_tensor = tensor_list['sent_tensor']
                            sent_sum = tensor_list['sent_sum']
                            log.output(f"[CORRECTNESS][{group_label}]   - Source Global Rank {global_rank} (Group Rank {group_rank})")
                            log.output(f"[CORRECTNESS][{group_label}]   - Sent tensor: {sent_tensor.flatten()[:10].tolist()}... (showing first 10 elements)")
                            log.output(f"[CORRECTNESS][{group_label}]   - Sent tensor sum: {sent_sum}")
                        elif 'type' in tensor_list and tensor_list['type'] == 'receiver' and 'received_tensor' in tensor_list:
                            # Broadcast validation - receiver rank
                            log.output(f"[CORRECTNESS][{group_label}] Broadcast validation (Receiver):")
                            global_rank = tensor_list['global_rank']
                            group_rank = tensor_list['group_rank']
                            source_rank = tensor_list['source_rank']
                            received_tensor = tensor_list['received_tensor']
                            received_sum = tensor_list['received_sum']
                            log.output(f"[CORRECTNESS][{group_label}]   - Receiver Global Rank {global_rank} (Group Rank {group_rank})")
                            log.output(f"[CORRECTNESS][{group_label}]   - Source was Global Rank {source_rank}")
                            log.output(f"[CORRECTNESS][{group_label}]   - Received tensor: {received_tensor.flatten()[:10].tolist()}... (showing first 10 elements)")
                            log.output(f"[CORRECTNESS][{group_label}]   - Received tensor sum: {received_sum}")
                        elif 'type' in tensor_list and tensor_list['type'] == 'source' and 'scattered_data' in tensor_list:
                            # Scatter validation - source rank
                            log.output(f"[CORRECTNESS][{group_label}] Scatter validation (Source):")
                            source_rank = tensor_list['source_global_rank']
                            scattered_data = tensor_list['scattered_data']
                            log.output(f"[CORRECTNESS][{group_label}]   - Source Global Rank: {source_rank}")
                            log.output(f"[CORRECTNESS][{group_label}]   - Scattered data:")
                            for data in scattered_data:
                                to_rank = data['to_group_rank']
                                value = data['value']
                                tensor_sum = data['tensor_sum']
                                log.output(f"[CORRECTNESS][{group_label}]     → To Group Rank {to_rank}: tensor filled with {value}, sum = {tensor_sum}")
                        elif 'type' in tensor_list and tensor_list['type'] == 'receiver' and 'receiver_global_rank' in tensor_list:
                            # Scatter validation - receiver rank
                            log.output(f"[CORRECTNESS][{group_label}] Scatter validation (Receiver):")
                            global_rank = tensor_list['receiver_global_rank']
                            group_rank = tensor_list['receiver_group_rank']
                            expected = tensor_list['expected_value']
                            received_sum = tensor_list['received_tensor_sum']
                            is_correct = tensor_list['is_correct']
                            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                            log.output(f"[CORRECTNESS][{group_label}]   - Global Rank {global_rank} (Group Rank {group_rank})")
                            log.output(f"[CORRECTNESS][{group_label}]   - Expected: tensor filled with {expected}")
                            log.output(f"[CORRECTNESS][{group_label}]   - Received sum: {received_sum} {status}")
                        elif 'sent_data' in tensor_list and 'received_data' in tensor_list:
                            # AllToAll validation
                            log.output(f"[CORRECTNESS][{group_label}] AllToAll validation:")
                            global_rank = tensor_list['global_rank']
                            group_rank = tensor_list['group_rank']
                            sent_data = tensor_list['sent_data']
                            received_data = tensor_list['received_data']
                            
                            log.output(f"[CORRECTNESS][{group_label}]   - Global Rank {global_rank} (Group Rank {group_rank})")
                            log.output(f"[CORRECTNESS][{group_label}]   - Sent data:")
                            for data in sent_data:
                                to_rank = data['to_group_rank']
                                sent_val = data['sent_value']
                                chunk_sum = data['chunk_sum']
                                log.output(f"[CORRECTNESS][{group_label}]     → To Group Rank {to_rank}: value {sent_val}, sum = {chunk_sum}")
                            
                            log.output(f"[CORRECTNESS][{group_label}]   - Received data:")
                            all_correct = True
                            for data in received_data:
                                from_rank = data['from_group_rank']
                                expected_val = data['expected_value']
                                expected_sum = data['expected_sum']
                                received_sum = data['received_sum']
                                is_correct = data['is_correct']
                                status = "✓" if is_correct else "✗"
                                if not is_correct:
                                    all_correct = False
                                log.output(f"[CORRECTNESS][{group_label}]     ← From Group Rank {from_rank}: expected {expected_val} (sum={expected_sum}), got sum={received_sum} {status}")
                            
                            overall_status = "✓ ALL CORRECT" if all_correct else "✗ SOME INCORRECT"
                            log.output(f"[CORRECTNESS][{group_label}]   - Overall: {overall_status}")
                        else:
                            # ReduceScatter validation
                            log.output(f"[CORRECTNESS][{group_label}] ReduceScatter validation:")
                            global_rank = tensor_list['global_rank']
                            group_rank = tensor_list['group_rank']
                            my_chunk = tensor_list['my_chunk_index']
                            expected_value = tensor_list['expected_value']
                            log.output(f"[CORRECTNESS][{group_label}]   - Global Rank {global_rank} (Group Rank {group_rank}) contributed: ones to all chunks")
                            log.output(f"[CORRECTNESS][{group_label}]   - Global Rank {global_rank} received chunk index: {my_chunk}")
                            log.output(f"[CORRECTNESS][{group_label}]   - Expected: each element = {expected_value} (sum of {int(expected_value)} ones)")
                            log.output(f"[CORRECTNESS][{group_label}]   - Actual result sum: {after_value}")
                
                del _before_values[group_label]
                
