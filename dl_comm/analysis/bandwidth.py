import re 
from dl_comm.config import parse_buffer_size
from dl_comm.timer.timer import TIMES

def gather_and_print_all_bandwidths(logger, cfg, mpi_size, ranks_responsible_for_logging, title="[BANDWIDTH]", adjusted_buffer_sizes=None, current_comm_mode=None, current_mode_cfg=None, collective_name=None):
    from mpi4py import MPI
    
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    
    my_data = None
    if mpi_rank in ranks_responsible_for_logging:
        my_data = {
            'rank': mpi_rank,
            'timers': dict(TIMES)
        }
    
    all_data = MPI.COMM_WORLD.gather(my_data, root=0)
    
    if mpi_rank == 0:
        logger.output("")
        logger.output(f"{title} -------------------------------------------")
        
        buffer_configs = {}
        if adjusted_buffer_sizes:
            buffer_configs = adjusted_buffer_sizes
            comm_mode = current_comm_mode
        else:
            logger.warning("[BANDWIDTH] Cannot calculate bandwidth - no buffer size information provided")
            return
        
        group_bandwidths = {}
        
        for data in all_data:
            if data is not None:
                rank = data['rank']
                timers = data['timers']
                
                for label, times_list in timers.items():
                    # Skip non-collective timers
                    if any(keyword in label for keyword in ["import", "setup", "mxm", "Group Creation"]):
                        continue
                        
                    # Extract group info from label
                    group_type = None
                    if "Flatview" in label:
                        group_type = "flatview"
                        buffer_size = buffer_configs.get('flatview', 0)
                    elif "Within-Group" in label:
                        group_type = "within"
                        buffer_size = buffer_configs.get('within', 0)
                    elif "Across-Group" in label:
                        group_type = "across"
                        buffer_size = buffer_configs.get('across', 0)
                    else:
                        continue
                    
                    if group_type not in group_bandwidths:
                        group_bandwidths[group_type] = {}
                    
                    if label not in group_bandwidths[group_type]:
                        group_bandwidths[group_type][label] = []
                    
                    # Calculate bandwidth for each iteration
                    for time_seconds in times_list:
                        if time_seconds > 0:
                            # Get group size from label or use mpi_size
                            if "Group" in label:
                                # Extract group size from current_mode_cfg if available
                                if current_mode_cfg and hasattr(current_mode_cfg, 'num_devices_per_node'):
                                    group_size = current_mode_cfg.num_devices_per_node
                                else:
                                    group_size = mpi_size
                            else:
                                group_size = mpi_size
                            
                            bandwidth_gbps = calculate_group_bandwidth(group_size, buffer_size, time_seconds)
                            group_bandwidths[group_type][label].append(bandwidth_gbps)
        
        # Print bandwidth table in iteration table format (like timer table)
        if group_bandwidths:
            logger.output("")
            if collective_name:
                logger.output(f"[BANDWIDTH] BANDWIDTH TABLE FOR {collective_name.upper()} (bytes/seconds):")
            else:
                logger.output("[BANDWIDTH] BANDWIDTH TABLE (bytes/seconds):")
            
            # Collect all labels and their bandwidth data
            iteration_data = {}
            for group_type, group_data in group_bandwidths.items():
                for label, bandwidths in group_data.items():
                    if bandwidths:
                        # Find the actual rank that has data for this label
                        actual_rank = None
                        for data in all_data:
                            if data is not None and label in data['timers']:
                                actual_rank = data['rank']
                                break
                        if actual_rank is None:
                            actual_rank = mpi_rank
                        iteration_data[label] = {'vals': bandwidths, 'rank': actual_rank}
            
            if iteration_data:
                headers = list(iteration_data.keys())
                max_iterations = max(len(data['vals']) for data in iteration_data.values())
                col_width = 20
                
                # Header line 1 - Labels
                header_line1 = f"{'Iteration':<12}"
                for label in headers:
                    header_line1 += f"{label:^{col_width}}"
                logger.output(header_line1)
                
                # Header line 2 - Ranks
                header_line2 = f"{'':12}"
                for label in headers:
                    rank = iteration_data[label]['rank']
                    rank_str = f"LOGGING RANK - {rank}"
                    header_line2 += f"{rank_str:^{col_width}}"
                logger.output(header_line2)
                
                # Separator
                separator = "-" * len(header_line1)
                logger.output(separator)
                
                # Data rows
                for i in range(max_iterations):
                    row = f"{i:<12}"
                    for label in headers:
                        vals = iteration_data[label]['vals']
                        if i < len(vals):
                            row += f"{vals[i]:.0f}".center(col_width)
                        else:
                            row += f"{'-':^{col_width}}"
                    logger.output(row)
                
                logger.output(separator)
                logger.output("")
        
        logger.output(f"{title} -------------------------------------------")


def calculate_group_bandwidth(group_size, buffer_size, time_seconds):
    total_bytes = group_size * buffer_size
    bandwidth_bytes_per_sec = total_bytes / time_seconds
    return bandwidth_bytes_per_sec


def print_all_bandwidths(logger, cfg, mpi_size, ranks_responsible_for_logging, phase_filter=None, adjusted_buffer_sizes=None, current_comm_mode=None, current_mode_cfg=None):
    from mpi4py import MPI
    
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    
    my_data = None
    if mpi_rank in ranks_responsible_for_logging:
        my_data = {
            'rank': mpi_rank,
            'timers': dict(TIMES)
        }
    
    all_data = MPI.COMM_WORLD.gather(my_data, root=0)
    
    if mpi_rank == 0:
        logger.output("")
         
        title = "[BANDWIDTH]"
        logger.output(f"{title} -------------------------------------------")
        
        group_bandwidths = {}
        
        buffer_configs = {}
        if adjusted_buffer_sizes:
            # Use the adjusted buffer sizes passed from main
            buffer_configs = adjusted_buffer_sizes
            # Use the current mode passed from main
            comm_mode = current_comm_mode
        else:
            # Fallback to parsing from config (original behavior)
            # Only access comm_group if adjusted_buffer_sizes is not provided
            comm_mode = cfg.comm_group.mode if hasattr(cfg, 'comm_group') else None
            if comm_mode == "flatview":
                coll_cfg = cfg.comm_group.flatview.collective
                buffer_in_bytes = parse_buffer_size(coll_cfg.payload.buffer_size)
                buffer_configs['flatview'] = buffer_in_bytes
            elif comm_mode == "within_node":
                coll_cfg = cfg.comm_group.within_node.collective
                buffer_in_bytes = parse_buffer_size(coll_cfg.payload.buffer_size)
                buffer_configs['within'] = buffer_in_bytes
            elif comm_mode == "across_node":
                coll_cfg = cfg.comm_group.across_node.collective
                buffer_in_bytes = parse_buffer_size(coll_cfg.payload.buffer_size)
                buffer_configs['across'] = buffer_in_bytes
            elif comm_mode == "combined":
                coll_within_cfg = cfg.comm_group.combined.within_node.collective
                coll_across_cfg = cfg.comm_group.combined.across_node.collective
                buffer_within_bytes = parse_buffer_size(coll_within_cfg.payload.buffer_size)
                buffer_across_bytes = parse_buffer_size(coll_across_cfg.payload.buffer_size)
                buffer_configs['within'] = buffer_within_bytes
                buffer_configs['across'] = buffer_across_bytes
        
        group_timers = {}
        
        for data in all_data:
            if data is not None:
                rank = data['rank']
                timers = data['timers']
                
                for label, vals in timers.items():
                    if "(flatview)" == label.lower():
                        group_key = "flatview"
                    elif "(within-group-" in label.lower():
                        match = re.search(r'\(within-group-(\d+)\)', label.lower())
                        group_key = f"within-{match.group(1)}"
                    elif "(across-group-" in label.lower():
                        match = re.search(r'\(across-group-(\d+)\)', label.lower())
                        group_key = f"across-{match.group(1)}"
                    else:
                        continue
                    
                    # Apply phase filtering
                    if phase_filter == "within" and not group_key.startswith("within-"):
                        continue
                    elif phase_filter == "across" and not group_key.startswith("across-"):
                        continue
                    
                    if group_key not in group_timers:
                        group_timers[group_key] = {}
                    
                    if label not in group_timers[group_key] or rank < group_timers[group_key][label]['rank']:
                        group_timers[group_key][label] = {
                            'vals': vals,
                            'rank': rank
                        }
        
        for group_key, labels in group_timers.items():
            for label, timer_data in labels.items():
                vals = timer_data['vals']
                rank = timer_data['rank']
                first_iteration_time = vals[0]
                
                if group_key == "flatview":
                    # Calculate actual flatview group size from config
                    if current_mode_cfg and comm_mode == "flatview":
                        group_size = current_mode_cfg.num_devices_per_node * current_mode_cfg.num_compute_nodes
                    else:
                        # Fallback to mpi_size if config not available (shouldn't happen in normal flow)
                        group_size = mpi_size
                    buffer_size = buffer_configs.get('flatview', 0)
                    bandwidth = calculate_group_bandwidth(group_size, buffer_size, first_iteration_time)
                    group_bandwidths['Flatview'] = {
                        'bandwidth': bandwidth,
                        'group_size': group_size,
                        'buffer_size': buffer_size,
                        'time': first_iteration_time,
                        'rank': rank
                    }
                    
                elif group_key.startswith("within-"):
                    group_id = group_key.split("-")[1]
                    if current_mode_cfg and comm_mode == "within_node":
                        within_mode_cfg = current_mode_cfg
                    else:
                        within_mode_cfg = cfg.comm_group.combined.within_node if comm_mode == "combined" else cfg.comm_group.within_node
                    group_size = within_mode_cfg.num_devices_per_node
                    buffer_size = buffer_configs.get('within', 0)
                    bandwidth = calculate_group_bandwidth(group_size, buffer_size, first_iteration_time)
                    group_bandwidths[f'Within-Group-{group_id}'] = {
                        'bandwidth': bandwidth,
                        'group_size': group_size,
                        'buffer_size': buffer_size,
                        'time': first_iteration_time,
                        'rank': rank
                    }
                        
                elif group_key.startswith("across-"):
                    group_id = group_key.split("-")[1]
                    if current_mode_cfg and comm_mode == "across_node":
                        across_mode_cfg = current_mode_cfg
                    else:
                        across_mode_cfg = cfg.comm_group.combined.across_node if comm_mode == "combined" else cfg.comm_group.across_node
                    group_size = across_mode_cfg.num_compute_nodes
                    buffer_size = buffer_configs.get('across', 0)
                    bandwidth = calculate_group_bandwidth(group_size, buffer_size, first_iteration_time)
                    group_bandwidths[f'Across-Group-{group_id}'] = {
                        'bandwidth': bandwidth,
                        'group_size': group_size,
                        'buffer_size': buffer_size,
                        'time': first_iteration_time,
                        'rank': rank
                    }
        
        logger.output(f"{title.replace(' -------------------------------------------', '')} Communication Group Bandwidths:")
        logger.output("")
        
        for group_name, data in group_bandwidths.items():
            bandwidth_prefix = title.replace(' -------------------------------------------', '').replace('[', '').replace(']', '')
            logger.output(f"[{bandwidth_prefix}] {group_name}:")
            logger.output(f"[{bandwidth_prefix}]   Group Size     : {data['group_size']} GPUs")
            logger.output(f"[{bandwidth_prefix}]   Buffer Size    : {data['buffer_size']} bytes")
            logger.output(f"[{bandwidth_prefix}]   Time (iter 0)  : {data['time']:.6f} s")
            logger.output(f"[{bandwidth_prefix}]   Bandwidth      : {data['bandwidth']:.0f} bytes/s")
            logger.output(f"[{bandwidth_prefix}]   Logging Rank   : {data['rank']}")
            logger.output("")
        
        logger.output(f"{title} -------------------------------------------")
        logger.output("")










