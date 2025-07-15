import re 
from dl_comm.config import parse_buffer_size
from dl_comm.timer.timer import TIMES


def calculate_group_bandwidth(group_size, buffer_size, time_seconds):
    total_bytes = group_size * buffer_size
    bandwidth_bytes_per_sec = total_bytes / time_seconds
    return bandwidth_bytes_per_sec


def print_all_bandwidths(logger, cfg, mpi_size, ranks_responsible_for_logging, phase_filter=None, adjusted_buffer_sizes=None):
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
        
        comm_mode = cfg.comm_group.mode
        group_bandwidths = {}
        
        buffer_configs = {}
        if adjusted_buffer_sizes:
            # Use the adjusted buffer sizes passed from main
            buffer_configs = adjusted_buffer_sizes
        else:
            # Fallback to parsing from config (original behavior)
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
                    world_size = mpi_size
                    buffer_size = buffer_configs.get('flatview', 0)
                    bandwidth = calculate_group_bandwidth(world_size, buffer_size, first_iteration_time)
                    group_bandwidths['Flatview'] = {
                        'bandwidth': bandwidth,
                        'group_size': world_size,
                        'buffer_size': buffer_size,
                        'time': first_iteration_time,
                        'rank': rank
                    }
                    
                elif group_key.startswith("within-"):
                    group_id = group_key.split("-")[1]
                    within_mode_cfg = cfg.comm_group.combined.within_node if comm_mode == "combined" else cfg.comm_group.within_node
                    group_size = within_mode_cfg.num_gpus_per_node
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










