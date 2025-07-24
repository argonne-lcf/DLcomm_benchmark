import torch
from mpi4py import MPI
from omegaconf import DictConfig
from dl_comm.timer import timer


def setup_communication_groups(mode_cfg, mpi_rank, log, dist=None, force_mode=None):
 
    
    # With the new structure, force_mode is always required as we pass the specific mode_cfg
    if not force_mode:
        raise ValueError("setup_communication_groups() requires force_mode parameter")
    
    comm_mode = force_mode
    
    my_within_group = None
    my_across_group = None
    flat_group = None
    #device asigned at the begining of main py
    within_group_id = None
    across_group_id = None
    ranks_responsible_for_logging = set([0])  # Rank 0 always responsible for world/flatview

    mpi_size=MPI.COMM_WORLD.Get_size()

    
    # ----------------------------------------------------------------------------
    # WITHIN NODE MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "within_node":
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Setting up communication groups for mode: Within")


        # CONFIG PARSING
        within_config = mode_cfg
        num_gpus_per_node = within_config.num_gpus_per_node
        num_compute_nodes = within_config.num_compute_nodes
        gpu_ids_per_node = within_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Within-node: {num_gpus_per_node} GPUs per node, Device IDs: {gpu_ids_per_node}")
            log.info("[COMM][GROUP CREATION] Within-node groups:")
            log.info("")
        with timer("Group Creation (Within)"):
            my_within_group = None
            within_group_id = None
            
            
            if torch.cuda.is_available():
                available_devices = torch.cuda.device_count()
            elif torch.xpu.is_available():
                available_devices = torch.xpu.device_count()
            else:
                available_devices = 1
            
            
            rank_inside_node = mpi_rank % available_devices

            for node in range(num_compute_nodes):
                group_ranks = []
                group = None
                
                for each_rank in range(mpi_size):
                    rank_node = each_rank // available_devices
                    rank_device_id = each_rank % available_devices
                    
                    if rank_node == node and rank_device_id in gpu_ids_per_node:
                        group_ranks.append(each_rank)

                group = dist.new_group(ranks=group_ranks,use_local_synchronization=True) 
                if group_ranks:
                    responsible_rank = min(group_ranks)
                    ranks_responsible_for_logging.add(responsible_rank)
                if mpi_rank in group_ranks:
                    my_within_group = group
                    within_group_id = node
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Within Group-{node}] Ranks: {group_ranks}, Required GPUs: {gpu_ids_per_node}, Logging: rank {responsible_rank}")
                    
 
                    

 
    

        # Device already allocated globally - no device allocation needed here

        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {num_compute_nodes} within-node groups")
            log.info("")

    # ----------------------------------------------------------------------------
    # ACROSS NODE MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "across_node":
        if mpi_rank == 0:
            log.info("")
            log.info(f"[COMM][CONFIG] Setting up communication groups for mode: Across")
        # CONFIG PARSING
        across_config = mode_cfg
        num_compute_nodes = across_config.num_compute_nodes
        num_gpus_per_node = across_config.num_gpus_per_node
        gpu_ids_per_node = across_config.gpu_ids_per_node
        
        if mpi_rank == 0:

            log.info(f"[COMM][CONFIG] Across-node: {num_compute_nodes} nodes, {num_gpus_per_node} GPUs per node, Device IDs: {gpu_ids_per_node}")

            log.info("[COMM][GROUP CREATION] Across-node groups:")
        with timer("Group Creation (Across)"):
            my_across_group = None
            across_group_id = None
            
             
            if torch.cuda.is_available():
                available_devices = torch.cuda.device_count()
            elif torch.xpu.is_available():
                available_devices = torch.xpu.device_count()
            else:
                available_devices = 1
            
             
            
             
            for gpu_idx, required_gpu_id in enumerate(gpu_ids_per_node):
                group_ranks = []
                for node in range(num_compute_nodes):
                    for rank in range(mpi_size):
                        rank_device_id = rank % available_devices
                        rank_node = rank // available_devices
                        
                        if rank_node == node and rank_device_id == required_gpu_id:
                            group_ranks.append(rank)
                            break
                
                if group_ranks:   
                     
                    responsible_rank = min(group_ranks)
                    ranks_responsible_for_logging.add(responsible_rank)
                    if mpi_rank == 0:
                        log.info(f"[COMM][GROUP CREATION][Across Group-{gpu_idx}] Ranks: {group_ranks}, GPU ID: {required_gpu_id}, Logging: rank {responsible_rank}")
                     
                    group = dist.new_group(ranks=group_ranks,use_local_synchronization=True)
                    if mpi_rank in group_ranks:
                        my_across_group = group
                        across_group_id = gpu_idx


        
 

        # Device already allocated globally 

        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {num_gpus_per_node} across-node groups")
            log.info("")

    # ----------------------------------------------------------------------------
    # FLATVIEW MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "flatview":
        
        # CONFIG PARSING
        flatview_config = mode_cfg
        num_compute_nodes = flatview_config.num_compute_nodes
        num_gpus_per_node = flatview_config.num_gpus_per_node
        gpu_ids_per_node = flatview_config.gpu_ids_per_node
        
        
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Flatview: {num_compute_nodes} nodes, {num_gpus_per_node} GPUs per node, Device IDs: {gpu_ids_per_node}")
            log.info("")
            log.info("[COMM][GROUP CREATION] Flatview groups:")

        with timer("Group Creation (Flatview)"):
            if torch.cuda.is_available():
                available_devices = torch.cuda.device_count()
            elif torch.xpu.is_available():
                available_devices = torch.xpu.device_count()
            else:
                available_devices = 1
            
            group_ranks = []
            for node in range(num_compute_nodes):
                for gpu_idx, required_gpu_id in enumerate(gpu_ids_per_node):
                    for rank in range(mpi_size):
                        rank_device_id = rank % available_devices
                        rank_node = rank // available_devices
                        
                        if rank_node == node and rank_device_id == required_gpu_id:
                            if rank not in group_ranks:
                                group_ranks.append(rank)
            
            
            if group_ranks:
                responsible_rank = min(group_ranks)
                ranks_responsible_for_logging.add(responsible_rank)
                
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Flatview] Ranks: {group_ranks}, Required GPUs: {gpu_ids_per_node}, Logging: rank {responsible_rank}")
                
                flat_group = dist.new_group(ranks=group_ranks, use_local_synchronization=True)
                flat_group_ranks = group_ranks
            else:
                flat_group = None
                flat_group_ranks = []



    return {
        'my_within_group': my_within_group,
        'my_across_group': my_across_group, 
        'flat_group': flat_group,
        'within_group_id': within_group_id,
        'across_group_id': across_group_id,
        'ranks_responsible_for_logging': ranks_responsible_for_logging,
    }