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
    device = None  # Will be assigned based on group membership
    within_group_id = None
    across_group_id = None
    ranks_responsible_for_logging = set([0])  # Rank 0 always responsible for world/flatview
    participating = False  # Flag to indicate if this rank participates in collectives

    mpi_size=MPI.COMM_WORLD.Get_size()
    
    # Calculate available devices once at the beginning
    if torch.cuda.is_available():
        available_devices = torch.cuda.device_count()
    elif torch.xpu.is_available():
        available_devices = torch.xpu.device_count()
    else:
        available_devices = 1

    
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
            
            
           
            rank_inside_node = mpi_rank % available_devices

            for node in range(num_compute_nodes):
                group_ranks = []
                group = None
                
                # Use actual ranks per physical node from MPI configuration
                ranks_per_physical_node = mpi_size // num_compute_nodes
                for gpu_idx, gpu_id in enumerate(gpu_ids_per_node):
                    rank = node * ranks_per_physical_node + gpu_idx
                    if rank < mpi_size:  # Ensure rank exists
                        group_ranks.append(rank)
 

                group = dist.new_group(ranks=group_ranks,use_local_synchronization=True) 
                if group_ranks:
                    responsible_rank = min(group_ranks)
                    ranks_responsible_for_logging.add(responsible_rank)
                if mpi_rank in group_ranks:
                    my_within_group = group
                    within_group_id = node
                    participating = True  # This rank participates in within_node collectives
                    
                    # Assign device based on position in gpu_ids list
                    gpu_idx_in_group = group_ranks.index(mpi_rank)
                    assigned_gpu_id = gpu_ids_per_node[gpu_idx_in_group]
                    
                    # Set device based on backend
                  
                    if torch.cuda.is_available():
                        device = torch.device(f"cuda:{assigned_gpu_id}")
                        torch.cuda.set_device(assigned_gpu_id)
                    elif torch.xpu.is_available():
                        device = torch.device(f"xpu:{assigned_gpu_id}")
                    else:
                        device = torch.device('cpu')
                        
 
                        
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
            
             
             
            
             
            for gpu_idx, required_gpu_id in enumerate(gpu_ids_per_node):
                group_ranks = []
                # Use actual ranks per physical node from MPI configuration
                ranks_per_physical_node = mpi_size // num_compute_nodes
                for node in range(num_compute_nodes):
                    rank = node * ranks_per_physical_node + gpu_idx 
                    group_ranks.append(rank)
                
                if group_ranks:   
                     
                    responsible_rank = min(group_ranks)
                    ranks_responsible_for_logging.add(responsible_rank)
                    if mpi_rank == 0:
                        log.info(f"[COMM][GROUP CREATION][Across Group-{gpu_idx}] Ranks: {group_ranks}, GPU ID: {required_gpu_id}, Logging: rank {responsible_rank}")
                     
                    group = dist.new_group(ranks=group_ranks,use_local_synchronization=True)
                    if mpi_rank in group_ranks:
                        my_across_group = group
                        across_group_id = gpu_idx
                        
                        # Assign device based on gpu_id for this group
                        assigned_gpu_id = required_gpu_id
                        
                        # Set device based on backend
                        if torch.cuda.is_available():
                            device = torch.device(f"cuda:{assigned_gpu_id}")
                            torch.cuda.set_device(assigned_gpu_id)
                        elif torch.xpu.is_available():
                            device = torch.device(f"xpu:{assigned_gpu_id}")
                        else:
                            device = torch.device('cpu')
 


        
 

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
            # Use actual ranks per physical node from MPI configuration  
            ranks_per_physical_node = mpi_size // num_compute_nodes
            for node in range(num_compute_nodes):
                for gpu_idx, required_gpu_id in enumerate(gpu_ids_per_node):
                    # Sequential rank assignment based on gpu_ids order
                    rank = node * ranks_per_physical_node + gpu_idx
                    if rank < mpi_size and rank not in group_ranks:
                        group_ranks.append(rank)
            
            
            if group_ranks:
                responsible_rank = min(group_ranks)
                ranks_responsible_for_logging.add(responsible_rank)
                
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Flatview] Ranks: {group_ranks}, Required GPUs: {gpu_ids_per_node}, Logging: rank {responsible_rank}")
                
                flat_group = dist.new_group(ranks=group_ranks, use_local_synchronization=True)
                flat_group_ranks = group_ranks
                
                # Assign device if this rank is in the flatview group
                if mpi_rank in group_ranks:
                    # Calculate which GPU this rank should use
                    rank_idx_in_group = group_ranks.index(mpi_rank)
                    node_id = rank_idx_in_group // len(gpu_ids_per_node)
                    gpu_idx_in_node = rank_idx_in_group % len(gpu_ids_per_node)
                    assigned_gpu_id = gpu_ids_per_node[gpu_idx_in_node]
                    
                    # Set device based on backend
                    if torch.cuda.is_available():
                        device = torch.device(f"cuda:{assigned_gpu_id}")
                        torch.cuda.set_device(assigned_gpu_id)
                    elif torch.xpu.is_available():
                        device = torch.device(f"xpu:{assigned_gpu_id}")
                    else:
                        device = torch.device('cpu')
                        
 
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
        'device': device,
        'participating': participating,
    }