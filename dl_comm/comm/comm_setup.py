import torch
from mpi4py import MPI
from omegaconf import DictConfig
from dl_comm.timer import timer


def setup_communication_groups(cfg: DictConfig, mpi_rank, log, dist=None, force_mode=None):
 
    
    comm_config = cfg.comm_group
    if force_mode:
        comm_mode = force_mode
    else:
        # Handle both single mode and list of modes - take first valid mode for setup
        raw_mode = comm_config.mode
        if isinstance(raw_mode, (list, tuple)) or (hasattr(raw_mode, '__iter__') and not isinstance(raw_mode, str)):
            # For list of modes, we need the current mode from the main loop
            # This function should be called with force_mode when in multi-mode
            raise ValueError("setup_communication_groups() requires force_mode when using multi-mode configuration")
        else:
            comm_mode = raw_mode
    
    my_within_group = None
    my_across_group = None
    world_group = None
    device = None
    within_group_id = None
    across_group_id = None
    within_group_ranks = None
    across_group_ranks = None
    world_group_ranks = None
    ranks_responsible_for_logging = set([0])  # Rank 0 always responsible for world/flatview


    
    # ----------------------------------------------------------------------------
    # WITHIN NODE MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "within_node":
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Setting up communication groups for mode: Within")


        # CONFIG PARSING
        within_config = comm_config.within_node
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
            
            for node in range(num_compute_nodes):
                group_ranks = []
                for gpu in range(num_gpus_per_node):
                    rank = node * num_gpus_per_node + gpu
                    group_ranks.append(rank)
                
                # First rank in each within group is responsible for logging
                responsible_rank = min(group_ranks)
                ranks_responsible_for_logging.add(responsible_rank)
                
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Within Group-{node}] Ranks: {group_ranks}, Logging: rank {responsible_rank}")
                
                # Only create group if current rank belongs to it
                group = dist.new_group(ranks=group_ranks,use_local_synchronization=True)
                if mpi_rank in group_ranks:
                    my_within_group = group
                    within_group_id = node
 
 
        
        # Calculate the ranks for this rank's within-group
        within_group_ranks = []
        if within_group_id is not None:
            for gpu in range(num_gpus_per_node):
                rank = within_group_id * num_gpus_per_node + gpu
                within_group_ranks.append(rank)

        # DEVICE ALLOCATION - WITHIN NODE
        mpi_size_of_comm_group = num_compute_nodes * num_gpus_per_node
        if mpi_rank < mpi_size_of_comm_group:
            rank_id_per_node = mpi_rank % num_gpus_per_node
            if cfg.ccl_backend == "nccl" and torch.cuda.is_available():
                device_id = gpu_ids_per_node[rank_id_per_node]
                device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device_id)
               
            elif cfg.ccl_backend in ["ccl", "xccl"] and torch.xpu.is_available():
                device_id = gpu_ids_per_node[rank_id_per_node]
                device = torch.device(f"xpu:{device_id}")
            else:
                device = torch.device('cpu')
                if mpi_rank == 0:
                    log.info("[COMM] Using CPU device")
        else:
            # Excluded ranks use CPU device
            device = torch.device('cpu')
            log.info(f"[COMM] Rank {mpi_rank} excluded from within_node groups, using CPU device")

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
        across_config = comm_config.across_node
        num_compute_nodes = across_config.num_compute_nodes
        num_gpus_per_node = across_config.num_gpus_per_node
        gpu_ids_per_node = across_config.gpu_ids_per_node
        
        if mpi_rank == 0:

            log.info(f"[COMM][CONFIG] Across-node: {num_compute_nodes} nodes, {num_gpus_per_node} GPUs per node, Device IDs: {gpu_ids_per_node}")

            log.info("[COMM][GROUP CREATION] Across-node groups:")
        with timer("Group Creation (Across)"):
            my_across_group = None
            across_group_id = None
            
            for i in range(num_gpus_per_node):
                group_ranks = []
                for node in range(num_compute_nodes):
                    rank = node * num_gpus_per_node + i
                    group_ranks.append(rank)
                
                # First rank in each across group is responsible for logging
                responsible_rank = min(group_ranks)
                ranks_responsible_for_logging.add(responsible_rank)
                
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Across Group-{i}] Ranks: {group_ranks}, Logging: rank {responsible_rank}")
                
                # Only create group if current rank belongs to it
                group = dist.new_group(ranks=group_ranks,use_local_synchronization=True)
                if mpi_rank in group_ranks:
                    my_across_group = group
                    across_group_id = i


        
        # Calculate the ranks for this rank's across-group
        across_group_ranks = []
        if across_group_id is not None:
            for node in range(num_compute_nodes):
                rank = node * num_gpus_per_node + across_group_id
                across_group_ranks.append(rank)

        # DEVICE ALLOCATION - ACROSS NODE
        mpi_size_of_comm_group = num_compute_nodes * num_gpus_per_node
        if mpi_rank < mpi_size_of_comm_group:
            rank_id_per_node = mpi_rank % num_gpus_per_node
            if cfg.ccl_backend == "nccl" and torch.cuda.is_available():
                device_id = gpu_ids_per_node[rank_id_per_node]
                device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device_id)
          
            elif cfg.ccl_backend in ["ccl", "xccl"] and torch.xpu.is_available():
                device_id = gpu_ids_per_node[rank_id_per_node]
                device = torch.device(f"xpu:{device_id}")

            else:
                device = torch.device('cpu')
                if mpi_rank == 0:
                    log.info("[COMM] Using CPU device")
        else:
            # Excluded ranks use CPU device
            device = torch.device('cpu')
            log.info(f"[COMM] Rank {mpi_rank} excluded from across_node groups, using CPU device")

        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {num_gpus_per_node} across-node groups")
            log.info("")

    # ----------------------------------------------------------------------------
    # FLATVIEW MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "flatview":
        
        # CONFIG PARSING
        flatview_config = comm_config.flatview
        num_compute_nodes = flatview_config.num_compute_nodes
        num_gpus_per_node = flatview_config.num_gpus_per_node
        gpu_ids_per_node = flatview_config.gpu_ids_per_node
        
        # For flatview, all ranks participate
        
        mpi_size = MPI.COMM_WORLD.Get_size()
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Flatview: {num_compute_nodes} nodes, {num_gpus_per_node} GPUs per node, Device IDs: {gpu_ids_per_node}")
            log.info("")
            log.info(f"[COMM][GROUP CREATION] Flatview groups: All ranks (0-{mpi_size-1}) use world group")
             
 
        # DEVICE ALLOCATION
        world_group = None  
        mpi_size_of_comm_group = num_compute_nodes * num_gpus_per_node
        if mpi_rank < mpi_size_of_comm_group:
            rank_id_per_node = mpi_rank % num_gpus_per_node
            if cfg.ccl_backend == "nccl" and torch.cuda.is_available():
                device_id = gpu_ids_per_node[rank_id_per_node]
                device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device_id)
            elif cfg.ccl_backend in ["ccl", "xccl"] and torch.xpu.is_available():
                device_id = gpu_ids_per_node[rank_id_per_node]
                device = torch.device(f"xpu:{device_id}")
            else:
                device = torch.device("cpu")
                if mpi_rank == 0:
                    log.info("[COMM] Using CPU device")
        else:
            # Excluded ranks use CPU device
            device = torch.device('cpu')
            log.info(f"[COMM] Rank {mpi_rank} excluded from flatview groups, using CPU device")
        
        world_group_ranks = list(range(mpi_size))



    return {
        'my_within_group': my_within_group,
        'my_across_group': my_across_group, 
        'world_group': world_group,
        'device': device,
        'within_group_id': within_group_id,
        'across_group_id': across_group_id,
        'within_group_ranks': within_group_ranks,
        'across_group_ranks': across_group_ranks,
        'world_group_ranks': world_group_ranks,
        'ranks_responsible_for_logging': ranks_responsible_for_logging,
    }