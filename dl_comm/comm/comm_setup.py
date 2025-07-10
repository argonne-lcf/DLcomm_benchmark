import torch
from mpi4py import MPI
from omegaconf import DictConfig
from dl_comm.timer import timer


def setup_communication_groups(cfg: DictConfig, mpi_rank, log, dist=None):
 
    
    comm_config = cfg.comm_group
    comm_mode = comm_config.mode
    
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
    
    if comm_mode == "within_node" or comm_mode == "combined":
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Setting up communication groups for mode: Within")


        # CONFIG PARSING
        if comm_mode == "combined":
            within_config = comm_config.combined.within_node
        else:
            within_config = comm_config.within_node
        num_gpus_per_node = within_config.num_gpus_per_node
        num_compute_nodes = within_config.num_compute_nodes
        gpu_ids_per_node = within_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Within-node: {num_gpus_per_node} GPUs per node, Device IDs: {gpu_ids_per_node}")
            log.info("[COMM][GROUP CREATION] Within-node groups:")

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

        # DEVICE ALLOCATION
        rank_id_per_node = mpi_rank % num_gpus_per_node
        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[rank_id_per_node]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")

        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {num_compute_nodes} within-node groups")

    # ----------------------------------------------------------------------------
    # ACROSS NODE MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "across_node" or comm_mode == "combined":
        if mpi_rank == 0:
            log.info("")
            log.info(f"[COMM][CONFIG] Setting up communication groups for mode: Across")
        # CONFIG PARSING
        if comm_mode == "combined":
            across_config = comm_config.combined.across_node
        else:
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

        # DEVICE ALLOCATION
        rank_id_per_node = mpi_rank % num_gpus_per_node
        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[rank_id_per_node]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")

        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {num_gpus_per_node} across-node groups")


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
        if torch.xpu.is_available():
            rank_id_per_node = mpi_rank % num_gpus_per_node
            device_id = gpu_ids_per_node[rank_id_per_node]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device("cpu")
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")
        
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