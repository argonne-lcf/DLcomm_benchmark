import torch
from omegaconf import DictConfig
from dl_comm.timer import timer


def setup_communication_groups(cfg: DictConfig, mpi_rank, log, dist=None):
 
    
    comm_config = cfg.collective.comm_group
    comm_mode = comm_config.mode
    
    my_within_group = None
    my_across_group = None
    world_group = None
    device = None

    if mpi_rank == 0:
        log.info(f"[COMM][CONFIG] Setting up communication groups for mode: {comm_mode}")
    
    # ----------------------------------------------------------------------------
    # WITHIN NODE MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "within_node":

        # CONFIG PARSING
        within_config = comm_config.within_node
        num_gpus_per_node = within_config.num_gpus_per_node
        num_compute_nodes = within_config.num_compute_nodes
        gpu_ids_per_node = within_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Within-node config: {num_gpus_per_node} GPUs, Device IDs: {gpu_ids_per_node}")
            log.info("")
        # DISTRIBUTED GROUP CREATION
        if mpi_rank == 0:
            log.info("[COMM][GROUP CREATION] Within-node groups:")
        with timer("Group Creation (Within)"):
            within_groups = []
            for node in range(num_compute_nodes):
                group_ranks = []
                for gpu in range(num_gpus_per_node):
                    rank = node * num_gpus_per_node + gpu
                    group_ranks.append(rank)
                within_groups.append(dist.new_group(ranks=group_ranks))
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Within Group-{node}] Ranks: {group_ranks}")
        
        node_id = mpi_rank // num_gpus_per_node
        rank_id_per_node = mpi_rank % num_gpus_per_node
        my_within_group = within_groups[node_id]

        # DEVICE ALLOCATION
        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[rank_id_per_node]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")

    # ----------------------------------------------------------------------------
    # ACROSS NODE MODE
    # ----------------------------------------------------------------------------
    
    elif comm_mode == "across_node":
        
        if mpi_rank == 0:
            log.info("[COMM][CONFIG] Setting up across-node groups")

        # CONFIG PARSING
        across_config = comm_config.across_node
        num_compute_nodes = across_config.num_compute_nodes
        num_gpus_per_node = across_config.num_gpus_per_node
        gpu_ids_per_node = across_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Across-node config: {num_compute_nodes} nodes, {num_gpus_per_node} GPUs per node, Device IDs: {gpu_ids_per_node}")
            log.info("")
        # DISTRIBUTED GROUP CREATION
        if mpi_rank == 0:
            log.info("[COMM][GROUP CREATION] Across-node groups:")
        with timer("Group Creation (Across)"):
            across_groups = []
            for i in range(num_gpus_per_node):
                group_ranks = []
                for node in range(num_compute_nodes):
                    rank = node * num_gpus_per_node + i
                    group_ranks.append(rank)
                across_groups.append(dist.new_group(ranks=group_ranks))
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Across Group-{i}] Ranks: {group_ranks}")
        
        rank_id_per_node = mpi_rank % num_gpus_per_node
        my_across_group = across_groups[rank_id_per_node]

        # DEVICE ALLOCATION
        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[rank_id_per_node]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")

        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {len(across_groups)} across-node groups")

    # ----------------------------------------------------------------------------
    # COMBINED MODE
    # ----------------------------------------------------------------------------
    
    elif comm_mode == "combined":
       
        if mpi_rank == 0:
            log.info("[COMM][CONFIG] Setting up combined (within + across) groups")
             

        # CONFIG PARSING
        within_config = comm_config.combined.within_node
        across_config = comm_config.combined.across_node
        
        num_gpus_per_node = within_config.num_gpus_per_node
        gpu_ids_per_node = within_config.gpu_ids_per_node
        num_compute_nodes = across_config.num_compute_nodes
        across_gpu_ids = across_config.gpu_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Combined config - Within: {num_gpus_per_node} GPUs Device IDs: {gpu_ids_per_node}, Across: {num_compute_nodes} nodes Device IDs: {across_gpu_ids}")
            log.info("")
            log.info("[COMM][GROUP CREATION] Combined mode groups:")

        # DISTRIBUTED GROUP CREATION - WITHIN
        with timer("Group Creation (Combined-Within)"):
            within_groups = []
            for node in range(num_compute_nodes):
                group_ranks = []
                for gpu in range(num_gpus_per_node):
                    rank = node * num_gpus_per_node + gpu
                    group_ranks.append(rank)
                within_groups.append(dist.new_group(ranks=group_ranks))
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Within Group-{node}] Ranks: {group_ranks}")
        
        node_id = mpi_rank // num_gpus_per_node
        rank_id_per_node = mpi_rank % num_gpus_per_node
        my_within_group = within_groups[node_id]

        # DISTRIBUTED GROUP CREATION - ACROSS
        with timer("Group Creation (Combined-Across)"):
            across_groups = []
            for i, gpu_id in enumerate(across_gpu_ids):
                rank_idx_per_node = gpu_ids_per_node.index(gpu_id)
                group_ranks = []
                for node in range(num_compute_nodes):
                    rank = node * num_gpus_per_node + rank_idx_per_node
                    group_ranks.append(rank)
                across_groups.append(dist.new_group(ranks=group_ranks))
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Across Group-{i}] Ranks: {group_ranks}")
        
        current_gpu_id = gpu_ids_per_node[rank_id_per_node]
        my_across_group = None
        if current_gpu_id in across_gpu_ids:
            across_idx = across_gpu_ids.index(current_gpu_id)
            my_across_group = across_groups[across_idx]

        # DEVICE ALLOCATION
        if torch.xpu.is_available():
            device_id = gpu_ids_per_node[rank_id_per_node]
            device = torch.device(f"xpu:{device_id}")
        else:
            device = torch.device('cpu')
            if mpi_rank == 0:
                log.info("[COMM] XPU not available, using CPU")
        
        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {len(within_groups)} within-node groups and {len(across_groups)} across-node groups")

    # ----------------------------------------------------------------------------
    # FLATVIEW MODE
    # ----------------------------------------------------------------------------
    
    elif comm_mode == "flatview":
         
        if mpi_rank == 0:
            log.info("[COMM][CONFIG] Using flatview (world group)")
            log.info("")
            log.info("[COMM][GROUP CREATION] Flatview: Using world group (all ranks)")

        # CONFIG PARSING and DEVICE ALLOCATION
        world_group = None  
        if torch.xpu.is_available():
            device = torch.device(f"xpu:{mpi_rank % torch.xpu.device_count()}")
        else:
            device = torch.device("cpu")

    else:
        raise ValueError(f"Unknown comm_mode: '{comm_mode}'. Valid: within, across, combined, flatview")
    
    return {
        'my_within_group': my_within_group,
        'my_across_group': my_across_group, 
        'world_group': world_group,
        'device': device,
    }