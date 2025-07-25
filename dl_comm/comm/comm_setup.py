import torch
from mpi4py import MPI
from omegaconf import DictConfig
from dl_comm.timer import timer


def setup_communication_groups(mode_cfg, mpi_rank, log, dist=None, force_mode=None, ccl_backend=None):
 
    
    # With the new structure, force_mode is always required as we pass the specific mode_cfg
    if not force_mode:
        raise ValueError("setup_communication_groups() requires force_mode parameter")
    
    comm_mode = force_mode
    
    my_within_group = None
    my_across_group = None
    flat_group = None
    within_group_id = None
    across_group_id = None
    ranks_responsible_for_logging = set([0])  # Rank 0 always responsible for world/flatview
    participating = False  # Flag to track if rank participates in collective operations

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
            
            for node in range(num_compute_nodes):
                group_ranks = []
                group = None

                for gpu_index in range(len(gpu_ids_per_node)):
                    rank = node * len(gpu_ids_per_node) + gpu_index
                    group_ranks.append(rank)
                group = dist.new_group(ranks=group_ranks,use_local_synchronization=True) 

                if group_ranks:
                    responsible_rank = min(group_ranks)
                    ranks_responsible_for_logging.add(responsible_rank)
                if mpi_rank in group_ranks:
                    my_within_group = group
                    within_group_id = node
                    participating = True
                else:
                    # Create a dummy group for non-participating ranks
                    non_participating_ranks = list(range(num_compute_nodes * len(gpu_ids_per_node), mpi_size))
                    if non_participating_ranks:
                        dist.new_group(ranks=non_participating_ranks, use_local_synchronization=True)
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Within Group-{node}] Ranks: {group_ranks}, Required GPUs: {gpu_ids_per_node}, Logging: rank {responsible_rank}")
            
      
            if mpi_rank < num_compute_nodes * len(gpu_ids_per_node):
                 
                rank_gpu_index = mpi_rank % len(gpu_ids_per_node)
                assigned_gpu_id = gpu_ids_per_node[rank_gpu_index]
                
         
                if torch.cuda.is_available():
                    torch.cuda.set_device(assigned_gpu_id)
                    device = torch.device(f"cuda:{assigned_gpu_id}")
                    device_type = "cuda"
                elif torch.xpu.is_available():
                    device = torch.device(f"xpu:{assigned_gpu_id}")
                    device_type = "xpu"
                else:
                    device = torch.device('cpu')
                    device_type = "cpu"
                    assigned_gpu_id = "cpu"
            else:
                # Non-participating ranks get CPU device
                device = torch.device('cpu')
                device_type = "cpu"
                assigned_gpu_id = "cpu"


            
   

 
    

        # Device assignment completed above

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
                    # Calculate rank for this GPU index across nodes (sequential ranks)
                    rank = node * len(gpu_ids_per_node) + gpu_idx
                    if rank < mpi_size:  # Ensure rank exists
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
                        participating = True
                    else:
                        # Create a dummy group for non-participating ranks
                        non_participating_ranks = list(range(num_compute_nodes * len(gpu_ids_per_node), mpi_size))
                        if non_participating_ranks:
                            dist.new_group(ranks=non_participating_ranks, use_local_synchronization=True)
            
            # Device assignment logic for across_node mode
            if mpi_rank < num_compute_nodes * len(gpu_ids_per_node):
                # Calculate which GPU ID this rank should use
                rank_gpu_index = mpi_rank % len(gpu_ids_per_node)
                assigned_gpu_id = gpu_ids_per_node[rank_gpu_index]
                
                # Assign device based on backend
                if torch.cuda.is_available():
                    torch.cuda.set_device(assigned_gpu_id)
                    device = torch.device(f"cuda:{assigned_gpu_id}")
                    device_type = "cuda"
                elif torch.xpu.is_available():
                    device = torch.device(f"xpu:{assigned_gpu_id}")
                    device_type = "xpu"
                else:
                    device = torch.device('cpu')
                    device_type = "cpu"
                    assigned_gpu_id = "cpu"
            else:
                # Non-participating ranks get CPU device
                device = torch.device('cpu')
                device_type = "cpu"
                assigned_gpu_id = "cpu"
            
  

        # Device assignment completed above 

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
                for gpu_index in range(len(gpu_ids_per_node)):
                    # Calculate sequential rank for this GPU index on this node
                    rank = node * len(gpu_ids_per_node) + gpu_index
                    if rank < mpi_size and rank not in group_ranks:
                        group_ranks.append(rank)
            
            
            if group_ranks:
                responsible_rank = min(group_ranks)
                ranks_responsible_for_logging.add(responsible_rank)
                
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Flatview] Ranks: {group_ranks}, Required GPUs: {gpu_ids_per_node}, Logging: rank {responsible_rank}")
                
                flat_group = dist.new_group(ranks=group_ranks, use_local_synchronization=True)
                flat_group_ranks = group_ranks
                if mpi_rank in group_ranks:
                    participating = True
            else:
                flat_group = None
                flat_group_ranks = []
                # Create a dummy group for non-participating ranks
                non_participating_ranks = list(range(num_compute_nodes * len(gpu_ids_per_node), mpi_size))
                if non_participating_ranks:
                    dist.new_group(ranks=non_participating_ranks, use_local_synchronization=True)
            
            # Device assignment logic for flatview mode
            if mpi_rank < num_compute_nodes * len(gpu_ids_per_node):
                # Calculate which GPU ID this rank should use
                rank_gpu_index = mpi_rank % len(gpu_ids_per_node)
                assigned_gpu_id = gpu_ids_per_node[rank_gpu_index]
                
                # Assign device based on backend
                if torch.cuda.is_available():
                    torch.cuda.set_device(assigned_gpu_id)
                    device = torch.device(f"cuda:{assigned_gpu_id}")
                    device_type = "cuda"
                elif torch.xpu.is_available():
                    device = torch.device(f"xpu:{assigned_gpu_id}")
                    device_type = "xpu"
                else:
                    device = torch.device('cpu')
                    device_type = "cpu"
                    assigned_gpu_id = "cpu"
            else:
                # Non-participating ranks get CPU device
                device = torch.device('cpu')
                device_type = "cpu"
                assigned_gpu_id = "cpu"
            
 
    return {
        'my_within_group': my_within_group,
        'my_across_group': my_across_group, 
        'flat_group': flat_group,
        'within_group_id': within_group_id,
        'across_group_id': across_group_id,
        'ranks_responsible_for_logging': ranks_responsible_for_logging,
        'device': device if 'device' in locals() else torch.device('cpu'),
        'participating': participating,
    }