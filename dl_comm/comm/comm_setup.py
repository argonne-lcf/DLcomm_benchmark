# Framework imports are handled dynamically
from mpi4py import MPI
from omegaconf import DictConfig
from dl_comm.timer import timer
import os


def allocate_device(device_type, assigned_device_id, log, mpi_rank, framework):
    
    # FRAMEWORK-PYTORCH
    if framework == 'pytorch':
        import torch

        # GPU ALLOCATION
        if device_type == 'gpu':
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{assigned_device_id}")
                torch.cuda.set_device(assigned_device_id)
                return device
            elif torch.xpu.is_available():
                device = torch.device(f"xpu:{assigned_device_id}")
                return device


        # CPU ALLOCATION
        elif device_type == 'cpu':
             
            cpu_binding = os.environ.get("CPU_BINDING")
            if cpu_binding and cpu_binding.startswith("list:"):
                
                cpu_core = assigned_device_id
                os.sched_setaffinity(0, {cpu_core})

                #log.info(f"[CPU] Rank {mpi_rank} bound to CPU core {cpu_core}")
                
            return torch.device('cpu')

    # FRAMEWORK-JAX
    elif framework == 'jax':
        import jax

        # GPU ALLOCATION
        if device_type == 'gpu':
            device = jax.devices('gpu')[assigned_device_id]
            return device

        # CPU ALLOCATION
        elif device_type == 'cpu':
            cpu_binding = os.environ.get("CPU_BINDING")
            if cpu_binding and cpu_binding.startswith("list:"):
                cpu_core = assigned_device_id
                os.sched_setaffinity(0, {cpu_core})
            
            return jax.devices('cpu')[0]



def setup_communication_groups(mode_cfg, mpi_rank, log, dist=None, force_mode=None, full_cfg=None):
 
    
 
    
    device_type = full_cfg.device_type.lower()
    framework = full_cfg.framework.lower()
    
    comm_mode = force_mode
    
    my_within_group = None
    my_across_group = None
    flat_group = None
    device = None  # Will be assigned based on group membership
    within_group_id = None
    across_group_id = None
    ranks_responsible_for_logging = set([0])  # Rank 0 always responsible for world/flatview

    mpi_size=MPI.COMM_WORLD.Get_size()
    
    # Calculate available devices once at the beginning
    if framework == 'pytorch':
        import torch
        if device_type == 'cpu':
            available_devices = mode_cfg.num_devices_per_node
        elif torch.cuda.is_available():
            available_devices = torch.cuda.device_count()
        elif torch.xpu.is_available():
            available_devices = torch.xpu.device_count()
        else:
            available_devices = 1
    elif framework == 'jax':
        import jax
        if device_type == 'cpu':
            available_devices = mode_cfg.num_devices_per_node
        else:
            available_devices = jax.device_count()

    
    # ----------------------------------------------------------------------------
    # WITHIN NODE MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "within_node":
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Setting up communication groups for mode: Within")


        # CONFIG PARSING
        within_config = mode_cfg
        num_devices_per_node = within_config.num_devices_per_node
        num_compute_nodes = within_config.num_compute_nodes
        device_ids_per_node = within_config.device_ids_per_node
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Within-node: {num_devices_per_node} devices per node, Device IDs: {device_ids_per_node}")
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
                for gpu_idx, gpu_id in enumerate(device_ids_per_node):
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
                    
                    # Assign device based on position in gpu_ids list
                    gpu_idx_in_group = group_ranks.index(mpi_rank)
                    assigned_device_id = device_ids_per_node[gpu_idx_in_group]
                    
                    # Set device based on device_type configuration
                    device = allocate_device(device_type, assigned_device_id, log, mpi_rank, framework)
            
 
                        
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Within Group-{node}] Ranks: {group_ranks}, Required Devices: {device_ids_per_node}, Logging: rank {responsible_rank}")
                    
           
                    

 
    

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
        num_devices_per_node = across_config.num_devices_per_node
        device_ids_per_node = across_config.device_ids_per_node
        
        if mpi_rank == 0:

            log.info(f"[COMM][CONFIG] Across-node: {num_compute_nodes} nodes, {num_devices_per_node} devices per node, Device IDs: {device_ids_per_node}")

            log.info("[COMM][GROUP CREATION] Across-node groups:")
        with timer("Group Creation (Across)"):
            my_across_group = None
            across_group_id = None
            
             
             
            
             
            for gpu_idx, required_gpu_id in enumerate(device_ids_per_node):
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
                        log.info(f"[COMM][GROUP CREATION][Across Group-{gpu_idx}] Ranks: {group_ranks}, Device ID: {required_gpu_id}, Logging: rank {responsible_rank}")
                     
                    group = dist.new_group(ranks=group_ranks,use_local_synchronization=True)
                    if mpi_rank in group_ranks:
                        my_across_group = group
                        across_group_id = gpu_idx
                        
                        # Assign device based on gpu_id for this group
                        assigned_device_id = required_gpu_id
                        
                        # Set device based on device_type configuration
                        device = allocate_device(device_type, assigned_device_id, log, mpi_rank, framework)
 


        
 

        # Device already allocated globally 

        if mpi_rank == 0:
            log.info(f"[COMM][GROUP CREATION] Created {num_devices_per_node} across-node groups")
            log.info("")

    # ----------------------------------------------------------------------------
    # FLATVIEW MODE
    # ----------------------------------------------------------------------------
    
    if comm_mode == "flatview":
        
        # CONFIG PARSING
        flatview_config = mode_cfg
        num_compute_nodes = flatview_config.num_compute_nodes
        num_devices_per_node = flatview_config.num_devices_per_node
        device_ids_per_node = flatview_config.device_ids_per_node
        
        
        
        if mpi_rank == 0:
            log.info(f"[COMM][CONFIG] Flatview: {num_compute_nodes} nodes, {num_devices_per_node} devices per node, Device IDs: {device_ids_per_node}")
            log.info("")
            log.info("[COMM][GROUP CREATION] Flatview groups:")

        with timer("Group Creation (Flatview)"):
            group_ranks = []
            # Use actual ranks per physical node from MPI configuration  
            ranks_per_physical_node = mpi_size // num_compute_nodes
            for node in range(num_compute_nodes):
                for gpu_idx, required_gpu_id in enumerate(device_ids_per_node):
                    # Sequential rank assignment based on gpu_ids order
                    rank = node * ranks_per_physical_node + gpu_idx
                    if rank < mpi_size and rank not in group_ranks:
                        group_ranks.append(rank)
            
            
            if group_ranks:
                responsible_rank = min(group_ranks)
                ranks_responsible_for_logging.add(responsible_rank)
                
                if mpi_rank == 0:
                    log.info(f"[COMM][GROUP CREATION][Flatview] Ranks: {group_ranks}, Required Devices: {device_ids_per_node}, Logging: rank {responsible_rank}")
                
                flat_group = dist.new_group(ranks=group_ranks, use_local_synchronization=True)
                flat_group_ranks = group_ranks
                
                # Assign device if this rank is in the flatview group
                if mpi_rank in group_ranks:
                    # Calculate which device this rank should use
                    rank_idx_in_group = group_ranks.index(mpi_rank)
                    node_id = rank_idx_in_group // len(device_ids_per_node)
                    device_idx_in_node = rank_idx_in_group % len(device_ids_per_node)
                    assigned_device_id = device_ids_per_node[device_idx_in_node]
                    
                    # Set device based on device_type configuration
                    device = allocate_device(device_type, assigned_device_id, log, mpi_rank, framework)
                        
 
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
    }