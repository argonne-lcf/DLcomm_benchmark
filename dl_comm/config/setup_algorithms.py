import os
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Optional


def setup_collective_algorithms_ccl(cfg: DictConfig, coll_cfg, comm_mode: str, log=None):
    """Setup CCL algorithm overrides for a specific collective and mode"""
    if cfg.ccl_debug:
        os.environ["CCL_LOG_LEVEL"] = "debug"
    
    scale_up_algorithm = getattr(coll_cfg, 'scale_up_algorithm', '') or ''
    if scale_up_algorithm and scale_up_algorithm.strip().lower() != 'default':
        scale_up_override = f"CCL_{coll_cfg.name.upper()}"
        os.environ[scale_up_override] = coll_cfg.scale_up_algorithm.lower()

    scale_out_algorithm = getattr(coll_cfg, 'scale_out_algorithm', '') or ''
    if scale_out_algorithm and scale_out_algorithm.strip().lower() != 'default':
        scale_out_override = f"CCL_{coll_cfg.name.upper()}_SCALEOUT"
        os.environ[scale_out_override] = coll_cfg.scale_out_algorithm.lower()


def setup_nccl_algorithms_all(cfg: DictConfig, log=None):
    """Setup NCCL algorithm overrides for all collectives at once"""
    ccl_backend = cfg.ccl_backend.lower()
    
    if ccl_backend not in ["nccl", "rccl"]:
        return
        
    if cfg.ccl_debug:
        if ccl_backend == "nccl":
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_DEBUG_SUBSYS"] = "COLL,TUNING"
        else:
            os.environ["RCCL_DEBUG"] = "INFO"
    
    algo_parts = []
    
    for impl_config in cfg.implementations:
        impl_name = impl_config.name
        if impl_name not in cfg.order_of_run:
            continue
            
        comm_groups = impl_config.comm_groups
        available_modes = []
        if hasattr(comm_groups, 'within_node'):
            available_modes.append(('within_node', comm_groups.within_node))
        if hasattr(comm_groups, 'across_node'):
            available_modes.append(('across_node', comm_groups.across_node))
        if hasattr(comm_groups, 'flatview'):
            available_modes.append(('flatview', comm_groups.flatview))
        
        for comm_mode, mode_cfg in available_modes:
            coll_cfg = mode_cfg.collective
            
            scale_up_algorithm = getattr(coll_cfg, 'scale_up_algorithm', '') or ''
            scale_out_algorithm = getattr(coll_cfg, 'scale_out_algorithm', '') or ''
            
            if scale_up_algorithm:
                scale_up_algorithm = scale_up_algorithm.strip().lower()
            if scale_out_algorithm:
                scale_out_algorithm = scale_out_algorithm.strip().lower()
            
            algorithms = []
            if scale_up_algorithm and scale_up_algorithm != 'default':
                algorithms.append(scale_up_algorithm)
            if scale_out_algorithm and scale_out_algorithm != 'default' and scale_out_algorithm != scale_up_algorithm:
                algorithms.append(scale_out_algorithm)
            
            if algorithms:
                collective_name = coll_cfg.name.lower()
                algo_string = f"{collective_name}:{','.join(algorithms)}"
                if algo_string not in algo_parts:
                    algo_parts.append(algo_string)
    
    if algo_parts:
        algo_env_var = "NCCL_ALGO" if ccl_backend == "nccl" else "RCCL_ALGO"
        algo_value = ";".join(algo_parts)
        os.environ[algo_env_var] = algo_value


def setup_ccl_algorithms_all(cfg: DictConfig, log=None):
    """Setup CCL algorithm overrides for all collectives at once"""
    ccl_backend = cfg.ccl_backend.lower()
    
    if ccl_backend not in ["ccl", "xccl"]:
        return
        
    if cfg.ccl_debug:
        os.environ["CCL_LOG_LEVEL"] = "debug"
    
    # Collect all unique collective + algorithm combinations
    collective_algorithms = {}
    
    for impl_config in cfg.implementations:
        impl_name = impl_config.name
        if impl_name not in cfg.order_of_run:
            continue
            
        comm_groups = impl_config.comm_groups
        available_modes = []
        if hasattr(comm_groups, 'within_node'):
            available_modes.append(('within_node', comm_groups.within_node))
        if hasattr(comm_groups, 'across_node'):
            available_modes.append(('across_node', comm_groups.across_node))
        if hasattr(comm_groups, 'flatview'):
            available_modes.append(('flatview', comm_groups.flatview))
        
        for comm_mode, mode_cfg in available_modes:
            coll_cfg = mode_cfg.collective
            collective_name = coll_cfg.name.upper()
            
            # Map collective names to CCL environment variable names
            ccl_env_name_map = {
                'REDUCESCATTER': 'REDUCE_SCATTER',  # reducescatter -> CCL_REDUCE_SCATTER
                'ALLTOALLSINGLE': 'ALLTOALL'  # alltoallsingle -> CCL_ALLTOALL (they share same algorithm)
            }
            ccl_collective_name = ccl_env_name_map.get(collective_name, collective_name)
            
            scale_up_algorithm = getattr(coll_cfg, 'scale_up_algorithm', '') or ''
            scale_out_algorithm = getattr(coll_cfg, 'scale_out_algorithm', '') or ''
            
            # Set scale_up algorithm
            if scale_up_algorithm and scale_up_algorithm.strip().lower() != 'default':
                env_var = f"CCL_{ccl_collective_name}"
                algorithm = scale_up_algorithm.lower()
                collective_algorithms[env_var] = algorithm
            
            # Set scale_out algorithm
            if scale_out_algorithm and scale_out_algorithm.strip().lower() != 'default':
                collective_algorithms[f"CCL_{ccl_collective_name}_SCALEOUT"] = scale_out_algorithm.lower()
    
    # Set all environment variables
    for env_var, algorithm in collective_algorithms.items():
        os.environ[env_var] = algorithm


def setup_algorithm_overrides(cfg: DictConfig, log=None):
    """Setup algorithm overrides for all implementations before they start"""
    ccl_backend = cfg.ccl_backend.lower()
    
    if ccl_backend in ["nccl", "rccl"]:
        setup_nccl_algorithms_all(cfg, log)
    elif ccl_backend in ["ccl", "xccl"]:
        setup_ccl_algorithms_all(cfg, log)


def get_algorithm_info(cfg: DictConfig) -> Dict[str, List[Tuple[str, str, str]]]:
    """Get algorithm information for display purposes"""
    algorithm_info = {}
    
    for impl_config in cfg.implementations:
        impl_name = impl_config.name
        if impl_name not in cfg.order_of_run:
            continue
            
        algorithm_info[impl_name] = []
        comm_groups = impl_config.comm_groups
        
        available_modes = []
        if hasattr(comm_groups, 'within_node'):
            available_modes.append(('within_node', comm_groups.within_node))
        if hasattr(comm_groups, 'across_node'):
            available_modes.append(('across_node', comm_groups.across_node))
        if hasattr(comm_groups, 'flatview'):
            available_modes.append(('flatview', comm_groups.flatview))
        
        for comm_mode, mode_cfg in available_modes:
            coll_cfg = mode_cfg.collective
            
            scale_up_algorithm = getattr(coll_cfg, 'scale_up_algorithm', '') or 'default'
            scale_out_algorithm = getattr(coll_cfg, 'scale_out_algorithm', '') or 'default'
            
            algorithms = []
            if scale_up_algorithm.strip().lower() != 'default':
                algorithms.append(f"scale_up:{scale_up_algorithm}")
            if scale_out_algorithm.strip().lower() != 'default':
                algorithms.append(f"scale_out:{scale_out_algorithm}")
            
            if not algorithms:
                algorithms.append("default")
                
            algorithm_info[impl_name].append((
                comm_mode,
                coll_cfg.name,
                ", ".join(algorithms)
            ))
    
    return algorithm_info