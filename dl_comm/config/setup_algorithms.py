import os
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Optional


# Algorithm names NCCL and RCCL accept in NCCL_ALGO / RCCL_ALGO. Anything
# else makes the tuner reject the whole string, so names meant for another
# backend have to be filtered out rather than forwarded.
_NCCL_ALGOS = {"TREE", "RING", "COLLNET", "COLLNETDIRECT", "COLLNETCHAIN",
               "NVLS", "NVLSTREE", "PAT"}


def setup_collective_algorithms_ccl(cfg: DictConfig, coll_cfg, comm_mode: str, log=None):
    """Setup CCL algorithm overrides for a specific collective and mode"""
    if cfg.extended_logging:
        os.environ["CCL_LOG_LEVEL"] = "debug"
    
    scale_up_algorithm = getattr(coll_cfg, 'scale_up_algorithm', '') or ''
    if scale_up_algorithm and scale_up_algorithm.strip().lower() != 'default':
        scale_up_override = f"CCL_{coll_cfg.collective_name.upper()}"
        os.environ[scale_up_override] = coll_cfg.scale_up_algorithm.lower()

    scale_out_algorithm = getattr(coll_cfg, 'scale_out_algorithm', '') or ''
    if scale_out_algorithm and scale_out_algorithm.strip().lower() != 'default':
        scale_out_override = f"CCL_{coll_cfg.collective_name.upper()}_SCALEOUT"
        os.environ[scale_out_override] = coll_cfg.scale_out_algorithm.lower()


def setup_nccl_algorithms_all(cfg: DictConfig, log=None):
    """Setup NCCL algorithm overrides for all collectives at once"""
    ccl_backend = cfg.ccl_backend.lower()
    
    if ccl_backend not in ["nccl", "rccl"]:
        return
        
    if cfg.extended_logging:
        if ccl_backend == "nccl":
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_DEBUG_SUBSYS"] = "COLL,TUNING"
        else:
            os.environ["RCCL_DEBUG"] = "INFO"
    
    algo_parts = []
    
    # Handle new task structure
    tasks_to_run = cfg.order_of_run
    if isinstance(tasks_to_run, str):
        tasks_to_run = [tasks_to_run]
        
    for task_name in tasks_to_run:
        if not hasattr(cfg, task_name):
            continue
            
        task_config = getattr(cfg, task_name)
        comm_mode = task_config.comm_group
        coll_cfg = task_config.collective
        
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

        # Keep only names NCCL's tuner understands. The algorithm names in a
        # config are written for whichever backend the config was authored
        # against, and the oneCCL vocabulary does not match NCCL's: a config
        # asking for "topo" makes NCCL log
        #     WARN Unrecognized element token "topo" when parsing
        #          "allreduce:topo"
        # and then fail the collective with "invalid usage". Passing an
        # unknown token through is never useful, so drop it and say so.
        unknown = [a for a in algorithms if a.upper() not in _NCCL_ALGOS]
        algorithms = [a for a in algorithms if a.upper() in _NCCL_ALGOS]
        if unknown and log:
            log.output(
                f"[ALGO] {coll_cfg.collective_name}: ignoring "
                f"{', '.join(unknown)} because {ccl_backend.upper()} does not "
                f"define {'it' if len(unknown) == 1 else 'them'}; "
                f"supported values are {', '.join(sorted(_NCCL_ALGOS))}. "
                f"The collective runs with the backend default.")
        
        if algorithms:
            collective_name = coll_cfg.collective_name.lower()
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
        
    if cfg.extended_logging:
        os.environ["CCL_LOG_LEVEL"] = "debug"
    
    # Collect all unique collective + algorithm combinations
    collective_algorithms = {}
    
    # Handle new task structure
    tasks_to_run = cfg.order_of_run
    if isinstance(tasks_to_run, str):
        tasks_to_run = [tasks_to_run]
        
    for task_name in tasks_to_run:
        if not hasattr(cfg, task_name):
            continue
            
        task_config = getattr(cfg, task_name)
        comm_mode = task_config.comm_group
        coll_cfg = task_config.collective
        collective_name = coll_cfg.collective_name.upper()
        
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
    
    # Handle new task structure
    tasks_to_run = cfg.order_of_run
    if isinstance(tasks_to_run, str):
        tasks_to_run = [tasks_to_run]
        
    for task_name in tasks_to_run:
        if not hasattr(cfg, task_name):
            continue
            
        algorithm_info[task_name] = []
        task_config = getattr(cfg, task_name)
        comm_mode = task_config.comm_group
        coll_cfg = task_config.collective
        
        scale_up_algorithm = getattr(coll_cfg, 'scale_up_algorithm', '') or 'default'
        scale_out_algorithm = getattr(coll_cfg, 'scale_out_algorithm', '') or 'default'
        
        algorithms = []
        if scale_up_algorithm.strip().lower() != 'default':
            algorithms.append(f"scale_up:{scale_up_algorithm}")
        if scale_out_algorithm.strip().lower() != 'default':
            algorithms.append(f"scale_out:{scale_out_algorithm}")
        
        if not algorithms:
            algorithms.append("default")
            
        algorithm_info[task_name].append((
            comm_mode,
            coll_cfg.collective_name,
            ", ".join(algorithms)
        ))
    
    return algorithm_info