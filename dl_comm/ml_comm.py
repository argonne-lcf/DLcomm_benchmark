# IMPORTINGS
# ----------------------------------------------------------------------------
import os
import re
import subprocess
import hydra
from omegaconf import DictConfig
import importlib
import json
from pathlib import Path
import subprocess
from dl_comm.utils.utility import DLIOLogger
from dl_comm.helpers import run_and_split, report_ccl_selection

log = DLIOLogger.get_instance()
# ----------------------------------------------------------------------------

# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def parse_buffer_size(size_str: str) -> int:

    s = size_str.strip().upper()
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024)
    elif s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    elif s.endswith("B"):
        return int(float(s[:-1]))
    else:
        raise ValueError(f"payload.size='{size_str}' has unknown format. Use '1MB', '512KB' etc")


# ----------------------------------------------------------------------------

# VALIDATION
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


class ConfigValidator:
    def __init__(self, spec: dict):
        self.spec = spec

    def validate(self, cfg: DictConfig):
 
        errors = []

        # framework
        framework = cfg.framework
        if framework not in self.spec["framework"]:
            errors.append(
                f"Invalid framework '{framework}'. Valid options: {self.spec['framework']}"
            )

        # ccl_backend
        backend = getattr(cfg, "ccl_backend", None)
        valid_backends = self.spec["backend"].get(framework, [])
        if backend not in valid_backends:
            errors.append(
                f"Invalid ccl_backend '{backend}' for framework '{framework}'. "
                f"Valid: {valid_backends}"
            )

        # collective.name
        collective = cfg.collective.name
        if collective not in self.spec["collective"]:
            errors.append(
                f"Invalid collective '{collective}'. Valid: {self.spec['collective']}"
            )

        # collective.op
        op = cfg.collective.op
        valid_ops = self.spec["op"].get(collective, [])
        if op not in valid_ops:
            errors.append(
                f"Invalid op '{op}' for collective '{collective}'. Valid: {valid_ops}"
            )

        # collective.algo
        algo = cfg.collective.algo
        valid_algos = self.spec["algo"].get(collective, [])
        if algo not in valid_algos:
            errors.append(
                f"Invalid algo '{algo}' for collective '{collective}'. Valid: {valid_algos}"
            )

        # dtype 
        dtype = cfg.collective.payload.dtype
        if dtype not in self.spec["dtype"]:
            errors.append(
                f"Invalid dtype '{dtype}'. Valid: {self.spec['dtype']}"
            )

        # buffer_size
        try:
            buffer_bytes = parse_buffer_size(cfg.collective.payload.buffer_size)
        except ValueError as ve:
            errors.append(str(ve))

    
        if errors:
            raise ValueError("ALl ERRORS:\n" + "\n".join(errors))

     
        return buffer_bytes



# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

"""def run_communication(cfg):
    if cfg.package == "ccl" and cfg.algorithm == "ring":
        allreduce_ccl_ring(cfg)
    elif cfg.package == "torch" and cfg.algorithm == "ring":
        allreduce_ring_torch(cfg)
    elif cfg.package == "ccl" and cfg.algorithm == "tree":
        allreduce_tree_ccl(cfg)
    elif cfg.package == "mpi" and cfg.algorithm == "ring":
        allreduce_ring_mpi(cfg)
    else:
        raise NotImplementedError(
            f"No implementation for (package={cfg.package}, algorithm={cfg.algorithm})"
        )
"""
# ----------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
  
    log.info("-------------------------------------------------------------------------")
    log.info("[CONFIG] Loading schema and validating user YAML")
   
    config_spec_path = Path(__file__).parent / "config" / "config_spec.json"

    with open(config_spec_path, "r") as f:
        spec = json.load(f)
    validator = ConfigValidator(spec)
    buffer_in_bytes = validator.validate(cfg)

   
 
    log.info("[CONFIG] Final validated settings\n")
 
    log.info(f"  • framework           = {cfg.framework}")
    log.info(f"  • backend             = {cfg.ccl_backend}")
    log.info(f"  • collective_name     = {cfg.collective.name}")
    log.info(f"  • op                  = {cfg.collective.op}")
    log.info(f"  • algo                = {cfg.collective.algo}")
    log.info(f"  • buffer_size         = {cfg.collective.payload.buffer_size} ({buffer_in_bytes} bytes)")
    log.info(f"  • dtype               = {cfg.collective.payload.dtype}")
    log.info(f"  • horizontal.num_gpus  = {cfg.horizontal.tp_degree}")
    log.info(f"  • vertical.num_nodes   = {cfg.vertical.dp_degree}")
    log.info(f"  • use_unitrace        = {cfg.use_unitrace}")
 
  
    log.info("-------------------------------------------------------------------------")
    log.info("[APP] Determining Profiling Module Path")
 
    framework = cfg.framework
    backend = cfg.ccl_backend
    module_name = f"dl_comm.profile_apps.{framework}_{backend}"
    path_to_module_py = Path(__file__).parent / "profile_apps" / f"{framework}_{backend}.py"
    if not path_to_module_py.exists():
        raise RuntimeError(f"Cannot find profiling module: '{framework}_{backend}.py'")
    log.info(f"[APP] Will use: {module_name}  (file: {path_to_module_py})")
    log.info("-------------------------------------------------------------------------")

 
    log.info("[MPI] Computing rank counts")
 
    num_nodes = cfg.vertical.dp_degree
    ranks_per_node = cfg.horizontal.tp_degree
    total_ranks = num_nodes * ranks_per_node
    log.info(f"[MPI] num_nodes       = {num_nodes}")
    log.info(f"[MPI] ranks_per_node  = {ranks_per_node}")
    log.info(f"[MPI] total_ranks     = {total_ranks}")
    log.info(f"\n")
    log.info("[MPI] Building mpiexec command")
 

    
    mpi_cmd = [
        
        "mpiexec",
        "--env", "CCL_ATL_TRANSPORT=mpi",
        "--env", "CCL_ATL_SHM=0",
        "--env", "CCL_LOG_LEVEL=debug",
        "--env", "TORCH_CPP_LOG_LEVEL=error",
        "--env", "CCL_PROCESS_LAUNCHER=pmix",
        "--np", str(total_ranks),
        "-ppn", str(ranks_per_node),
        "python3", "-m", module_name,

        cfg.framework,
        cfg.collective.name,
        cfg.collective.op,
        str(buffer_in_bytes),
        str(cfg.collective.iterations),
        cfg.collective.payload.dtype,
        str(cfg.horizontal.tp_degree),
        str(cfg.vertical.dp_degree),
        str(cfg.flatview)
    ]


    log.output(f"[MPI] Command → {' '.join(mpi_cmd)}")
    
    log.info("[MPI] Launching profiling job")
  

 
    ccl_log_path="ccl_info.log"
    run_and_split(mpi_cmd, ccl_log_path=ccl_log_path)

    log.info("-------------------------------------------------------------------------")
    log.info("[MPI] Job complete")
    log.info("-------------------------------------------------------------------------")

    log.info("Parsing selection")
    report_ccl_selection(ccl_log_path, cfg.collective.name, log)
            
    log.info("-------------------------------------------------------------------------")
    log.info("[EXIT] All Done.")
    log.info("-------------------------------------------------------------------------")

if __name__ == "__main__":
    main()

    