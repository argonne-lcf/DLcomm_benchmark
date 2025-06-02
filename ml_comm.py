# IMPORTINGS
# ----------------------------------------------------------------------------

import hydra
from omegaconf import DictConfig
import importlib
import json
import trace_utils
from pathlib import Path

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
        dtype = cfg.dtype
        if dtype not in self.spec["dtype"]:
            errors.append(
                f"Invalid dtype '{dtype}'. Valid: {self.spec['dtype']}"
            )

        # buffer_size
        try:
            buffer_bytes = parse_buffer_size(cfg.buffer_size)
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
    
    config_spec_path="./config/config_spec.json"
    with open(config_spec_path, "r") as f:
        spec = json.load(f)

    validator = ConfigValidator(spec)
    buffer_in_bytes = validator.validate(cfg)

    print("Final config:")
    print(f"  • framework        = {cfg.framework}")
    print(f"  • backend          = {cfg.ccl_backend}")
    print(f"  • collective_name  = {cfg.collective.name}")
    print(f"  • op               = {cfg.collective.op}")
    print(f"  • algo             = {cfg.collective.algo}")
    print(f"  • buffer_size      = {cfg.buffer_size} bytes")
    print(f"  • dtype            = {cfg.dtype}")
    print(f"  • horizontal.num_gpus = {cfg.horizontal.num_gpus}")
    print(f"  • vertical.num_nodes  = {cfg.vertical.num_nodes}")
    print(f"  • use_unitrace     = {cfg.use_unitrace}")

    print(f"\n\n ------------------------------------------------------------------------- \n\n")

    #dynamically we are creating the correspoding file name in profile apps
    framework = cfg.framework        # "pytorch tensorflow jax etc
    backend   = cfg.ccl_backend      # xccl" "xla" based on intel device or nvida



    module_name = f"profile_apps.{framework}_{backend}"

    # checking whether it exist or not
    path_to_module_py = Path(__file__).parent / "profile_apps" / f"{framework}_{backend}.py"
    if not path_to_module_py.exists():
        raise RuntimeError(f"Cannot find {framework}_{backend}.py")




    # Having MPI info and ranks
    num_nodes = cfg.vertical.num_nodes
    ranks_per_node = cfg.horizontal.num_gpus
    total_ranks = num_nodes * ranks_per_node



    #  trace directory  
    trace_root = Path.cwd() / "trace"
    trace_root.mkdir(parents=True, exist_ok=True)

    #  environment variables for unitrace 
    env_vars = {}
    if cfg.use_unitrace:
        env_vars["UNITRACE_LOG_LEVEL"] = "INFO"
        # If using XCCL, turn on CCL logging; otherwise default
        # env_vars["CCL_LOG_LEVEL"] = "DEBUG" if backend == "xccl" else "INFO"

    
    
    #env_vars["CCL_ALLREDUCE"] = "topo",
    #env_vars["CCL_ALLREDUCE_SCALEOUT"] = "rabenseifner",
     


    #How should we handle different ranks and cpu binding, we can generate
    # cpu_bind str with a function

    # launch MPI+UniTrace
    trace_utils.run_mpiexec_and_unitrace(
        python_module = module_name,
        buf_size_bytes = buffer_in_bytes,
        trace_dir = trace_root,
        env_vars = env_vars,
        np = total_ranks,
        ppn = ranks_per_node,
        cpu_bind = "list:4:9:14:19:20:25:56:61:66:71:74:79"   
    )
 

if __name__ == "__main__":
    main()