# IMPORTINGS
# ----------------------------------------------------------------------------

import hydra
from omegaconf import DictConfig
import importlib

from allreduce import (
    allreduce_ccl_ring,
    allreduce_ring_torch,
    allreduce_tree_ccl,
    allreduce_ring_mpi,
)

# ----------------------------------------------------------------------------

# HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def parse_buffer_size(size_str):
     
    size_str = size_str.strip().upper()
    if size_str.endswith("MB"):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith("KB"):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith("B"):
        return int(float(size_str[:-1]))
    else:
        raise ValueError(f"Unknown format: {size_str}")

# ----------------------------------------------------------------------------

# VALIDATION
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class ConfigValidator:
    def __init__(self, spec):
        self.spec = spec

    def validate(self, cfg):
        errors = []

        framework = cfg.framework
        if framework not in self.spec["framework"]:
            errors.append(f"Invalid framework '{framework}'. Valid options: {self.spec['framework']}")

        backend = getattr(cfg, "ccl_backend", None)
        if backend not in self.spec["backend"].get(framework, []):
            errors.append(f"Invalid backend '{backend}' for framework '{framework}'. Valid: {self.spec['backend'].get(framework, [])}")

        collective = cfg.collective.name
        if collective not in self.spec["collective"]:
            errors.append(f"Invalid collective '{collective}'. Valid: {self.spec['collective']}")

        op = cfg.collective.op
        valid_ops = self.spec["op"].get(collective, [])
        if op not in valid_ops:
            errors.append(f"Invalid op '{op}' for collective '{collective}'. Valid: {valid_ops}")

        algo = cfg.collective.algo
        valid_algos = self.spec["algo"].get(collective, [])
        if algo not in valid_algos:
            errors.append(f"Invalid algorithm '{algo}' for collective '{collective}'. Valid: {valid_algos}")

        dtype = cfg.payload.dtype
        if dtype not in self.spec["dtype"]:
            errors.append(f"Invalid dtype '{dtype}'. Valid: {self.spec['dtype']}")

        if errors:
            raise ValueError("CONFIG VALIDATION ERRORS:\n" + "\n".join(errors))

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

def run_communication(cfg):
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

# ----------------------------------------------------------------------------


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    config_spec_path="./config_spec.json"
    with open(config_spec_path, "r") as f:
        spec = json.load(f)

    validator = ConfigValidator(spec)
    validator.validate(cfg)

    cfg.buffer_size = parse_buffer_size(cfg.payload.size)


    print(cfg)

    run_communication(cfg)

if __name__ == "__main__":
    main()