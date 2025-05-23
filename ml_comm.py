#IMPORTINGS
#----------------------------------------------------------------------------



import hydra
from omegaconf import DictConfig
import importlib

from allreduce import (
    allreduce_ccl_ring,
    allreduce_ring_torch,
    allreduce_tree_ccl,
    allreduce_ring_mpi,
)




#----------------------------------------------------------------------------



#VALIDATION
#----------------------------------------------------------------------------

def validate_config(cfg):
     
    if cfg.package == "torch":
        
        if not importlib.util.find_spec("torch"):
            raise RuntimeError("PyTorch not found. Install torch.")
        
        if cfg.backend not in ["nccl", "mpi"]:
            raise ValueError(
                f"Invalid backend '{cfg.backend}' for package 'torch'. "
                "Choose 'nccl' or 'mpi'."
            )

    elif cfg.package == "mpi":
        if not importlib.util.find_spec("mpi4py"):
            raise RuntimeError("mpi4py not found. Install mpi4py.")

    elif cfg.package == "ccl":
         
        pass
    else:
        raise ValueError(f"Unknown package '{cfg.package}'.")





    if not isinstance(cfg.buffer_size, int) or cfg.buffer_size <= 0:
        raise ValueError(f"buffer_size must be a positive integer, got {cfg.buffer_size}")


#----------------------------------------------------------------------------

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
        raise NotImplementedError(f"No implementation for (package={cfg.package}, algorithm={cfg.algorithm})")



@hydra.main(config_path="config", config_name="config",version_base=None)
def main(cfg: DictConfig):
    validate_config(cfg)
    print("Parsed config:")
    print(cfg)
    run_communication(cfg)

if __name__ == "__main__":
    main()
