 

from omegaconf import DictConfig


def parse_buffer_size(size_str: str) -> int:
    
    s = size_str.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1024 * 1024 * 1024)
    elif s.endswith("MB"):
        return int(float(s[:-2]) * 1024 * 1024)
    elif s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    elif s.endswith("B"):
        return int(float(s[:-1]))
    else:
        raise ValueError(f"payload.size='{size_str}' has unknown format. Use '1GB', '1MB', '512KB' etc")


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

        # collective.scale_up_algorithm
        scale_up_algo = cfg.collective.scale_up_algorithm
        valid_algos = self.spec["algo"].get(collective, [])
        if scale_up_algo not in valid_algos:
            errors.append(
                f"Invalid scale_up_algorithm '{scale_up_algo}' for collective '{collective}'. Valid: {valid_algos}"
            )
            
        # collective.scale_out_algorithm  
        scale_out_algo = cfg.collective.scale_out_algorithm
        if scale_out_algo not in valid_algos:
            errors.append(
                f"Invalid scale_out_algorithm '{scale_out_algo}' for collective '{collective}'. Valid: {valid_algos}"
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

        # comm_group validation
        comm_group = cfg.collective.comm_group
        comm_mode = comm_group.mode
        valid_modes = ["within_node", "across_node", "combined", "flatview"]
        
        if comm_mode not in valid_modes:
            errors.append(f"Invalid comm_mode '{comm_mode}'. Valid: {valid_modes}")
        
        # Mode-specific validation
        if comm_mode == "within_node":
            if not hasattr(comm_group, 'within_node'):
                errors.append("comm_mode 'within_node' requires 'within_node' configuration")
            else:
                within_config = comm_group.within_node
                if not hasattr(within_config, 'num_gpus_per_node') or not hasattr(within_config, 'gpu_ids_per_node'):
                    errors.append("within_node config requires 'num_gpus_per_node' and 'gpu_ids_per_node'")
        
        elif comm_mode == "across_node":
            if not hasattr(comm_group, 'across_node'):
                errors.append("comm_mode 'across_node' requires 'across_node' configuration")
            else:
                across_config = comm_group.across_node
                if not hasattr(across_config, 'num_compute_nodes') or not hasattr(across_config, 'num_gpus_per_node') or not hasattr(across_config, 'gpu_ids_per_node'):
                    errors.append("across_node config requires 'num_compute_nodes', 'num_gpus_per_node' and 'gpu_ids_per_node'")
        
        elif comm_mode == "combined":
            if not hasattr(comm_group, 'combined'):
                errors.append("comm_mode 'combined' requires 'combined' configuration")
            else:
                combined_config = comm_group.combined
                if not hasattr(combined_config, 'within_node') or not hasattr(combined_config, 'across_node'):
                    errors.append("combined config requires both 'within_node' and 'across_node' sub-configurations")

        if errors:
            raise ValueError("ALL ERRORS:\n" + "\n".join(errors))

        return buffer_bytes