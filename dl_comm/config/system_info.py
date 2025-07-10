import os
import sys
import platform
import subprocess
import torch


def print_system_info(log, mpi_rank):
    
    if mpi_rank == 0:
        log.info("")
        log.info("[SYSTEM] Collecting environment information...")
        log.info("[SYSTEM] ======================================================")
        
        # PyTorch Information
        log.info(f"[SYSTEM] PyTorch version: {torch.__version__}")
        
        # CUDA Information
        if torch.cuda.is_available():
            log.info(f"[SYSTEM] CUDA used to build PyTorch: {torch.version.cuda}")
        else:
            log.info("[SYSTEM] CUDA used to build PyTorch: N/A")
        
        # System Information
        log.info("")
        log.info(f"[SYSTEM] OS: {platform.platform()}")
        
        # GCC version
        try:
            gcc_result = subprocess.run(['gcc', '--version'], capture_output=True, text=True, timeout=5)
            if gcc_result.returncode == 0:
                gcc_version = gcc_result.stdout.split('\n')[0]
                log.info(f"[SYSTEM] GCC version: {gcc_version}")
            else:
                log.info("[SYSTEM] GCC version: Could not collect")
        except:
            log.info("[SYSTEM] GCC version: Could not collect")
        
        # Python Information
        log.info("")
        log.info(f"[SYSTEM] Python version: {sys.version}")
        log.info(f"[SYSTEM] Python platform: {platform.platform()}")
        
        # GPU Information
        log.info(f"[SYSTEM] Is CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.info(f"[SYSTEM] CUDA runtime version: {torch.version.cuda}")
            log.info("[SYSTEM] GPU models and configuration:")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                log.info(f"[SYSTEM] GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            log.info("[SYSTEM] CUDA runtime version: N/A")
            log.info("[SYSTEM] GPU models and configuration: N/A")
        
        # Intel XPU Information
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            log.info(f"[SYSTEM] Is XPU available: {torch.xpu.is_available()}")
            log.info(f"[SYSTEM] XPU device count: {torch.xpu.device_count()}")
        
        # Distributed Backend Availability
        log.info("")
        log.info("[SYSTEM] Distributed Backend Availability:")
        
        # NCCL Backend
        try:
            import torch.distributed as dist
            nccl_available = dist.is_nccl_available()
            log.info(f"[SYSTEM] NCCL backend available: {nccl_available}")
        except AttributeError:
            log.info("[SYSTEM] NCCL backend available: Unknown (API not available)")
        except Exception:
            log.info("[SYSTEM] NCCL backend available: Error checking")
        
        # MPI Backend
        try:
            mpi_available = dist.is_mpi_available()
            log.info(f"[SYSTEM] MPI backend available: {mpi_available}")
        except AttributeError:
            log.info("[SYSTEM] MPI backend available: Unknown (API not available)")
        except Exception:
            log.info("[SYSTEM] MPI backend available: Error checking")
        
        # XCCL Backend (Intel CCL)
        try:
            from torch.distributed import distributed_c10d
            xccl_available = distributed_c10d.is_xccl_available()
            log.info(f"[SYSTEM] XCCL backend available: {xccl_available}")
        except AttributeError:
            log.info("[SYSTEM] XCCL backend available: Unknown (API not available)")
        except Exception:
            log.info("[SYSTEM] XCCL backend available: Error checking")
        
        # Library Versions
        log.info("")
        log.info("[SYSTEM] Versions of relevant libraries:")
        
        # NumPy
        try:
            import numpy as np
            log.info(f"[SYSTEM] numpy: {np.__version__}")
        except:
            log.info("[SYSTEM] numpy: Not available")
        
        # MPI4py
        try:
            import mpi4py
            log.info(f"[SYSTEM] mpi4py: {mpi4py.__version__}")
        except:
            log.info("[SYSTEM] mpi4py: Not available")
        
        # Hydra
        try:
            import hydra
            log.info(f"[SYSTEM] hydra-core: {hydra.__version__}")
        except:
            log.info("[SYSTEM] hydra-core: Not available")
        
        # OmegaConf
        try:
            import omegaconf
            log.info(f"[SYSTEM] omegaconf: {omegaconf.__version__}")
        except:
            log.info("[SYSTEM] omegaconf: Not available")
        
        # CCL
        try:
            import oneccl_bindings_for_pytorch as ccl
            if hasattr(ccl, '__version__'):
                log.info(f"[SYSTEM] oneccl_bindings_for_pytorch: {ccl.__version__}")
            else:
                log.info("[SYSTEM] oneccl_bindings_for_pytorch: Available (version unknown)")
        except:
            log.info("[SYSTEM] oneccl_bindings_for_pytorch: Not available")
        
        # Intel Extension for PyTorch
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(ipex, '__version__'):
                log.info(f"[SYSTEM] intel_extension_for_pytorch: {ipex.__version__}")
            else:
                log.info("[SYSTEM] intel_extension_for_pytorch: Available (version unknown)")
        except:
            log.info("[SYSTEM] intel_extension_for_pytorch: Not available")
        
        # Environment Variables
        log.info("")
        log.info("[SYSTEM] Relevant Environment Variables:")
        env_vars = sorted([k for k in os.environ.keys() if 'CCL' in k or 'FI_' in k or 'CUDA' in k or 'XPU' in k])
        for var in env_vars:
            log.info(f"[SYSTEM] {var:<30} = {os.environ[var]}")
        
        log.info("[SYSTEM] ======================================================")
        log.info("")