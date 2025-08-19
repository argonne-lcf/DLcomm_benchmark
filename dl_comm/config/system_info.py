import os
import sys
import platform
import subprocess


def print_system_info(log, mpi_rank, framework):
    if mpi_rank != 0:
        return

    log.info("")
    log.info("[SYSTEM] Collecting environment information...")
    log.info("[SYSTEM] ======================================================")

    # ---------------------------------------------------------------------
    # Framework Information (NO device/backend probes)
    # ---------------------------------------------------------------------
    if framework == 'pytorch':
        try:
            import torch
            log.info(f"[SYSTEM] PyTorch version: {torch.__version__}")
            cuda_avail = torch.cuda.is_available()
            log.info(f"[SYSTEM] Is CUDA available: {cuda_avail}")
            if cuda_avail:
                try:
                    log.info(f"[SYSTEM] CUDA used to build PyTorch: {torch.version.cuda}")
                    log.info("[SYSTEM] GPU models and configuration:")
                    for i in range(torch.cuda.device_count()):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        log.info(f"[SYSTEM] GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                except Exception:
                    log.info("[SYSTEM] CUDA/GPU details: error collecting")
            else:
                log.info("[SYSTEM] CUDA runtime version: N/A")
                log.info("[SYSTEM] GPU models and configuration: N/A")
            # Optional Intel XPU info (best-effort)
            try:
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    log.info(f"[SYSTEM] Is XPU available: {torch.xpu.is_available()}")
                    log.info(f"[SYSTEM] XPU device count: {torch.xpu.device_count()}")
            except Exception:
                pass
        except Exception as e:
            log.info(f"[SYSTEM] PyTorch info: error collecting ({e})")

    elif framework == 'jax':
        # Only version and env hints. NO jax.devices(), NO backend calls.
        try:
            import jax  # safe for version
            log.info(f"[SYSTEM] JAX version: {jax.__version__}")
        except Exception as e:
            log.info(f"[SYSTEM] JAX version: unknown ({e})")

        backend_hint = os.environ.get("JAX_PLATFORMS", "").strip()
        log.info(f"[SYSTEM] JAX backend platform (hint): {backend_hint or 'unset'}")

        xla_flags = os.environ.get("XLA_FLAGS", "")
        if xla_flags:
            trimmed = (xla_flags[:200] + "...") if len(xla_flags) > 200 else xla_flags
            log.info(f"[SYSTEM] XLA_FLAGS: {trimmed}")
        else:
            log.info("[SYSTEM] XLA_FLAGS: unset")

        rocm_path = os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME", ""))
        log.info(f"[SYSTEM] ROCm path hint: {rocm_path or 'unset'}")

    # ---------------------------------------------------------------------
    # System Information
    # ---------------------------------------------------------------------
    log.info("")
    log.info(f"[SYSTEM] OS: {platform.platform()}")

    try:
        gcc_result = subprocess.run(['gcc', '--version'], capture_output=True, text=True, timeout=5)
        if gcc_result.returncode == 0:
            gcc_version = gcc_result.stdout.split('\n')[0]
            log.info(f"[SYSTEM] GCC version: {gcc_version}")
        else:
            log.info("[SYSTEM] GCC version: Could not collect")
    except Exception:
        log.info("[SYSTEM] GCC version: Could not collect")

    log.info("")
    log.info(f"[SYSTEM] Python version: {sys.version}")
    log.info(f"[SYSTEM] Python platform: {platform.platform()}")

    # ---------------------------------------------------------------------
    # Distributed Backend Availability (safe checks only)
    # ---------------------------------------------------------------------
    log.info("")
    log.info("[SYSTEM] Distributed Backend Availability:")

    if framework == 'pytorch':
        try:
            import torch.distributed as dist
            try:
                nccl_available = dist.is_nccl_available()
            except Exception:
                nccl_available = "Unknown"
            log.info(f"[SYSTEM] NCCL backend available: {nccl_available}")
        except Exception:
            log.info("[SYSTEM] NCCL backend available: Error checking")

        try:
            import torch.distributed as dist
            try:
                mpi_available = dist.is_mpi_available()
            except Exception:
                mpi_available = "Unknown"
            log.info(f"[SYSTEM] MPI backend available: {mpi_available}")
        except Exception:
            log.info("[SYSTEM] MPI backend available: Error checking")

        try:
            from torch.distributed import distributed_c10d
            xccl_available = distributed_c10d.is_xccl_available()
            log.info(f"[SYSTEM] XCCL backend available: {xccl_available}")
        except Exception:
            log.info("[SYSTEM] XCCL backend available: Unknown/Error")

    elif framework == 'jax':
        try:
            import jax.distributed as jd  # noqa: F401
            log.info("[SYSTEM] JAX distributed module import: OK")
        except Exception as e:
            log.info(f"[SYSTEM] JAX distributed module import: Failed ({e})")
        log.info("[SYSTEM] JAX collective comms: Skipping device checks (deferred)")

    # ---------------------------------------------------------------------
    # Library Versions
    # ---------------------------------------------------------------------
    log.info("")
    log.info("[SYSTEM] Versions of relevant libraries:")

    try:
        import numpy as np
        log.info(f"[SYSTEM] numpy: {np.__version__}")
    except Exception:
        log.info("[SYSTEM] numpy: Not available")

    try:
        import mpi4py
        log.info(f"[SYSTEM] mpi4py: {mpi4py.__version__}")
    except Exception:
        log.info("[SYSTEM] mpi4py: Not available")

    try:
        import hydra
        log.info(f"[SYSTEM] hydra-core: {hydra.__version__}")
    except Exception:
        log.info("[SYSTEM] hydra-core: Not available")

    try:
        import omegaconf
        log.info(f"[SYSTEM] omegaconf: {omegaconf.__version__}")
    except Exception:
        log.info("[SYSTEM] omegaconf: Not available")

    if framework == 'pytorch':
        try:
            import oneccl_bindings_for_pytorch as ccl
            ver = getattr(ccl, '__version__', None)
            log.info(f"[SYSTEM] oneccl_bindings_for_pytorch: {ver or 'Available (version unknown)'}")
        except Exception:
            log.info("[SYSTEM] oneccl_bindings_for_pytorch: Not available")

        try:
            import intel_extension_for_pytorch as ipex
            ver = getattr(ipex, '__version__', None)
            log.info(f"[SYSTEM] intel_extension_for_pytorch: {ver or 'Available (version unknown)'}")
        except Exception:
            log.info("[SYSTEM] intel_extension_for_pytorch: Not available")

        try:
            import torch
            version_tuple = torch.cuda.nccl.version()
            version_str = ".".join(map(str, version_tuple))
            log.info(f"[SYSTEM] NCCL version: {version_str}")
        except Exception:
            log.info("[SYSTEM] NCCL version: Not available")

    elif framework == 'jax':
        log.info("[SYSTEM] NCCL/RCCL: Used internally by JAX on GPU backends (version not exposed)")

    # ---------------------------------------------------------------------
    # Environment Variables
    # ---------------------------------------------------------------------
    log.info("")
    log.info("[SYSTEM] Relevant Environment Variables:")
    interesting_keys = ("CCL", "FI_", "CUDA", "XPU", "ROCR", "HIP", "HSA",
                        "NCCL", "RCCL", "JAX", "XLA", "GRPC", "ROCM",
                        "MASTER", "COORDINATOR", "WORLD", "RANK")
    env_vars = sorted([k for k in os.environ.keys()
                       if any(tag in k for tag in interesting_keys)])
    for var in env_vars:
        try:
            log.info(f"[SYSTEM] {var:<30} = {os.environ[var]}")
        except Exception:
            pass

    log.info("[SYSTEM] ======================================================")
    log.info("")
