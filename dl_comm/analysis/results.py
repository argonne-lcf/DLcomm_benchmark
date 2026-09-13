"""Structured, machine-readable results for DLcomm.

The benchmark previously emitted only a human-formatted log, so downstream
plots were produced by scraping text and no provenance travelled with the
numbers. This module writes a single ``results.json`` carrying the measured
values, the run's correctness verdict, and enough provenance (git SHA, config
hash, launch geometry, library versions) to reproduce it.

See ``docs/fixes/04-fail-loudly.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _git_describe(repo_dir):
    """Return commit SHA and dirty flag for the checkout, or None."""
    try:
        sha = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10)
        return {
            "commit": sha.stdout.strip(),
            "dirty": bool(dirty.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def config_hash(cfg) -> str:
    """Stable SHA256 over the resolved configuration.

    Must not fall back to ``repr()``: the default object repr embeds the
    instance's memory address, so the same config hashed twice in one process
    produced different digests and the provenance field was worthless.
    """
    try:
        from omegaconf import OmegaConf
        text = OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:
        text = _stable_repr(cfg)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_repr(obj, _depth=0):
    """Deterministic textual form for plain objects, dicts and sequences."""
    if _depth > 8:
        return "..."
    if isinstance(obj, dict):
        items = sorted((str(k), _stable_repr(v, _depth + 1)) for k, v in obj.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(v, _depth + 1) for v in obj) + "]"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return repr(obj)
    attrs = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        attrs[name] = _stable_repr(value, _depth + 1)
    return "{" + ",".join(f"{k}:{v}" for k, v in sorted(attrs.items())) + "}"


def _library_versions():
    versions: dict = {"python": sys.version.split()[0]}
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["torch_backend_cuda"] = bool(torch.cuda.is_available())
        try:
            versions["torch_backend_xpu"] = bool(torch.xpu.is_available())
        except AttributeError:
            versions["torch_backend_xpu"] = False
    except ImportError:
        pass
    try:
        import mpi4py
        versions["mpi4py"] = mpi4py.__version__
    except ImportError:
        pass
    try:
        import oneccl_bindings_for_pytorch as ccl
        versions["oneccl_bindings"] = getattr(ccl, "__version__", "unknown")
    except ImportError:
        pass
    return versions


def _launch_environment():
    keys = (
        "PALS_LOCAL_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE",
        "SLURM_NTASKS_PER_NODE", "PBS_JOBID", "SLURM_JOB_ID", "PBS_NODEFILE",
        "CCL_OP_SYNC", "CCL_ALLREDUCE", "CCL_ALLGATHERV", "CCL_PROCESS_LAUNCHER",
        "FI_PROVIDER", "ZE_AFFINITY_MASK",
    )
    return {k: os.environ[k] for k in keys if k in os.environ}


def build_results(cfg, mpi_size, comm_mode, collective_name, measurements,
                  correctness, repo_dir=None, extra=None):
    """Assemble the results document."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git": _git_describe(repo_dir),
            "config_sha256": config_hash(cfg),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "libraries": _library_versions(),
            "launch_env": _launch_environment(),
        },
        "run": {
            "mpi_size": mpi_size,
            "comm_mode": comm_mode,
            "collective": collective_name,
            "framework": getattr(cfg, "framework", None),
        },
        "correctness": correctness,
        "measurements": measurements,
    }
    if extra:
        doc.update(extra)
    return doc


def write_results(path, document, log=None):
    """Write ``document`` as JSON. Returns the path, or None on failure."""
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(document, handle, indent=2, default=str)
            handle.write("\n")
        if log is not None:
            log.output(f"[RESULTS] Wrote {path}")
        return path
    except OSError as exc:
        if log is not None:
            log.warning(f"[RESULTS] Could not write {path}: {exc}")
        return None


def write_csv(path, measurements, log=None):
    """Write a flat CSV of per-group summary statistics alongside the JSON."""
    import csv
    if not measurements:
        return None
    columns = ["label", "collective", "comm_mode", "group_size", "buffer_bytes",
               "logging_rank", "t_min_s", "t_median_s", "t_mean_s", "t_stddev_s",
               "t_p99_s", "algbw_median_bytes_per_s", "busbw_median_bytes_per_s",
               "busbw_factor", "iterations"]
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for m in measurements:
                stats = m.get("time_stats_s") or {}
                writer.writerow({
                    "label": m.get("label"),
                    "collective": m.get("collective"),
                    "comm_mode": m.get("comm_mode"),
                    "group_size": m.get("group_size"),
                    "buffer_bytes": m.get("buffer_bytes"),
                    "logging_rank": m.get("logging_rank"),
                    "t_min_s": stats.get("min"),
                    "t_median_s": stats.get("median"),
                    "t_mean_s": stats.get("mean"),
                    "t_stddev_s": stats.get("stddev"),
                    "t_p99_s": stats.get("p99"),
                    "algbw_median_bytes_per_s": m.get("algbw_median_bytes_per_s"),
                    "busbw_median_bytes_per_s": m.get("busbw_median_bytes_per_s"),
                    "busbw_factor": m.get("busbw_factor"),
                    "iterations": stats.get("count"),
                })
        if log is not None:
            log.output(f"[RESULTS] Wrote {path}")
        return path
    except OSError as exc:
        if log is not None:
            log.warning(f"[RESULTS] Could not write {path}: {exc}")
        return None
