"""Rank-topology validation for DLcomm.

The original benchmark built communication groups from ``num_devices_per_node``
while striding by the same value, silently assuming that ranks-per-node equals
devices-per-node. When a job was launched with a different ``-ppn`` than the
config described, ranks that fell outside the pattern joined no group, ran no
collective, and were never reported -- the run still exited 0 with a clean
timing table.

This module makes that condition a hard, explicit failure.
See ``docs/fixes/03-rank-topology-validation.md``.
"""

from __future__ import annotations

import os

# Environment variables that expose ranks-per-node, in order of preference.
# PALS_* is the Aurora/PBS launcher (mpiexec from Intel MPI / PALS),
# MPI_LOCALNRANKS is MPICH, OMPI_* is OpenMPI, SLURM_* is srun.
_LOCAL_SIZE_VARS = (
    "PALS_LOCAL_SIZE",
    "MPI_LOCALNRANKS",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "SLURM_NTASKS_PER_NODE",
)


def detect_ranks_per_node():
    """Return ranks-per-node reported by the launcher, or ``None``."""
    for var in _LOCAL_SIZE_VARS:
        raw = os.environ.get(var)
        if raw:
            try:
                value = int(raw.split(",")[0])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value, var
    return None, None


def expected_rank_count(num_compute_nodes, device_ids_per_node,
                        num_devices_per_node=None):
    """Total ranks the configuration describes."""
    if device_ids_per_node is not None:
        devices = len(device_ids_per_node)
    elif num_devices_per_node is not None:
        devices = int(num_devices_per_node)
    else:
        raise ValueError("need device_ids_per_node or num_devices_per_node")
    return int(num_compute_nodes) * devices, devices


def covered_ranks(comm_mode, num_compute_nodes, devices_per_node, mpi_size):
    """Ranks that the group-construction pattern will actually place.

    Mirrors the arithmetic in :func:`dl_comm.comm.comm_setup.setup_communication_groups`
    so that the discrepancy can be reported precisely rather than inferred.

    Note the stride: the shipped code computes
    ``ranks_per_physical_node = mpi_size // num_compute_nodes`` and indexes
    ``node * ranks_per_physical_node + gpu_idx``. It strides by the ranks the
    *launcher* placed per node but only iterates ``device_ids_per_node`` entries,
    which is precisely why the surplus ranks on each node are skipped.
    """
    placed = set()
    n_nodes = int(num_compute_nodes)
    n_dev = int(devices_per_node)
    if n_nodes <= 0 or n_dev <= 0:
        return placed

    ranks_per_physical_node = mpi_size // n_nodes if n_nodes else 0

    if comm_mode in ("within_node", "across_node"):
        for node in range(n_nodes):
            for gpu_idx in range(n_dev):
                rank = node * ranks_per_physical_node + gpu_idx
                if 0 <= rank < mpi_size:
                    placed.add(rank)
    elif comm_mode == "flatview":
        placed = set(range(min(mpi_size, n_nodes * n_dev)))
    return placed


def validate_rank_topology(comm_mode, mode_cfg, mpi_size, mpi_rank, log,
                           strict=True):
    """Verify the launch geometry matches the configured geometry.

    Returns ``(ok, messages)``. When ``strict`` is False the mismatch is
    reported as a warning and ``ok`` stays True, which exists only to support
    deliberate oversubscription experiments; the default is a hard failure.
    """
    messages = []

    device_ids = getattr(mode_cfg, "device_ids_per_node", None)
    num_devices = getattr(mode_cfg, "num_devices_per_node", None)
    num_nodes = getattr(mode_cfg, "num_compute_nodes", None)
    if num_nodes is None or (device_ids is None and num_devices is None):
        return True, messages

    expected_total, devices_per_node = expected_rank_count(
        num_nodes, device_ids, num_devices)

    # 1. device_ids_per_node must agree with num_devices_per_node.
    if device_ids is not None and num_devices is not None:
        if len(device_ids) != int(num_devices):
            messages.append(
                f"num_devices_per_node ({num_devices}) does not match "
                f"len(device_ids_per_node) ({len(device_ids)}): {list(device_ids)}")

    # 2. The total rank count must match exactly.
    if expected_total != mpi_size:
        placed = covered_ranks(comm_mode, num_nodes, devices_per_node, mpi_size)
        orphaned = sorted(set(range(mpi_size)) - placed)
        preview = orphaned[:16]
        suffix = f" ... (+{len(orphaned) - len(preview)} more)" if len(orphaned) > len(preview) else ""
        messages.append(
            f"rank count mismatch: config describes {num_nodes} nodes x "
            f"{devices_per_node} devices = {expected_total} ranks, but "
            f"MPI_COMM_WORLD has {mpi_size}")
        if orphaned:
            messages.append(
                f"{len(orphaned)} of {mpi_size} ranks would join NO group and run "
                f"NO collective: {preview}{suffix}")
        if mpi_size > expected_total:
            messages.append(
                f"launch with -ppn {devices_per_node} across {num_nodes} nodes, "
                f"or set num_devices_per_node/device_ids_per_node to match the job")

    # 3. Cross-check against what the launcher reports, when available.
    local_size, source_var = detect_ranks_per_node()
    if local_size is not None and local_size != devices_per_node:
        messages.append(
            f"launcher reports {local_size} ranks per node ({source_var}) but "
            f"config uses {devices_per_node} devices per node")

    # 4. across_node requires an exact multiple; unlike within_node the group
    #    construction has no `rank < mpi_size` guard.
    if comm_mode == "across_node" and devices_per_node:
        if mpi_size % devices_per_node != 0:
            messages.append(
                f"across_node requires mpi_size ({mpi_size}) to be a multiple of "
                f"devices per node ({devices_per_node}); otherwise groups reference "
                f"ranks that do not exist")

    if not messages:
        return True, messages

    if mpi_rank == 0 and log is not None:
        for msg in messages:
            if strict:
                log.error(f"[VALIDATION][TOPOLOGY] {msg}")
            else:
                log.warning(f"[VALIDATION][TOPOLOGY] {msg}")

    return (not strict), messages


def assert_group_membership(comm_mode, group_ranks, mpi_size, mpi_rank, log):
    """Fail if a constructed group references a rank that does not exist."""
    invalid = [r for r in group_ranks if r >= mpi_size or r < 0]
    if invalid:
        if mpi_rank == 0 and log is not None:
            log.error(f"[VALIDATION][TOPOLOGY] {comm_mode}: group references "
                      f"nonexistent ranks {invalid} (mpi_size={mpi_size})")
        return False
    return True
