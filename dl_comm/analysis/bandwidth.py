"""Bandwidth accounting for DLcomm.

Two things were wrong in the original module and are corrected here
(see ``docs/fixes/02-bandwidth-group-size.md``):

1. The live code path derived the communicator size from
   ``num_devices_per_node`` for *every* mode. For ``across_node`` a group spans
   one device index across all nodes, so its size is ``num_compute_nodes``. At
   1024 nodes x 4 devices this scaled every reported number by 256x.

2. ``group_size * buffer / time`` is not a standard bandwidth metric. This
   module now reports the two conventional figures:

   * **algbw** = ``buffer_bytes / time`` -- algorithmic bandwidth, the rate at
     which the caller's buffer is processed.
   * **busbw** = ``algbw * factor(collective, n)`` -- bus bandwidth, which
     normalises out the collective's intrinsic traffic pattern so that results
     are comparable across collectives and scales.

   Factors follow the usual convention (as used by nccl-tests):

   ===================  ==========================
   collective           busbw factor
   ===================  ==========================
   allreduce            ``2 (n-1) / n``
   allgather            ``(n-1) / n``
   reducescatter        ``(n-1) / n``
   alltoall             ``(n-1) / n``
   alltoallsingle       ``(n-1) / n``
   reduce               ``1``
   broadcast            ``1``
   gather               ``(n-1) / n``
   scatter              ``(n-1) / n``
   ===================  ==========================
"""

import re

from dl_comm.config import parse_buffer_size
from dl_comm.timer.timer import TIMES


# --------------------------------------------------------------------------
# Group sizing
# --------------------------------------------------------------------------

def group_size_for(comm_mode, mode_cfg, mpi_size=None):
    """Number of ranks in one communicator of ``comm_mode``.

    ``flatview``     -> nodes * devices_per_node (one group spanning everything)
    ``within_node``  -> devices_per_node        (one group per node)
    ``across_node``  -> num_compute_nodes       (one group per device index)
    """
    if mode_cfg is None:
        return mpi_size if mpi_size else 1

    devices = getattr(mode_cfg, "num_devices_per_node", None)
    device_ids = getattr(mode_cfg, "device_ids_per_node", None)
    if devices is None and device_ids is not None:
        devices = len(device_ids)
    nodes = getattr(mode_cfg, "num_compute_nodes", None)

    if comm_mode == "flatview":
        if nodes and devices:
            return int(nodes) * int(devices)
    elif comm_mode == "within_node":
        if devices:
            return int(devices)
    elif comm_mode == "across_node":
        if nodes:
            return int(nodes)

    return mpi_size if mpi_size else 1


def _group_size_from_label(label, comm_mode, mode_cfg, mpi_size):
    """Infer group size from a timer label, falling back to the configured mode."""
    lowered = label.lower()
    if "flatview" in lowered:
        return group_size_for("flatview", mode_cfg, mpi_size)
    if "within-group" in lowered:
        return group_size_for("within_node", mode_cfg, mpi_size)
    if "across-group" in lowered:
        return group_size_for("across_node", mode_cfg, mpi_size)
    return group_size_for(comm_mode, mode_cfg, mpi_size)


# --------------------------------------------------------------------------
# Bandwidth formulas
# --------------------------------------------------------------------------

def busbw_factor(collective_name, n):
    """Traffic multiplier that converts algbw into bus bandwidth."""
    if not collective_name or n is None or n < 2:
        return 1.0
    name = collective_name.lower()
    if name == "allreduce":
        return 2.0 * (n - 1) / n
    if name in ("allgather", "reducescatter", "reduce_scatter",
                "alltoall", "alltoallsingle", "gather", "scatter"):
        return (n - 1) / n
    if name in ("reduce", "broadcast", "bcast"):
        return 1.0
    return 1.0


def algorithmic_bandwidth(buffer_size, time_seconds):
    """``buffer_bytes / time`` in bytes per second."""
    if not time_seconds or time_seconds <= 0:
        return 0.0
    return buffer_size / time_seconds


def bus_bandwidth(buffer_size, time_seconds, group_size, collective_name):
    return algorithmic_bandwidth(buffer_size, time_seconds) * busbw_factor(
        collective_name, group_size)


def calculate_group_bandwidth(group_size, buffer_size, time_seconds,
                              collective_name=None):
    """Backwards-compatible entry point.

    .. deprecated::
       The historical definition ``group_size * buffer / time`` is not a
       standard metric and over-reported by a factor of ``group_size``. It is
       retained only so existing callers keep working; new code should call
       :func:`algorithmic_bandwidth` or :func:`bus_bandwidth`.
    """
    if collective_name is not None:
        return bus_bandwidth(buffer_size, time_seconds, group_size, collective_name)
    if not time_seconds or time_seconds <= 0:
        return 0.0
    return (group_size * buffer_size) / time_seconds


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def summarize(values, drop_first=True):
    """Return min/median/mean/stddev/p99 for a list of per-iteration values.

    Iteration 0 is reported separately and excluded from the summary by
    default: it carries connection-establishment and lazy-allocation cost that
    is not part of steady-state performance. See
    ``docs/fixes/05-timing-and-statistics.md``.
    """
    if not values:
        return None
    vals = list(values)
    first = vals[0]
    body = vals[1:] if (drop_first and len(vals) > 1) else vals
    if not body:
        body = vals

    ordered = sorted(body)
    count = len(ordered)
    mean = sum(ordered) / count
    if count > 1:
        var = sum((v - mean) ** 2 for v in ordered) / (count - 1)
    else:
        var = 0.0
    median = (ordered[count // 2] if count % 2
              else (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0)
    p99_index = max(0, min(count - 1, int(round(0.99 * (count - 1)))))

    return {
        "count": count,
        "first": first,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "median": median,
        "stddev": var ** 0.5,
        "p99": ordered[p99_index],
        "dropped_first": bool(drop_first and len(vals) > 1),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _collect(all_data, buffer_configs, comm_mode, mode_cfg, mpi_size,
             collective_name):
    """Build ``{label: {...}}`` of per-iteration timings and bandwidths."""
    collected = {}
    for data in all_data:
        if not data:
            continue
        rank = data["rank"]
        for label, times_list in data["timers"].items():
            if any(k in label for k in ("import", "setup", "mxm", "Group Creation")):
                continue
            lowered = label.lower()
            if "flatview" in lowered:
                buffer_size = buffer_configs.get("flatview", 0)
            elif "within-group" in lowered:
                buffer_size = buffer_configs.get("within", 0)
            elif "across-group" in lowered:
                buffer_size = buffer_configs.get("across", 0)
            else:
                continue
            if label in collected:
                continue

            n = _group_size_from_label(label, comm_mode, mode_cfg, mpi_size)
            algbw = [algorithmic_bandwidth(buffer_size, t) for t in times_list]
            factor = busbw_factor(collective_name, n)
            busbw = [a * factor for a in algbw]

            collected[label] = {
                "rank": rank,
                "group_size": n,
                "buffer_size": buffer_size,
                "times": list(times_list),
                "algbw": algbw,
                "busbw": busbw,
                "busbw_factor": factor,
            }
    return collected


def _print_table(logger, collected, key, heading, fmt="{:.3e}"):
    labels = list(collected.keys())
    if not labels:
        return
    max_iters = max(len(collected[l][key]) for l in labels)
    col = 22

    logger.output(heading)
    line = f"{'Iteration':<12}" + "".join(f"{l:^{col}}" for l in labels)
    logger.output(line)
    logger.output(f"{'':12}" + "".join(
        f"{'LOGGING RANK - ' + str(collected[l]['rank']):^{col}}" for l in labels))
    logger.output("-" * len(line))
    for i in range(max_iters):
        row = f"{i:<12}"
        for l in labels:
            vals = collected[l][key]
            row += (fmt.format(vals[i]) if i < len(vals) else "-").center(col)
        logger.output(row)
    logger.output("-" * len(line))


def _print_summary(logger, collected, collective_name):
    logger.output("")
    logger.output("[BANDWIDTH] SUMMARY (iteration 0 excluded; time in s, bandwidth in bytes/s):")
    header = (f"{'group':<22}{'n':>7}{'bytes':>12}{'t_min':>12}{'t_med':>12}"
              f"{'t_mean':>12}{'t_std':>12}{'t_p99':>12}{'algbw_med':>14}{'busbw_med':>14}")
    logger.output(header)
    logger.output("-" * len(header))
    for label, info in collected.items():
        ts = summarize(info["times"])
        if not ts:
            continue
        # Median bandwidth is derived from the median time, not averaged over
        # per-iteration bandwidths, so it stays consistent with t_med.
        algbw_med = algorithmic_bandwidth(info["buffer_size"], ts["median"])
        busbw_med = algbw_med * info["busbw_factor"]
        logger.output(
            f"{label[:22]:<22}{info['group_size']:>7}{info['buffer_size']:>12}"
            f"{ts['min']:>12.6f}{ts['median']:>12.6f}{ts['mean']:>12.6f}"
            f"{ts['stddev']:>12.6f}{ts['p99']:>12.6f}"
            f"{algbw_med:>14.3e}{busbw_med:>14.3e}")
    logger.output("-" * len(header))
    if collective_name:
        any_info = next(iter(collected.values()), None)
        if any_info:
            logger.output(f"[BANDWIDTH] busbw factor for {collective_name.lower()} "
                          f"at n={any_info['group_size']}: {any_info['busbw_factor']:.6f}")


def gather_and_print_all_bandwidths(logger, cfg, mpi_size,
                                    ranks_responsible_for_logging,
                                    title="[BANDWIDTH]", adjusted_buffer_sizes=None,
                                    current_comm_mode=None, current_mode_cfg=None,
                                    collective_name=None, results_sink=None):
    from mpi4py import MPI

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    my_data = None
    if mpi_rank in ranks_responsible_for_logging:
        my_data = {"rank": mpi_rank, "timers": dict(TIMES)}
    all_data = MPI.COMM_WORLD.gather(my_data, root=0)

    if mpi_rank != 0:
        return None

    logger.output("")
    logger.output(f"{title} -------------------------------------------")

    if not adjusted_buffer_sizes:
        logger.warning("[BANDWIDTH] Cannot calculate bandwidth - no buffer size information provided")
        return None

    collected = _collect(all_data, adjusted_buffer_sizes, current_comm_mode,
                         current_mode_cfg, mpi_size, collective_name)
    if not collected:
        logger.output(f"{title} -------------------------------------------")
        return None

    name = (collective_name or "collective").upper()
    logger.output("")
    _print_table(logger, collected, "algbw",
                 f"[BANDWIDTH] ALGBW TABLE FOR {name} (bytes/s, = buffer/time):")
    logger.output("")
    _print_table(logger, collected, "busbw",
                 f"[BANDWIDTH] BUSBW TABLE FOR {name} (bytes/s, = algbw x factor):")
    _print_summary(logger, collected, collective_name)
    logger.output("")
    logger.output(f"{title} -------------------------------------------")

    if results_sink is not None:
        for label, info in collected.items():
            ts = summarize(info["times"])
            algbw_med = algorithmic_bandwidth(info["buffer_size"], ts["median"]) if ts else 0.0
            results_sink.append({
                "label": label,
                "collective": collective_name,
                "comm_mode": current_comm_mode,
                "group_size": info["group_size"],
                "buffer_bytes": info["buffer_size"],
                "logging_rank": info["rank"],
                "times_s": info["times"],
                "time_stats_s": ts,
                "busbw_factor": info["busbw_factor"],
                "algbw_median_bytes_per_s": algbw_med,
                "busbw_median_bytes_per_s": algbw_med * info["busbw_factor"],
            })
    return collected


def print_all_bandwidths(logger, cfg, mpi_size, ranks_responsible_for_logging,
                         phase_filter=None, adjusted_buffer_sizes=None,
                         current_comm_mode=None, current_mode_cfg=None,
                         collective_name=None):
    """Deprecated alias kept for backwards compatibility.

    The original second implementation duplicated the reporting logic with a
    different (and inconsistent) group-size rule. It now forwards to
    :func:`gather_and_print_all_bandwidths` so there is exactly one definition
    of bandwidth in the package.
    """
    return gather_and_print_all_bandwidths(
        logger, cfg, mpi_size, ranks_responsible_for_logging,
        title="[BANDWIDTH]", adjusted_buffer_sizes=adjusted_buffer_sizes,
        current_comm_mode=current_comm_mode, current_mode_cfg=current_mode_cfg,
        collective_name=collective_name)
