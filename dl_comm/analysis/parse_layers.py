"""Parse per-layer benchmark output into comparable measurements.

Each layer writes a different format:

    OSU         whitespace columns: "<bytes> <latency_us>"
    C++ CCL     "LAYER=cpp_ccl ... OP=allreduce BYTES=... BUSBW=..."
    transfer    "LAYER=cpp PATTERN=h2d BYTES=... GBPS=..."
    PyTorch     DLcomm's own result records

This module turns all of them into `LayerMeasurement` objects so the
bottleneck analysis has one input type. Parsing is deliberately strict: an
unrecognised line is skipped rather than guessed at, and a line that looks
like a measurement but lacks a field raises, because a silently dropped
measurement turns into a "layer not measured" row that hides a real result.
"""

from __future__ import annotations

import re

from dl_comm.analysis.bandwidth import busbw_factor
from dl_comm.analysis.bottleneck import LayerMeasurement

# "LAYER=cpp_ccl BACKEND=xccl OP=allreduce BYTES=1048576 RANKS=12 ..."
_KV = re.compile(r"(\w+)=([^\s]+)")

# OSU writes a two-column table after a header block.
_OSU_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*$")

# Bus-bandwidth factors come from dl_comm.analysis.bandwidth.busbw_factor.
# An earlier draft of this module re-declared them locally and silently
# dropped sendrecv_async (factor 2.0, not 1.0) and alltoallsingle, which would
# have made C++ and Python numbers disagree for the same operation. There must
# be exactly one definition of these factors in the codebase.

# OSU binary names to the benchmark's canonical collective names.
OSU_ALIAS = {
    "osu_allreduce": "allreduce",
    "osu_allgather": "allgather",
    "osu_alltoall": "alltoall",
    "osu_bcast": "broadcast",
    "osu_reduce": "reduce",
    "osu_reduce_scatter": "reduce_scatter",
    "osu_barrier": "barrier",
    "osu_latency": "sendrecv",
    "osu_bw": "sendrecv",
}


def busbw_from_latency(collective: str, size_bytes: int, latency_s: float,
                       ranks: int) -> float | None:
    """Convert a latency measurement to bus bandwidth in bytes/second."""
    if latency_s <= 0 or ranks < 2:
        return None
    return (size_bytes / latency_s) * busbw_factor(collective, ranks)


def parse_kv_lines(text: str, layer: str, buffer: str = "device",
                   ) -> list[LayerMeasurement]:
    """Parse `LAYER=... OP=... BYTES=... BUSBW=...` records.

    Used for the C++ CCL layer and anything else emitting the same shape.
    """
    out: list[LayerMeasurement] = []
    for line in text.splitlines():
        if not line.startswith("LAYER="):
            continue
        kv = dict(_KV.findall(line))
        if kv.get("LAYER") != layer:
            continue
        op = kv.get("OP")
        if op is None:
            continue  # the header line carries no OP
        if "BYTES" not in kv or "BUSBW" not in kv:
            raise ValueError(f"measurement line missing fields: {line!r}")

        size = int(kv["BYTES"])
        busbw = float(kv["BUSBW"])
        ranks = int(kv.get("RANKS", 0))

        # A zero-byte or zero-bandwidth record is a real event (barrier, or a
        # failed measurement); keep it, but mark it unavailable so nothing
        # divides by it.
        out.append(LayerMeasurement(
            layer=layer, collective=op, size_bytes=size,
            busbw_bps=busbw if busbw > 0 else None,
            buffer=buffer, ranks=ranks,
            note="" if busbw > 0 else "zero/NA",
        ))
    return out


def parse_osu(text: str, binary: str, ranks: int,
              buffer: str = "host") -> list[LayerMeasurement]:
    """Parse an OSU collective benchmark table.

    OSU reports latency in microseconds against message size in bytes. The
    default buffer is host: OSU 7.1 has no SYCL or Level Zero support, so on
    Aurora it cannot allocate device memory. That default is what keeps the
    bottleneck analysis from comparing it against device-buffer layers.
    """
    collective = OSU_ALIAS.get(binary, binary)
    out: list[LayerMeasurement] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _OSU_ROW.match(line)
        if not m:
            continue
        size = int(m.group(1))
        latency_us = float(m.group(2))
        busbw = busbw_from_latency(collective, size, latency_us / 1e6, ranks)
        out.append(LayerMeasurement(
            layer="osu", collective=collective, size_bytes=size,
            busbw_bps=busbw, buffer=buffer, ranks=ranks,
            note="" if busbw else "unconvertible",
        ))
    return out


def parse_transfer(text: str, ranks: int) -> list[LayerMeasurement]:
    """Parse `LAYER=cpp PATTERN=h2d BYTES=... GBPS=...` transfer records.

    Transfer patterns are not collectives: there is no group traffic pattern,
    so bus bandwidth equals the measured rate. They are recorded as their own
    pseudo-collectives (h2d, d2h, d2d, bidirectional) and compared only
    against the same pattern at another layer.
    """
    out: list[LayerMeasurement] = []
    for line in text.splitlines():
        if not line.startswith("LAYER="):
            continue
        kv = dict(_KV.findall(line))
        pattern = kv.get("PATTERN")
        if pattern is None:
            continue  # device/header line
        if "BYTES" not in kv or "GBPS" not in kv:
            raise ValueError(f"transfer line missing fields: {line!r}")
        gbps = float(kv["GBPS"])
        # d2d never crosses PCIe, so it is labelled device-to-device rather
        # than sharing the "device" bucket with the host-crossing patterns.
        buffer = "d2d" if pattern == "d2d" else "pinned-host"
        out.append(LayerMeasurement(
            layer=kv.get("LAYER", "cpp"), collective=pattern,
            size_bytes=int(kv["BYTES"]),
            busbw_bps=gbps * 1e9 if gbps > 0 else None,
            buffer=buffer, ranks=int(kv.get("RANKS", ranks)),
            note="" if gbps > 0 else "zero/NA",
        ))
    return out


def group_by_op_size(
    measurements: list[LayerMeasurement],
) -> dict[tuple[str, int], list[LayerMeasurement]]:
    """Bucket measurements so each bucket is one collective at one size."""
    buckets: dict[tuple[str, int], list[LayerMeasurement]] = {}
    for m in measurements:
        buckets.setdefault((m.collective, m.size_bytes), []).append(m)
    return buckets
