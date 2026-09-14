"""Tests for per-layer output parsing.

The risk here is a parser that silently drops or mangles a measurement. A
dropped row shows up downstream as "layer not measured", which looks like a
missing job rather than a parsing bug, so several tests below assert on counts
and on refusal to guess.
"""

import pytest

from dl_comm.analysis.bandwidth import busbw_factor
from dl_comm.analysis.parse_layers import (
    OSU_ALIAS,
    busbw_from_latency,
    group_by_op_size,
    parse_kv_lines,
    parse_osu,
)

CCL_OUT = """\
LAYER=cpp_ccl BACKEND=xccl RANKS=12 ITERS=20
LAYER=cpp_ccl BACKEND=xccl OP=allreduce BYTES=1048576 RANKS=12 T_MED=0.00190954 ALGBW=5.49125e+08 BUSBW=1.00673e+09
LAYER=cpp_ccl BACKEND=xccl OP=allgather BYTES=1048576 RANKS=12 T_MED=0.00558107 ALGBW=1.87881e+08 BUSBW=1.72224e+08
LAYER=cpp_ccl BACKEND=xccl OP=barrier BYTES=0 RANKS=12 T_MED=4.2992e-05 ALGBW=0 BUSBW=0
"""

OSU_OUT = """\
# OSU MPI Allreduce Latency Test v7.1
# Size       Avg Latency(us)
1048576           2304.25
2097152           4575.60
4194304           9166.51
"""


# --- key/value layer output --------------------------------------------------


def test_parses_every_measurement_row():
    ms = parse_kv_lines(CCL_OUT, "cpp_ccl")
    assert len(ms) == 3, "header line must be skipped, data rows must not be"
    assert {m.collective for m in ms} == {"allreduce", "allgather", "barrier"}


def test_header_line_without_op_is_skipped_not_raised():
    ms = parse_kv_lines("LAYER=cpp_ccl BACKEND=xccl RANKS=12 ITERS=20", "cpp_ccl")
    assert ms == []


def test_values_survive_scientific_notation():
    ms = parse_kv_lines(CCL_OUT, "cpp_ccl")
    ar = next(m for m in ms if m.collective == "allreduce")
    assert ar.busbw_bps == pytest.approx(1.00673e09)
    assert ar.size_bytes == 1048576
    assert ar.ranks == 12


def test_zero_bandwidth_marked_unavailable_not_dropped():
    """barrier reports BUSBW=0; it must survive as a row but never be divided."""
    ms = parse_kv_lines(CCL_OUT, "cpp_ccl")
    b = next(m for m in ms if m.collective == "barrier")
    assert b.busbw_bps is None
    assert not b.available
    assert b.note == "zero/NA"


def test_other_layers_are_not_claimed():
    mixed = CCL_OUT + "LAYER=cpp PATTERN=h2d BYTES=999 GBPS=42.0\n"
    assert len(parse_kv_lines(mixed, "cpp_ccl")) == 3


def test_malformed_measurement_raises_rather_than_skipping():
    bad = "LAYER=cpp_ccl OP=allreduce RANKS=12\n"  # no BYTES, no BUSBW
    with pytest.raises(ValueError, match="missing fields"):
        parse_kv_lines(bad, "cpp_ccl")


def test_default_buffer_is_device_for_ccl():
    assert all(m.buffer == "device" for m in parse_kv_lines(CCL_OUT, "cpp_ccl"))


# --- OSU ---------------------------------------------------------------------


def test_osu_rows_parsed_and_comments_ignored():
    ms = parse_osu(OSU_OUT, "osu_allreduce", ranks=12)
    assert len(ms) == 3
    assert [m.size_bytes for m in ms] == [1048576, 2097152, 4194304]


def test_osu_defaults_to_host_buffer():
    """OSU 7.1 has no SYCL; defaulting to device would enable a false ratio."""
    ms = parse_osu(OSU_OUT, "osu_allreduce", ranks=12)
    assert all(m.buffer == "host" for m in ms)


def test_osu_latency_converted_with_shared_busbw_factor():
    ms = parse_osu(OSU_OUT, "osu_allreduce", ranks=12)
    m = ms[-1]  # 4 MiB @ 9166.51 us
    expected = (4194304 / (9166.51 / 1e6)) * busbw_factor("allreduce", 12)
    assert m.busbw_bps == pytest.approx(expected)


def test_osu_binary_names_map_to_canonical_collectives():
    assert OSU_ALIAS["osu_bcast"] == "broadcast"
    assert OSU_ALIAS["osu_latency"] == "sendrecv"
    ms = parse_osu(OSU_OUT, "osu_bcast", ranks=12)
    assert all(m.collective == "broadcast" for m in ms)


def test_unknown_osu_binary_keeps_its_own_name():
    ms = parse_osu(OSU_OUT, "osu_something_new", ranks=12)
    assert ms[0].collective == "osu_something_new"


# --- conversion guards -------------------------------------------------------


def test_zero_latency_yields_none_not_infinity():
    assert busbw_from_latency("allreduce", 4194304, 0.0, 12) is None


def test_single_rank_yields_none():
    """busbw is undefined for one rank; returning a number would invent one."""
    assert busbw_from_latency("allreduce", 4194304, 0.001, 1) is None


def test_factor_matches_the_single_shared_definition():
    """Guards against this module re-declaring the factors and drifting."""
    import dl_comm.analysis.parse_layers as pl

    assert not hasattr(pl, "BUSBW_FACTOR"), (
        "busbw factors must come from bandwidth.busbw_factor only"
    )
    # sendrecv_async is 2.0; a local copy previously got this wrong.
    assert busbw_factor("sendrecv_async", 12) == 2.0


# --- grouping ----------------------------------------------------------------


def test_group_by_op_size_buckets_for_comparison():
    ms = parse_kv_lines(CCL_OUT, "cpp_ccl") + parse_osu(OSU_OUT, "osu_allreduce", 12)
    buckets = group_by_op_size(ms)
    key = ("allreduce", 1048576)
    assert key in buckets
    assert {m.layer for m in buckets[key]} == {"cpp_ccl", "osu"}
