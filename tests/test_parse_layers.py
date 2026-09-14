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


# --- transfer patterns -------------------------------------------------------

TRANSFER_OUT = """\
LAYER=cpp DEVICE=Intel(R) Data Center GPU Max 1550 RANKS=12 BUFFER_BYTES=1073741824
LAYER=cpp PATTERN=h2d BYTES=12884901888 TIME_NS=303971886 GBPS=42.3885
LAYER=cpp PATTERN=d2h BYTES=12884901888 TIME_NS=259728500 GBPS=49.6091
LAYER=cpp PATTERN=bidirectional BYTES=25769803776 TIME_NS=468429033 GBPS=55.0133
LAYER=cpp PATTERN=d2d BYTES=12884901888 TIME_NS=99999999 GBPS=128.5
"""


def test_transfer_header_skipped_patterns_kept():
    from dl_comm.analysis.parse_layers import parse_transfer
    ms = parse_transfer(TRANSFER_OUT, ranks=12)
    assert len(ms) == 4
    assert {m.collective for m in ms} == {"h2d", "d2h", "bidirectional", "d2d"}


def test_transfer_gbps_converted_to_bytes_per_second():
    from dl_comm.analysis.parse_layers import parse_transfer
    ms = parse_transfer(TRANSFER_OUT, ranks=12)
    h2d = next(m for m in ms if m.collective == "h2d")
    assert h2d.busbw_bps == pytest.approx(42.3885e9)


def test_d2d_is_a_separate_buffer_class_from_pcie_patterns():
    """d2d never crosses PCIe; bucketing it with h2d would invite a false ratio."""
    from dl_comm.analysis.parse_layers import parse_transfer
    ms = parse_transfer(TRANSFER_OUT, ranks=12)
    by = {m.collective: m.buffer for m in ms}
    assert by["d2d"] == "d2d"
    assert by["h2d"] == by["d2h"] == "pinned-host"
    assert by["d2d"] != by["h2d"]


def test_transfer_line_missing_gbps_raises():
    from dl_comm.analysis.parse_layers import parse_transfer
    with pytest.raises(ValueError, match="missing fields"):
        parse_transfer("LAYER=cpp PATTERN=h2d BYTES=123\n", ranks=12)


# --- busbw numerator convention (A-Bot-CELS review point 10) ---------------
# allgather moves buffer*ranks, not buffer. Getting this wrong understated
# allgather by exactly `ranks` and made it look like the slowest collective
# on the machine. These tests fail if the convention silently changes.

def test_traffic_bytes_expands_allgather_by_ranks():
    from dl_comm.analysis.bandwidth import traffic_bytes
    assert traffic_bytes("allgather", 1 << 20, 12) == (1 << 20) * 12
    assert traffic_bytes("reduce_scatter", 1 << 20, 12) == (1 << 20) * 12


def test_traffic_bytes_leaves_fixed_volume_collectives_alone():
    from dl_comm.analysis.bandwidth import traffic_bytes
    for op in ("allreduce", "broadcast", "reduce", "sendrecv", "alltoall"):
        assert traffic_bytes(op, 1 << 20, 12) == (1 << 20), op


def test_traffic_bytes_single_rank_is_identity():
    from dl_comm.analysis.bandwidth import traffic_bytes
    assert traffic_bytes("allgather", 4096, 1) == 4096


def test_allgather_busbw_is_ranks_times_larger_than_naive():
    """The fix must change allgather by exactly the rank count."""
    from dl_comm.analysis.bandwidth import bus_bandwidth, busbw_factor
    buf, t, n = 1 << 22, 0.0267129, 12
    naive = (buf / t) * busbw_factor("allgather", n)
    fixed = bus_bandwidth(buf, t, n, "allgather")
    assert abs(fixed / naive - n) < 1e-9


def test_osu_allgather_conversion_uses_total_volume():
    """OSU prints per-rank size; the converter must expand it."""
    from dl_comm.analysis.parse_layers import busbw_from_latency
    from dl_comm.analysis.bandwidth import busbw_factor
    size, lat_s, n = 1 << 22, 0.0730529, 12
    got = busbw_from_latency("allgather", size, lat_s, n)
    want = (size * n / lat_s) * busbw_factor("allgather", n)
    assert abs(got - want) < 1e-6
