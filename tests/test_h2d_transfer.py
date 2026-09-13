"""Tests for host-to-device transfer measurement.

Runs on CPU with a fake torch module: the arithmetic, the overflow-safety
property, and the error paths are all exercised without a device.
"""

import pytest

from dl_comm.transfer.h2d import TransferResult, format_table


def _res(direction="h2d", nbytes=1 << 30, world=24, times=(0.005,)):
    return TransferResult(direction=direction, pinned=True, nbytes=nbytes,
                          world_size=world, times_s=list(times))


def test_byte_count_does_not_overflow_at_scale():
    """The reference C code computes N_byte*world_size in `int`.

    For a 1 GiB buffer that wraps at world_size >= 2 and lands on exactly 0
    at 4, 12, and 24 ranks -- a silent 0 GB/s. Python ints are arbitrary
    precision, so the same arithmetic must stay exact.
    """
    r = _res(nbytes=1 << 30, world=24)
    assert r.total_bytes == (1 << 30) * 24 == 25_769_803_776
    # The value that would wrap to zero in int32 must not be zero here.
    assert r.total_bytes % (2 ** 32) == 0      # this is why C hits exactly 0
    assert r.total_bytes > 2 ** 31             # and why it must not be an int


def test_bidirectional_counts_both_buffers():
    uni = _res(direction="h2d", nbytes=1 << 20, world=2)
    bi = _res(direction="bidirectional", nbytes=1 << 20, world=2)
    assert bi.total_bytes == 2 * uni.total_bytes


def test_best_is_min_not_mean():
    r = _res(times=(0.010, 0.004, 0.020))
    assert r.best_s == 0.004
    assert r.median_s == 0.010
    # Best-of-N must exceed median bandwidth, never the reverse.
    assert r.bandwidth_bps("best") > r.bandwidth_bps("median")


def test_bandwidth_arithmetic_is_right():
    # 1 GiB across 1 rank in exactly 1 second.
    r = _res(nbytes=1 << 30, world=1, times=(1.0,))
    assert r.bandwidth_bps() == float(1 << 30)
    assert r.gbps() == pytest.approx(1.073741824)


def test_no_timings_raises_rather_than_reporting_zero():
    r = _res(times=())
    with pytest.raises(ValueError, match="no timings"):
        _ = r.best_s
    with pytest.raises(ValueError, match="no timings"):
        _ = r.median_s


def test_nonpositive_time_raises():
    """A zero elapsed time would render as infinite bandwidth."""
    r = _res(times=(0.0,))
    with pytest.raises(ValueError, match="non-positive elapsed time"):
        r.bandwidth_bps()


def test_format_table_shows_both_columns():
    out = format_table([_res(times=(0.004, 0.010))])
    assert "best GB/s" in out and "median GB/s" in out
    assert "pinned" in out


def test_measure_rejects_bad_input():
    from dl_comm.transfer import h2d

    class FakeTorch:
        int32 = "int32"

    with pytest.raises(ValueError, match="nbytes must be positive"):
        h2d.measure(FakeTorch(), "cpu", nbytes=0)
    with pytest.raises(ValueError, match="iterations must be"):
        h2d.measure(FakeTorch(), "cpu", nbytes=1024, iterations=0)
