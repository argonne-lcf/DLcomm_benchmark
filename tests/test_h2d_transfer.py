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

# --- d2d (device-to-device) -------------------------------------------------
# d2d never crosses PCIe: it measures on-device HBM bandwidth and is the
# ceiling against which h2d/d2h should be read.


def test_d2d_is_measured_and_does_not_double_count():
    """d2d moves one buffer per rank, like h2d -- not two like bidirectional."""
    r = TransferResult(direction="d2d", pinned=False, nbytes=1 << 30,
                       world_size=12, times_s=[0.01])
    assert r.total_bytes == (1 << 30) * 12


def test_d2d_in_default_directions():
    import inspect
    from dl_comm.transfer.h2d import measure
    default = inspect.signature(measure).parameters["directions"].default
    assert "d2d" in default, "goal requires h2d, d2d and d2h"
    assert "h2d" in default and "d2h" in default


def test_unknown_direction_names_d2d_in_error():
    """The error text must list d2d so a typo points at the real options."""
    import inspect

    from dl_comm.transfer import h2d as mod
    src = inspect.getsource(mod.measure)
    assert "expected h2d, d2h, d2d, or bidirectional" in src

def test_fill_does_not_use_randperm():
    """randperm at 2^28 elements hung the PyTorch layer of job 8824725.

    It is O(n) single-threaded on host tensors; the fill must stay parallel.
    """
    import inspect

    from dl_comm.transfer import h2d as mod
    src = inspect.getsource(mod.fill_shuffled)
    # Strip the docstring: it legitimately mentions randperm to explain why
    # the function avoids it.
    doc = mod.fill_shuffled.__doc__ or ""
    code = src.replace(doc, "")
    assert "randperm" not in code, "randperm hangs at 1 GiB; use random_"
    assert "random_" in code
