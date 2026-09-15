"""Tests for DLcomm-vs-OSU comparison."""

import pytest

from dl_comm.osu.compare import compare, format_table
from dl_comm.osu.runner import parse_osu_output

BW = parse_osu_output(
    "# OSU MPI Bandwidth Test v7.1\n"
    "# Size      Bandwidth (MB/s)\n"
    "4194304             20000.00\n"
)


def test_ratio_is_computed_from_normalised_units():
    """20000 MB/s == 2e10 B/s; a DLcomm 1e10 B/s is exactly half."""
    c = compare("sendrecv", 4194304, 1.0e10, BW, dlcomm_buffer='host', osu_buffer='host')
    assert c.status == "ok"
    assert c.osu_bps == pytest.approx(2.0e10)
    assert c.ratio == pytest.approx(0.5)


def test_missing_osu_size_is_no_reference_not_zero():
    """A missing reference must never read as a 0x ratio."""
    c = compare("sendrecv", 1024, 1.0e10, BW)
    assert c.status.startswith("no_reference_at_")
    assert c.osu_bps is None
    assert c.ratio is None


def test_collective_without_analogue_is_not_compared():
    c = compare("not_a_collective", 4194304, 1.0e10, BW)
    assert c.status == "no_equivalent"
    assert c.osu_benchmark is None
    assert c.ratio is None


def test_absent_osu_run_reports_no_reference():
    c = compare("allreduce", 4194304, 1.0e10, None)
    assert c.status == "no_reference"
    assert c.osu_benchmark == "osu_allreduce"
    assert c.ratio is None


def test_missing_dlcomm_measurement_is_reported():
    c = compare("sendrecv", 4194304, None, BW)
    assert c.status == "no_dlcomm_measurement"
    assert c.ratio is None


def test_table_renders_missing_values_as_dashes_not_zeros():
    rows = [
        compare("sendrecv", 4194304, 1.0e10, BW, dlcomm_buffer='host', osu_buffer='host'),
        compare("allreduce", 4194304, 1.0e10, None, dlcomm_buffer='host', osu_buffer='host'),
    ]
    text = format_table(rows)
    assert "0.00x" not in text, "absent data must not render as a zero ratio"
    assert "no_reference" in text
    assert "0.50x" in text

# --- cross-path suppression -------------------------------------------------
# OSU 7.1 has no SYCL support, so on Aurora its collectives run in host memory
# while DLcomm runs on XPU device buffers. Dividing one by the other produced
# ratios up to 78x that read as "DLcomm beats MPI" but actually compare
# GPU-direct against host memcpy.


class _FakeOsu:
    def __init__(self, bps):
        self._bps = bps

    def bytes_per_second_at(self, size):
        return self._bps


def test_cross_path_ratio_is_suppressed():
    c = compare("allreduce", 4194304, 1.370e10, _FakeOsu(4.576e8),
                dlcomm_buffer="device", osu_buffer="host")
    assert c.comparable is False
    assert c.ratio is None, "a device-vs-host ratio must not be reported"
    assert "cross_path" in c.format_row()
    assert "29.9" not in c.format_row()


def test_same_path_ratio_is_reported():
    c = compare("allreduce", 4194304, 1.0e10, _FakeOsu(5.0e9),
                dlcomm_buffer="host", osu_buffer="host")
    assert c.comparable is True
    assert c.ratio == 2.0
    assert "2.00x" in c.format_row()


def test_default_is_cross_path_on_aurora():
    """Defaults must be conservative: device vs host unless stated."""
    c = compare("allreduce", 4194304, 1.0e10, _FakeOsu(5.0e9))
    assert c.ratio is None
