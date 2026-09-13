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
    c = compare("sendrecv", 4194304, 1.0e10, BW)
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
        compare("sendrecv", 4194304, 1.0e10, BW),
        compare("allreduce", 4194304, 1.0e10, None),
    ]
    text = format_table(rows)
    assert "0.00x" not in text, "absent data must not render as a zero ratio"
    assert "no_reference" in text
    assert "0.50x" in text
