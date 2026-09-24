"""Tests for cross-layer bottleneck analysis.

The central risk in this module is producing a confident-looking attribution
from measurements that were never comparable. Several tests below exist purely
to prevent that.
"""

import pytest

from dl_comm.analysis.bottleneck import (
    LAYER_ORDER,
    BottleneckReport,
    Gap,
    LayerMeasurement,
    analyse,
    format_report,
)


def m(layer, bps, buffer="device", note="", collective="allreduce",
      size=4194304):
    return LayerMeasurement(layer=layer, collective=collective,
                            size_bytes=size, busbw_bps=bps, buffer=buffer,
                            note=note)


# --- gap arithmetic ---------------------------------------------------------


def test_efficiency_and_overhead_are_complementary():
    g = Gap(lower="cpp_ccl", upper="torch_dist", collective="allreduce",
            size_bytes=1 << 22, lower_bps=100e9, upper_bps=62e9,
            component="PyTorch dispatch")
    assert g.efficiency == pytest.approx(0.62)
    assert g.overhead_pct == pytest.approx(38.0)
    assert not g.inverted


def test_gap_attributes_named_component():
    rep = analyse([m("cpp_ccl", 100e9), m("torch_dist", 62e9)])
    assert len(rep.gaps) == 1
    assert rep.gaps[0].component == "PyTorch dispatch + process group"


def test_dominant_gap_is_the_largest_real_loss():
    rep = analyse([
        m("cpp_ccl", 100e9),
        m("torch_dist", 95e9),
        m("torchcomms", 40e9),
    ])
    d = rep.dominant
    assert d is not None
    assert d.upper == "torchcomms"
    assert d.overhead_pct == pytest.approx(57.894, abs=0.01)


# --- comparability ----------------------------------------------------------


def test_different_buffers_are_never_divided():
    """The bug this module exists to prevent.

    OSU on host memory against DLcomm on device memory produced a 78x
    "speedup" in an earlier revision. No ratio may be computed here.
    """
    rep = analyse([
        m("osu", 5.8e7, buffer="host"),
        m("torch_dist", 4.5e9, buffer="device"),
    ])
    assert rep.gaps == []
    assert len(rep.incomparable) == 1
    assert "different memory" in rep.incomparable[0]
    out = format_report(rep)
    assert "78" not in out


def test_same_buffer_is_compared():
    rep = analyse([
        m("osu", 4.0e9, buffer="host"),
        m("torch_dist", 2.0e9, buffer="host"),
    ])
    assert len(rep.gaps) == 1
    assert rep.gaps[0].efficiency == pytest.approx(0.5)


def test_inverted_gap_is_flagged_not_hidden():
    rep = analyse([m("cpp_ccl", 50e9), m("torch_dist", 80e9)])
    assert rep.gaps[0].inverted
    assert "upper layer faster" in format_report(rep)
    # An inverted gap must not be chosen as the dominant cost.
    assert rep.dominant is None


# --- missing layers ---------------------------------------------------------


def test_missing_layer_is_reported_not_interpolated():
    rep = analyse([m("cpp_ccl", 100e9), m("torch_dist", 60e9)])
    joined = " ".join(rep.missing)
    assert "OSU / MPI" in joined
    assert "torchcomms" in joined
    assert "not measured" in joined


def test_unavailable_layer_carries_its_reason():
    rep = analyse([
        m("osu", None, note="no_sycl_support"),
        m("cpp_ccl", 100e9),
    ])
    assert any("no_sycl_support" in s for s in rep.missing)
    assert "no_sycl_support" in format_report(rep)


def test_gap_spans_the_hole_when_a_middle_layer_is_absent():
    """With cpp_ccl missing, osu->torch_dist must say so in the attribution."""
    rep = analyse([
        m("osu", 4.0e9, buffer="host"),
        m("torch_dist", 2.0e9, buffer="host"),
    ])
    assert rep.gaps[0].component == "MPI->PyTorch, CCL layer not measured"


# --- input validation -------------------------------------------------------


def test_mixed_collectives_rejected():
    with pytest.raises(ValueError, match="one collective"):
        analyse([m("osu", 1e9), m("cpp_ccl", 1e9, collective="alltoall")])


def test_mixed_sizes_rejected():
    with pytest.raises(ValueError, match="one collective"):
        analyse([m("osu", 1e9), m("cpp_ccl", 1e9, size=1024)])


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="no measurements"):
        analyse([])


# --- report shape -----------------------------------------------------------


def test_layer_order_is_stack_order():
    """Layers are ordered lowest to highest in the stack.

    Asserted as relative ordering rather than an exact tuple: the fabric
    layers (fi, nixl) were added below the collectives, and a new layer
    should not break this test when the ordering it checks still holds.
    """
    expected_below_to_above = ("fi", "nixl", "osu", "cpp_ccl", "torch_dist",
                               "torchcomms")
    for lower, upper in zip(expected_below_to_above,
                            expected_below_to_above[1:]):
        assert LAYER_ORDER.index(lower) < LAYER_ORDER.index(upper), (
            f"{lower} must sit below {upper} in LAYER_ORDER")


def test_report_lists_every_layer_even_when_absent():
    out = format_report(analyse([m("torch_dist", 1e9)]))
    for label in ("OSU / MPI", "C++ CCL", "torch.distributed", "torchcomms"):
        assert label in out


def test_zero_bandwidth_treated_as_unavailable():
    """A zero is a failed measurement, not a valid data point.

    pci.cpp printed exactly 0.00 GB/s at 12 ranks because of an int32
    overflow; that must never be divided into anything.
    """
    rep = analyse([m("cpp_ccl", 0.0), m("torch_dist", 5e9)])
    assert rep.gaps == []
    assert any("C++ CCL" in s for s in rep.missing)
