"""Tests for the fabric layers (libfabric / NIXL).

The risk these tests exist to prevent is the same one as in the collective
parsers, one layer down: reporting a confident bandwidth number for a transfer
that never touched the fabric. A NIXL READ that silently falls back to a host
memcpy still prints GB/s, so bandwidth alone proves nothing -- the NIC octet
counters and the byte-exact check are what make it a measurement.

The sample text below is copied from real Tara North job output (jobs
6925/6926), not composed for the test, so a format drift in the tools shows up
here as a failure.
"""

import pytest

from dl_comm.analysis.bottleneck import LAYER_LABEL, LAYER_ORDER, analyse
from dl_comm.analysis.parse_fabric import (
    parse_fi_info,
    parse_fi_pingpong,
    parse_nixl_transfer,
)

# --- real captured output ---------------------------------------------------

NIXL_DRAM = """\
[initiator r0 p0 x4820c7s6b1n0] warmup 0: 0.031s  (8.77 GB/s)
[initiator r0 p0 x4820c7s6b1n0] iter 0: 0.040s  6.72 GB/s
[initiator r0 p0 x4820c7s6b1n0] iter 1: 0.032s  8.39 GB/s
[initiator r0 p0 x4820c7s6b1n0] best  0.032s -> 8.39 GB/s
[initiator r0 p0 x4820c7s6b1n0] mean  0.035s -> 7.63 GB/s
[initiator r0 p0 x4820c7s6b1n0]   cxi0.rx             409,146,128 octets  (0.38 GiB)
[initiator r0 p0 x4820c7s6b1n0]   total rx: 1.52 GiB vs 1.50 GiB payload -> 1.02x
[initiator r0 p0 x4820c7s6b1n0]   RAILS CARRYING PAYLOAD: 4 (cxi0, cxi1, cxi2, cxi3)
[initiator r0 p0 x4820c7s6b1n0] verifying byte-exactness...
[initiator r0 p0 x4820c7s6b1n0] PASS: destination buffer is byte-exact.
[initiator r0 p0 x4820c7s6b1n0] COMPLETE.
"""

# A fallback: bandwidth looks fine, but the NIC never saw the bytes.
NIXL_FALLBACK = """\
[initiator r0 p0 host] best  0.010s -> 25.00 GB/s
[initiator r0 p0 host]   total rx: 0.00 GiB vs 1.50 GiB payload -> 0.00x
[initiator r0 p0 host]   RAILS CARRYING PAYLOAD: 0 ()
[initiator r0 p0 host] PASS: destination buffer is byte-exact.
"""

FI_INFO_CXI = """\
provider: cxi
    fabric: cxi
    domain: cxi0
    version: 2.3
provider: cxi
    domain: cxi1
provider: tcp
    domain: eth0
"""


# --- NIXL transfer ----------------------------------------------------------


def test_nixl_parses_best_and_mean_from_real_output():
    found, ev = parse_nixl_transfer(NIXL_DRAM, pattern="read_dram")
    by_op = {m.collective: m for m in found}
    assert by_op["read_dram"].busbw_bps == pytest.approx(8.39e9)
    assert by_op["read_dram_mean"].busbw_bps == pytest.approx(7.63e9)
    assert by_op["read_dram"].layer == "nixl"


def test_nixl_captures_fabric_evidence():
    _, ev = parse_nixl_transfer(NIXL_DRAM)
    assert ev.rails == 4
    assert ev.octet_ratio == pytest.approx(1.02)
    assert ev.byte_exact is True
    assert ev.direction == "rx"
    assert ev.carried_on_fabric is True


def test_silent_fallback_is_detectable_despite_good_bandwidth():
    """The whole point of the octet counters.

    This transfer reports 25 GB/s and passes byte-exactness, yet no bytes
    crossed the NIC. Bandwidth and correctness both look fine; only the
    counter ratio reveals it.
    """
    found, ev = parse_nixl_transfer(NIXL_FALLBACK)
    assert found[0].busbw_bps == pytest.approx(25e9)  # looks great
    assert ev.byte_exact is True                       # and is correct
    assert ev.carried_on_fabric is False               # but never hit the wire
    assert ev.rails == 0


def test_unverified_transfer_reports_no_bandwidth():
    """A transfer with no byte-exact check is unavailable, not fast."""
    text = "[initiator r0 p0 h] best  0.032s -> 8.39 GB/s\n"
    found, ev = parse_nixl_transfer(text)
    assert ev.byte_exact is None
    assert found[0].busbw_bps is None
    assert found[0].note == "unverified"
    assert not found[0].available


def test_failed_byte_exactness_reports_no_bandwidth():
    text = ("[initiator r0 p0 h] best  0.032s -> 8.39 GB/s\n"
            "[initiator r0 p0 h] FAIL: destination buffer differs\n")
    found, ev = parse_nixl_transfer(text)
    assert ev.byte_exact is False
    assert found[0].busbw_bps is None
    assert "FAILED" in found[0].note


def test_carried_on_fabric_is_unknown_without_a_sample():
    """Unknown must not collapse to False."""
    _, ev = parse_nixl_transfer(
        "[initiator r0 p0 h] best 0.03s -> 8.0 GB/s\n"
        "[initiator r0 p0 h] PASS: destination buffer is byte-exact.\n")
    assert ev.octet_ratio is None
    assert ev.carried_on_fabric is None


def test_vram_and_dram_stay_distinct():
    dram, _ = parse_nixl_transfer(NIXL_DRAM, pattern="read_dram",
                                  buffer="host")
    vram, _ = parse_nixl_transfer(NIXL_DRAM, pattern="read_vram",
                                  buffer="device")
    assert dram[0].collective != vram[0].collective
    assert dram[0].buffer != vram[0].buffer


# --- fi_pingpong ------------------------------------------------------------


def test_fi_pingpong_table_parses():
    text = ("bytes   iters   total   time    MB/sec\n"
            "64      1000    7.8k    0.02s   3.20\n"
            "1048576 100     100m    0.50s   2097.15\n")
    found = parse_fi_pingpong(text)
    assert len(found) == 2
    assert found[0].size_bytes == 64
    assert found[0].busbw_bps == pytest.approx(3.20e6)
    assert found[1].busbw_bps == pytest.approx(2097.15e6)
    assert all(m.layer == "fi" for m in found)


def test_fi_pingpong_defaults_to_host_buffers():
    """fi_pingpong uses host memory; mislabelling it as device would let the
    bottleneck analysis divide it against a GPU layer."""
    found = parse_fi_pingpong("64  1000  7.8k  0.02s  3.20\n")
    assert found[0].buffer == "host"


# --- fi_info ----------------------------------------------------------------


def test_fi_info_groups_domains_by_provider():
    got = parse_fi_info(FI_INFO_CXI)
    assert got["cxi"] == ["cxi0", "cxi1"]
    assert got["tcp"] == ["eth0"]


def test_fi_info_empty_means_no_provider():
    """On a login node with no Slingshot NICs `fi_info -p cxi` prints nothing.
    That is a real answer, not a parse failure."""
    assert parse_fi_info("") == {}


# --- integration with the layer stack ---------------------------------------


def test_fabric_layers_are_in_the_stack():
    assert LAYER_ORDER.index("fi") < LAYER_ORDER.index("nixl")
    assert LAYER_ORDER.index("nixl") < LAYER_ORDER.index("osu")
    assert LAYER_LABEL["fi"] and LAYER_LABEL["nixl"]


def test_fi_to_nixl_gap_is_attributed():
    from dl_comm.analysis.bottleneck import LayerMeasurement

    def m(layer, bps):
        return LayerMeasurement(layer=layer, collective="read_dram",
                                size_bytes=1 << 28, busbw_bps=bps,
                                buffer="host", ranks=2)

    rep = analyse([m("fi", 10e9), m("nixl", 8.39e9)])
    assert len(rep.gaps) == 1
    assert rep.gaps[0].component == "NIXL agent + memory registration"
    assert rep.gaps[0].efficiency == pytest.approx(0.839)


def test_host_and_device_fabric_results_are_not_divided():
    """Same guard as the collective layers, one level down."""
    from dl_comm.analysis.bottleneck import LayerMeasurement

    rep = analyse([
        LayerMeasurement(layer="fi", collective="read_dram",
                         size_bytes=1 << 28, busbw_bps=10e9, buffer="host"),
        LayerMeasurement(layer="nixl", collective="read_dram",
                         size_bytes=1 << 28, busbw_bps=16.2e9,
                         buffer="device"),
    ])
    assert rep.gaps == []
    assert any("different memory" in s for s in rep.incomparable)
