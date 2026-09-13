"""Bandwidth arithmetic (fix #2).

See ``docs/fixes/02-bandwidth-group-size.md``.
"""

import pytest

from dl_comm.analysis.bandwidth import (
    algorithmic_bandwidth,
    bus_bandwidth,
    busbw_factor,
    group_size_for,
    summarize,
)


class Cfg:
    def __init__(self, nodes, devices):
        self.num_compute_nodes = nodes
        self.num_devices_per_node = devices


# ---------------------------------------------------------------------------
# group_size_for: the actual defect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nodes,devices,mode,expected", [
    (1024, 4, "across_node", 1024),   # spans nodes -> group size is node count
    (1024, 4, "within_node", 4),      # stays on node -> group size is device count
    (1024, 4, "flatview", 4096),      # everything
    (8192, 6, "across_node", 8192),
    (8192, 6, "within_node", 6),
    (8192, 6, "flatview", 49152),
    (1, 12, "within_node", 12),
    (2, 4, "across_node", 2),
])
def test_group_size_matches_topology(nodes, devices, mode, expected):
    assert group_size_for(mode, Cfg(nodes, devices)) == expected


def test_across_node_group_size_is_not_devices_per_node():
    """The original live code returned devices-per-node for across_node.

    At the 13_for_paper geometry that is wrong by exactly nodes/devices.
    """
    cfg = Cfg(1024, 4)
    correct = group_size_for("across_node", cfg)
    old_buggy = cfg.num_devices_per_node
    assert correct == 1024
    assert correct / old_buggy == 256.0


# ---------------------------------------------------------------------------
# algbw / busbw
# ---------------------------------------------------------------------------

def test_algbw_is_size_over_time():
    assert algorithmic_bandwidth(10_485_760, 0.01) == pytest.approx(1.048576e9)


def test_algbw_rejects_nonpositive_time():
    assert algorithmic_bandwidth(1024, 0.0) == 0.0
    assert algorithmic_bandwidth(1024, -1.0) == 0.0


@pytest.mark.parametrize("collective,n,expected", [
    ("allreduce", 4, 2 * 3 / 4),
    ("allreduce", 1024, 2 * 1023 / 1024),
    ("reduce", 4, 1.0),
    ("broadcast", 4, 1.0),
    ("allgather", 4, 3 / 4),
    ("reducescatter", 4, 3 / 4),
    ("alltoall", 4, 3 / 4),
])
def test_busbw_factors_match_nccl_convention(collective, n, expected):
    assert busbw_factor(collective, n) == pytest.approx(expected)


def test_busbw_single_rank_has_no_scaling():
    """With one rank there is no inter-rank traffic to scale."""
    assert busbw_factor("allreduce", 1) == pytest.approx(1.0)
    assert busbw_factor("allgather", 1) == pytest.approx(1.0)


def test_allreduce_busbw_at_paper_geometry():
    """Concrete numbers quoted in the fix document."""
    size, t, n = 10_485_760, 0.01, 1024
    algbw = algorithmic_bandwidth(size, t)
    busbw = bus_bandwidth(size, t, n, "allreduce")
    assert algbw == pytest.approx(1.048576e9)
    assert busbw == pytest.approx(algbw * 2 * 1023 / 1024)


# ---------------------------------------------------------------------------
# summarize: iteration statistics (fix #5)
# ---------------------------------------------------------------------------

def test_summarize_reports_distribution_not_just_mean():
    times = [0.000731, 0.00117, 0.00118, 0.00116, 0.00119, 0.00117]
    stats = summarize(times)
    # iteration 0 is reported separately, not folded into the summary
    assert stats["first"] == pytest.approx(0.000731)
    assert stats["count"] == 5
    assert stats["min"] == pytest.approx(0.00116)
    assert stats["max"] == pytest.approx(0.00119)
    assert stats["stddev"] > 0
    assert "p99" in stats and "median" in stats


def test_summarize_excludes_first_iteration_outlier():
    """The 37%-low iteration 0 must not contaminate steady-state numbers."""
    settled = [0.00117] * 20
    with_outlier = [0.000731] + settled
    stats = summarize(with_outlier)
    assert stats["first"] == pytest.approx(0.000731)
    assert stats["min"] == pytest.approx(0.00117)
    assert stats["mean"] == pytest.approx(0.00117)
    assert stats["median"] == pytest.approx(summarize(settled, drop_first=False)["median"])


def test_summarize_keeps_first_when_asked():
    stats = summarize([0.000731, 0.00117, 0.00117], drop_first=False)
    assert stats["count"] == 3
    assert stats["min"] == pytest.approx(0.000731)


def test_summarize_handles_empty_and_single():
    assert summarize([]) is None
    single = summarize([0.5])
    assert single["count"] == 1
    assert single["median"] == pytest.approx(0.5)
    assert single["stddev"] == pytest.approx(0.0)
