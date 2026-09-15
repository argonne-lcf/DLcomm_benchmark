"""Rank-topology validation (fix #3).

See ``docs/fixes/03-rank-topology-validation.md``.
"""

import pytest

from dl_comm.config.topology import (
    covered_ranks,
    expected_rank_count,
    validate_rank_topology,
)
from conftest import FakeModeCfg


class NullLog:
    def __init__(self):
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)

    def warning(self, msg):
        self.errors.append(msg)

    info = output = lambda self, msg="": None


def test_matching_geometry_passes():
    cfg = FakeModeCfg(num_compute_nodes=2, device_ids_per_node=[0, 1, 2, 3])
    log = NullLog()
    ok, messages = validate_rank_topology("within_node", cfg, mpi_size=8,
                                          mpi_rank=0, log=log)
    assert ok
    assert messages == []


def test_the_original_orphaned_rank_scenario_is_rejected():
    """2 nodes x 4 devices launched with -ppn 12 = 24 ranks.

    The original code placed 8 of 24 ranks and silently dropped 16.
    """
    cfg = FakeModeCfg(num_compute_nodes=2, device_ids_per_node=[0, 1, 2, 3])
    log = NullLog()
    ok, messages = validate_rank_topology("within_node", cfg, mpi_size=24,
                                          mpi_rank=0, log=log)
    assert not ok, "a 24-rank launch against an 8-rank config must fail"
    joined = " ".join(messages)
    assert "8" in joined and "24" in joined
    assert any("NO group" in m for m in messages)
    assert log.errors, "the failure must be logged on rank 0"


def test_orphaned_rank_list_is_accurate():
    """The reported orphan set must match the group-construction arithmetic."""
    placed = covered_ranks("within_node", num_compute_nodes=2,
                           devices_per_node=4, mpi_size=24)
    assert placed == {0, 1, 2, 3, 12, 13, 14, 15}
    orphaned = sorted(set(range(24)) - placed)
    assert orphaned == [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23]
    assert len(orphaned) == 16


def test_undersized_launch_is_rejected():
    cfg = FakeModeCfg(num_compute_nodes=4, device_ids_per_node=[0, 1, 2, 3])
    ok, messages = validate_rank_topology("within_node", cfg, mpi_size=8,
                                          mpi_rank=0, log=NullLog())
    assert not ok
    assert any("16" in m for m in messages)


def test_device_ids_inconsistent_with_count_is_rejected():
    cfg = FakeModeCfg(num_compute_nodes=1, device_ids_per_node=[0, 1],
                      num_devices_per_node=4)
    ok, messages = validate_rank_topology("within_node", cfg, mpi_size=2,
                                          mpi_rank=0, log=NullLog())
    assert not ok
    assert any("does not match" in m for m in messages)


def test_across_node_requires_exact_multiple():
    cfg = FakeModeCfg(num_compute_nodes=3, device_ids_per_node=[0, 1, 2, 3])
    ok, messages = validate_rank_topology("across_node", cfg, mpi_size=10,
                                          mpi_rank=0, log=NullLog())
    assert not ok
    assert any("multiple" in m for m in messages)


def test_launcher_env_mismatch_is_reported(monkeypatch):
    monkeypatch.setenv("PALS_LOCAL_SIZE", "12")
    cfg = FakeModeCfg(num_compute_nodes=2, device_ids_per_node=[0, 1, 2, 3])
    ok, messages = validate_rank_topology("within_node", cfg, mpi_size=8,
                                          mpi_rank=0, log=NullLog())
    assert not ok
    assert any("12 ranks per node" in m for m in messages)


def test_strict_false_downgrades_to_warning():
    cfg = FakeModeCfg(num_compute_nodes=2, device_ids_per_node=[0, 1, 2, 3])
    ok, messages = validate_rank_topology("within_node", cfg, mpi_size=24,
                                          mpi_rank=0, log=NullLog(), strict=False)
    assert ok
    assert messages, "the mismatch is still reported"


@pytest.mark.parametrize("nodes,devices", [(1, 12), (8, 4), (1024, 6)])
def test_expected_rank_count(nodes, devices):
    total, per_node = expected_rank_count(nodes, list(range(devices)))
    assert total == nodes * devices
    assert per_node == devices
