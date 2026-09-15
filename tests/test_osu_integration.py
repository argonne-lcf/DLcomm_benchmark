"""Tests for the OSU integration.

Run on CPU with no OSU installed: discovery, parsing, unit conversion, and the
equivalence map are all exercised without touching a cluster.
"""

import os
from pathlib import Path

import pytest

from dl_comm.osu import runner as osu

# Real OSU 7.x output shapes. Copied from the documented format rather than
# invented, because a parser tested only against its own assumptions is not
# tested at all.
BW_OUTPUT = """# OSU MPI Bandwidth Test v7.1
# Size      Bandwidth (MB/s)
1                       3.45
1024                 1893.21
4194304             23012.88
"""

LAT_OUTPUT = """# OSU MPI Latency Test v7.1
# Size          Avg Latency(us)
0                       1.42
8                       1.51
4194304               412.77
"""

BARRIER_OUTPUT = """# OSU MPI Barrier Latency Test v7.1
# Avg Latency(us)
4.91
"""


def test_bandwidth_output_parses_size_and_metric():
    r = osu.parse_osu_output(BW_OUTPUT)
    assert r.metric == "bandwidth"
    assert r.value_at(4194304) == pytest.approx(23012.88)
    assert r.value_at(1024) == pytest.approx(1893.21)


def test_latency_output_parses_and_detects_metric():
    r = osu.parse_osu_output(LAT_OUTPUT)
    assert r.metric == "latency"
    assert r.value_at(0) == pytest.approx(1.42)
    assert r.value_at(4194304) == pytest.approx(412.77)


def test_barrier_single_value_is_recorded_at_size_zero():
    """Barrier has no size column; it must not silently parse as empty."""
    r = osu.parse_osu_output(BARRIER_OUTPUT)
    assert r.points, "barrier output produced no data points"
    assert r.value_at(0) == pytest.approx(4.91)


def test_bandwidth_converts_to_bytes_per_second():
    """OSU reports MB/s using 1e6, so 23012.88 MB/s is 2.301288e10 B/s."""
    r = osu.parse_osu_output(BW_OUTPUT)
    assert r.bytes_per_second_at(4194304) == pytest.approx(2.301288e10)


def test_latency_converts_to_bytes_per_second():
    """size / latency: 4 MiB in 412.77 us."""
    r = osu.parse_osu_output(LAT_OUTPUT)
    got = r.bytes_per_second_at(4194304)
    assert got == pytest.approx(4194304 / (412.77e-6), rel=1e-6)


def test_conversion_returns_none_for_missing_size():
    r = osu.parse_osu_output(BW_OUTPUT)
    assert r.bytes_per_second_at(999999) is None
    assert r.value_at(999999) is None


def test_every_dlcomm_collective_maps_to_a_real_osu_benchmark():
    """The map must not invent OSU binaries that do not exist upstream.

    OSU 7.1 ships these under c/mpi/collective/blocking and c/mpi/pt2pt.
    """
    real = {
        "osu_allreduce", "osu_allgather", "osu_alltoall", "osu_alltoallv",
        "osu_bcast", "osu_gather", "osu_scatter", "osu_reduce",
        "osu_reduce_scatter", "osu_barrier", "osu_bw", "osu_bibw",
    }
    for coll, bench in osu.OSU_EQUIVALENTS.items():
        assert bench in real, f"{coll} maps to '{bench}', not an OSU 7.1 binary"


def test_equivalence_map_covers_the_registry():
    """Every registered DLcomm collective is either mapped or knowingly absent."""
    from dl_comm.comm.collectives import COLLECTIVES, init_framework_constants

    init_framework_constants("pytorch")
    unmapped = sorted(set(COLLECTIVES) - set(osu.OSU_EQUIVALENTS))
    # Every registered collective currently has an OSU analogue. Asserting
    # emptiness means a newly added collective fails here instead of silently
    # going uncompared.
    assert unmapped == [], f"unmapped collectives have no OSU analogue: {unmapped}"

    # And the map must not reference collectives that do not exist.
    stale = sorted(set(osu.OSU_EQUIVALENTS) - set(COLLECTIVES))
    assert stale == [], f"OSU map references unregistered collectives: {stale}"


def test_unknown_collective_returns_none_not_a_guess():
    assert osu.equivalent_for("not_a_collective") is None
    assert osu.equivalent_for("ALLREDUCE") == "osu_allreduce"


def test_discover_raises_with_searched_paths(monkeypatch, tmp_path):
    """A missing OSU must raise, never be mistaken for a zero measurement."""
    monkeypatch.setenv("DLCOMM_OSU_DIR", str(tmp_path))
    monkeypatch.delenv("DLCOMM_OSU_BUILD", raising=False)
    monkeypatch.setattr(osu, "SITE_DIRS", ())
    monkeypatch.setattr(osu.shutil, "which", lambda _n: None)
    with pytest.raises(osu.OsuNotFound) as e:
        osu.discover("osu_allreduce")
    msg = str(e.value)
    assert "osu_allreduce" in msg
    assert str(tmp_path) in msg, "error must name where it looked"
    assert "DLCOMM_OSU_DIR" in msg, "error must name the override knob"


def test_discover_prefers_explicit_override(monkeypatch, tmp_path):
    fake = tmp_path / "osu_allreduce"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("DLCOMM_OSU_DIR", str(tmp_path))
    assert osu.discover("osu_allreduce") == str(fake)


def test_discover_finds_binary_one_level_down(monkeypatch, tmp_path):
    """Site layouts nest binaries under mpi/collective/."""
    sub = tmp_path / "collective"
    sub.mkdir()
    fake = sub / "osu_bcast"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("DLCOMM_OSU_DIR", str(tmp_path))
    assert osu.discover("osu_bcast") == str(fake)


def test_build_from_tarball_refuses_when_no_source(monkeypatch, tmp_path):
    monkeypatch.setattr(osu, "SITE_TARBALLS", ())
    with pytest.raises(osu.OsuNotFound):
        osu.build_from_tarball(str(tmp_path))
