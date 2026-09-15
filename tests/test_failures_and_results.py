"""Failure accounting and structured results (fix #4).

See ``docs/fixes/04-fail-loudly.md``.
"""

import json
import os

import pytest

from dl_comm.verify import failures
from dl_comm.analysis.results import build_results, write_results, write_csv, config_hash


@pytest.fixture(autouse=True)
def clean_tally():
    failures.reset()
    yield
    failures.reset()


def test_clean_run_reports_pass():
    failures.record_pass()
    failures.record_pass()
    snap = failures.snapshot()
    assert snap["checks"] == 2
    assert snap["failures"] == 0
    assert failures.failure_count() == 0


def test_failure_is_recorded_and_flips_verdict():
    failures.record_pass()
    failures.record_failure("allreduce/sum rank 3: expected 10.0 got 1.0")
    snap = failures.snapshot()
    assert snap["failures"] == 1
    assert failures.failure_count() == 1
    assert "allreduce/sum" in snap["details"][0]


def test_details_are_bounded():
    """A 8192-rank failure must not produce an unbounded detail list."""
    for i in range(5000):
        failures.record_failure(f"failure {i}")
    snap = failures.snapshot()
    assert snap["failures"] == 5000, "the COUNT must stay exact"
    assert len(snap["details"]) <= failures.MAX_DETAILS, "detail list must be capped"


def test_skips_are_tracked_separately():
    failures.record_skip("unsupported dtype")
    snap = failures.snapshot()
    assert snap["skipped"] == 1
    assert snap["failures"] == 0


def test_reset_clears_state():
    failures.record_failure("x")
    failures.reset()
    assert failures.snapshot()["failures"] == 0
    assert failures.failure_count() == 0


# ---------------------------------------------------------------------------
# results.json
# ---------------------------------------------------------------------------

class Cfg:
    framework = "pytorch"
    verify_correctness = True


def test_results_document_carries_verdict_and_provenance(tmp_path):
    doc = build_results(
        cfg=Cfg(), mpi_size=8, comm_mode="within_node", collective_name="allreduce",
        measurements=[{"label": "(Within-Group-0)", "group_size": 4,
                       "time_stats_s": {"median": 0.001, "count": 10}}],
        correctness={"passed": False, "total_failures": 2, "total_checks": 10})

    assert doc["correctness"]["passed"] is False
    assert doc["run"]["mpi_size"] == 8
    assert "config_sha256" in doc["provenance"]
    assert "libraries" in doc["provenance"]
    assert doc["schema_version"] == 1

    path = tmp_path / "results.json"
    written = write_results(str(path), doc)
    assert written and os.path.exists(written)
    reloaded = json.loads(path.read_text())
    assert reloaded["correctness"]["total_failures"] == 2


def test_config_hash_is_stable_and_sensitive():
    class A:
        x = 1

    class B:
        x = 2

    assert config_hash(A()) == config_hash(A())
    assert config_hash(A()) != config_hash(B())


def test_csv_has_one_row_per_measurement(tmp_path):
    measurements = [
        {"label": "(Within-Group-0)", "collective": "allreduce", "group_size": 4,
         "buffer_bytes": 1024, "busbw_median_bytes_per_s": 2.0e9,
         "time_stats_s": {"median": 0.001, "min": 0.0009, "count": 10}},
        {"label": "(Within-Group-1)", "collective": "allreduce", "group_size": 4,
         "buffer_bytes": 1024, "busbw_median_bytes_per_s": 2.1e9,
         "time_stats_s": {"median": 0.0011, "min": 0.001, "count": 10}},
    ]
    path = tmp_path / "results.csv"
    assert write_csv(str(path), measurements)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3, "header + 2 rows"
    assert "t_median_s" in lines[0]
    assert "(Within-Group-0)" in lines[1]


def test_csv_with_no_measurements_is_a_noop(tmp_path):
    assert write_csv(str(tmp_path / "empty.csv"), []) is None
