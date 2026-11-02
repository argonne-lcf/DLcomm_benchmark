"""Correctness gates must dominate performance reporting.

Every test here encodes a defect found in independent review: code that
published a bandwidth number for a run that had announced its own payload
was wrong, or that drew a confident conclusion from no data at all.

These are regression tests for the gates, so each one is written to fail
if the corresponding gate is weakened.
"""
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILD_REPORT = REPO / "tools" / "build_report.py"


def _load_build_report(tmp_path):
    """Import build_report without executing its argv-driven main()."""
    src = BUILD_REPORT.read_text().replace("\nmain()\n", "\npass\n")
    boot = tmp_path / "boot"
    boot.mkdir(exist_ok=True)
    saved = sys.argv
    sys.argv = ["build_report.py", str(boot), str(tmp_path / "boot.md")]
    try:
        mod = types.ModuleType("build_report_under_test")
        exec(compile(src, str(BUILD_REPORT), "exec"), mod.__dict__)
    finally:
        sys.argv = saved
    return mod


# nccl-tests writes its verdict on a '#' comment line, which every data
# parser skips. A validation failure there must still suppress the row.
# A real nccl-tests data row, 13 fields in the real column order:
#   size count type redop root | time algbw busbw #wrong | time algbw busbw #wrong
# The out-of-place wrong-count is field 8 and the in-place one field 12.
# The earlier fixture put a time value in field 8, so it could not express
# "this row admits wrong elements" and the row-level check went untested.
NCCL_ROW = ("     1048576    262144     float     sum      -1"
            "    12.3   85.21   85.21")


class TestCorruptLogsYieldNoBandwidth:
    def test_nccl_failed_validation_is_suppressed(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "nccl_bad.txt"
        log.write_text(f"{NCCL_ROW}     99    12.4   85.00   85.00     99\n"
                       "# Out of bounds values : 99 FAILED\n")
        assert br.nccl_best(log) is None, (
            "a run that reported FAILED validation must contribute no "
            "bandwidth number")

    def test_nccl_clean_log_still_reports(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "nccl_ok.txt"
        log.write_text(f"{NCCL_ROW}      0    12.4   85.00   85.00      0\n"
                       "# Out of bounds values : 0 OK\n")
        best = br.nccl_best(log)
        assert best is not None, "the gate must not suppress a clean run"
        assert best[2] == pytest.approx(85.21)

    def test_torch_mismatch_is_suppressed(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "torch_bad.txt"
        log.write_text("1048576 262144 0.1 10.5 20.5\nMISMATCH at offset 4096\n")
        assert br.torch_best(log) is None

    def test_torch_clean_log_still_reports(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "torch_ok.txt"
        log.write_text("1048576 262144 0.1 10.5 20.5\n")
        assert br.torch_best(log) is not None

    def test_nixl_backend_error_is_suppressed(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "nixl_bad.txt"
        log.write_text(
            "---- size 1GiB (1073741824)\n"
            "[nixl_putget] transfer best 44000 us -> 24.00 GB/s | "
            "mean 43000 us -> 23.00 GB/s\n"
            "NIXL_ERR_BACKEND during postXferReq\n")
        assert br.nixl_best(log) == []

    def test_osu_check_failed_is_suppressed(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "osu_bad.txt"
        log.write_text("1048576 55.2\nCHECK FAILED\n")
        assert br.osu_last(log) is None


class TestRankCountIsRecorded:
    """Bandwidth at 4 ranks and 8 ranks are not the same quantity."""

    def test_ranks_read_from_nccl_header(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "nccl_ranks.txt"
        log.write_text(
            "# Rank  0 Group  0 Pid 1 on nodeA device  0 [0000:01:00] GH200\n"
            "# Rank  1 Group  0 Pid 2 on nodeA device  1 [0000:02:00] GH200\n"
            "# Rank  2 Group  0 Pid 3 on nodeB device  0 [0000:01:00] GH200\n"
            f"{NCCL_ROW}      0\n")
        assert br.log_ranks(log) == 3

    def test_ranks_read_from_explicit_marker(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "torch_ranks.txt"
        log.write_text("LAYER=torch_dist OP=allreduce RANKS=8 ALGBW=10\n")
        assert br.log_ranks(log) == 8

    def test_unknown_rank_count_is_none_not_a_guess(self, tmp_path):
        br = _load_build_report(tmp_path)
        log = tmp_path / "quiet.txt"
        log.write_text("1048576 262144 0.1 10.5 20.5\n")
        assert br.log_ranks(log) is None


class TestEmptyInputMakesNoClaims:
    def test_no_data_run_does_not_claim_a_clean_sweep(self, tmp_path):
        sweeps = tmp_path / "sweeps"
        sweeps.mkdir()
        out = tmp_path / "report.md"
        proc = subprocess.run(
            [sys.executable, str(BUILD_REPORT), str(sweeps), str(out)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        text = out.read_text()
        assert "NO DATA" in text
        assert "every cell produced a measurement" not in text

    def test_no_data_run_invents_no_figures(self, tmp_path):
        sweeps = tmp_path / "sweeps"
        sweeps.mkdir()
        out = tmp_path / "report.md"
        subprocess.run(
            [sys.executable, str(BUILD_REPORT), str(sweeps), str(out)],
            capture_output=True, text=True, check=True)
        text = out.read_text()
        # Figures from the campaign this tool was written against. With no
        # input parsed, none of them may appear.
        for invented in ("327.57", "22.81", "38.35"):
            assert invented not in text, (
                f"report quoted {invented} without measuring it")
