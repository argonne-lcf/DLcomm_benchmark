"""Failure must dominate when ranks disagree about correctness.

Review finding: `parse_nixl_transfer` decided byte-exactness by whichever
marker it happened to see first, so an 8-rank run with one corrupt rank
published a clean bandwidth number. A collective is correct only if every
participant verified it.
"""
import pytest

from dl_comm.analysis.parse_fabric import parse_nixl_transfer

BLOCK = """---- size 1GiB (1073741824)
[initiator r0 p0 x4820c7s6b1n0] best  0.042s -> 25.00 GB/s
[initiator r0 p0 x4820c7s6b1n0] mean  0.045s -> 24.00 GB/s
"""
PASS_LINE = "PASS: destination buffer is byte-exact\n"
FAIL_LINE = "MISMATCH at offset 4096 on rank 3\n"


def _parse(text, tmp_path=None, name="nixl.txt"):
    """Returns (measurements, evidence)."""
    return parse_nixl_transfer(text)


class TestFailureDominance:
    def test_fail_then_pass_is_a_failure(self, tmp_path):
        meas, ev = _parse(BLOCK + FAIL_LINE + PASS_LINE, tmp_path)
        assert ev.byte_exact is False

    def test_pass_then_fail_is_also_a_failure(self, tmp_path):
        """Order must not decide correctness."""
        meas, ev = _parse(BLOCK + PASS_LINE + FAIL_LINE, tmp_path)
        assert ev.byte_exact is False, (
            "a PASS from one rank masked a later MISMATCH from another")

    def test_all_pass_is_a_pass(self, tmp_path):
        meas, ev = _parse(BLOCK + PASS_LINE + PASS_LINE, tmp_path)
        assert ev.byte_exact is True

    def test_no_marker_is_unverified_not_passed(self, tmp_path):
        meas, ev = _parse(BLOCK, tmp_path)
        assert ev.byte_exact is None, (
            "absence of a failure marker is not evidence of correctness")


class TestCorrectnessGatesTheMeasurement:
    def test_verified_transfer_publishes_bandwidth(self, tmp_path):
        meas, ev = _parse(BLOCK + PASS_LINE, tmp_path)
        assert meas, "a verified transfer should report its number"

    def test_failed_transfer_publishes_no_bandwidth(self, tmp_path):
        """The row survives for provenance; the number must not."""
        meas, ev = _parse(BLOCK + PASS_LINE + FAIL_LINE, tmp_path)
        assert meas, "the attempt should still be recorded"
        assert all(m.busbw_bps is None for m in meas), (
            "a run with a corrupt rank published a bandwidth number")

    def test_unverified_transfer_publishes_no_bandwidth(self, tmp_path):
        meas, ev = _parse(BLOCK, tmp_path)
        assert all(m.busbw_bps is None for m in meas)

    def test_failure_reason_is_recorded(self, tmp_path):
        meas, ev = _parse(BLOCK + PASS_LINE + FAIL_LINE, tmp_path)
        blob = " ".join(m.note or "" for m in meas)
        assert "byte-exact" in blob.lower() or "fail" in blob.lower(), (
            "suppressing the number is not enough; the reason must survive")
