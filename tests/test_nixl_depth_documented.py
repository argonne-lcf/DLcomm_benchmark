"""The example-19 README must carry the corrected NIXL depth result.

The Oct-24 per-rank figure (2.8-5.5 GB/s) was a queue-depth artifact, not a
NIXL ceiling. The Oct-29 campaign (PBS 11012-11015) measured 23.99 GB/s at
depth 16 on the same link, 100.3% of the 23.92 GB/s raw-CXI reference.

Published numbers drift when someone edits prose. These tests pin the ones
that took a job campaign to establish, and pin the build guidance that makes
them reproducible.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "examples" / "19_nixl_remote_rdma" / "README.md"
BUILD_DIR = REPO / "tools" / "nixl_cxi_build"
BUILD_DOC = BUILD_DIR / "README.md"


def _strip_code_fences(text: str) -> str:
    """Assert on prose and tables, not on shell snippets that may quote values."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


@pytest.fixture(scope="module")
def readme() -> str:
    assert README.is_file(), f"missing {README}"
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def build_doc() -> str:
    assert BUILD_DOC.is_file(), f"missing {BUILD_DOC}"
    return BUILD_DOC.read_text(encoding="utf-8")


class TestDepthCurveIsPublished:
    # (depth label, GB/s) straight off the nixlprof2 job, 2 GB transfers.
    MEASURED = [("2", "21.054"), ("4", "22.765"), ("8", "23.599"),
                ("16", "23.696"), ("32", "23.961")]

    def test_every_measured_depth_row_is_present(self, readme):
        body = _strip_code_fences(readme)
        missing = [f"depth {d} -> {v}" for d, v in self.MEASURED if v not in body]
        assert not missing, f"depth curve rows absent from README: {missing}"

    def test_headline_number_and_percentage(self, readme):
        body = _strip_code_fences(readme)
        assert "23.99" in body, "the 23.99 GB/s headline result is missing"
        assert "100.3" in body, "the 100.3% of line figure is missing"

    def test_raw_cxi_reference_is_stated(self, readme):
        # A bandwidth number with no reference is not a measurement.
        assert "23.92" in _strip_code_fences(readme), \
            "the 23.92 GB/s raw-CXI reference must be stated"

    def test_depth_is_named_as_the_variable(self, readme):
        body = _strip_code_fences(readme).lower()
        assert "queue depth" in body, "README must name queue depth as the variable"


class TestStaleClaimIsMarkedSuperseded:
    def test_old_per_rank_figure_is_not_presented_as_current(self, readme):
        """The 2.8-5.5 GB/s row may stay as history, but must be flagged."""
        body = _strip_code_fences(readme)
        if "2.8 – 5.5" in body or "2.8 - 5.5" in body:
            assert "uperseded" in body, (
                "the Oct-24 per-rank figure is still published without a "
                "superseded marker pointing at the depth result"
            )

    def test_two_gb_transfer_ceiling_is_documented(self, readme):
        body = _strip_code_fences(readme)
        assert "2 GB" in body, "the 2 GB per-transfer ceiling must be documented"


class TestBuildGuidanceShips:
    def test_build_scripts_are_present_and_executable_shell(self):
        script = BUILD_DIR / "build_nixl_cxi.sh"
        patcher = BUILD_DIR / "patch_nixl_cray.py"
        assert script.is_file(), f"missing {script}"
        assert patcher.is_file(), f"missing {patcher}"
        assert script.read_text(encoding="utf-8").startswith("#!"), \
            "build script needs a shebang"

    def test_all_three_blockers_are_documented(self, build_doc):
        body = _strip_code_fences(build_doc)
        for marker in ("rocr", "distutils", "lfabric"):
            assert marker in body, f"blocker '{marker}' is not documented"

    def test_listed_is_not_usable_gate_is_stated(self, build_doc):
        """getPluginParams must be named as the gate before trusting a number."""
        assert "getPluginParams" in build_doc, (
            "the verification gate is missing: a backend being listed by "
            "get_plugin_list() does not mean it was built"
        )

    def test_tcp_fallback_symptom_is_quantified(self, build_doc):
        body = _strip_code_fences(build_doc)
        assert "0.37" in body, \
            "the silent TCP fallback number (0.37 GB/s) must be stated"

    def test_libfabric_path_trap_is_called_out(self, build_doc):
        body = _strip_code_fences(build_doc)
        assert "lib64" in body, "the lib vs lib64 trap must be explained"
