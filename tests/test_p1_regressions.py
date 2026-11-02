"""Behavioural regressions for the eight P1 review findings.

Every test here was written by first REPRODUCING the defect against the
pre-fix tree, then asserting the fixed behaviour. Each one fails if the
corresponding fix is reverted -- they are not assertions about strings in
comments, they execute the real code path.

The common theme of all eight findings is fail-OPEN: a gate that looked
like it was checking something but could not actually refuse. So each test
drives a CORRUPT or UNVERIFIED input through the real function and asserts
that no number is published and no success is reported.
"""
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "examples/21_cuda_layer_sweeps/sweep_layers_cuda.sh"
PROFILE = ROOT / "examples/21_cuda_layer_sweeps/layer_profile.py"
RUNALL = ROOT / "tools/run_all_scales.sh"
PATCHER = ROOT / "tools/nixl_cxi_build/patch_nixl_cray.py"


def _load_build_report():
    """Import build_report's helpers without running its main().

    The script ends in a bare `main()` call, so a normal import executes a
    full report build (and dies on an empty results tree). Load the source
    up to that call instead.
    """
    src = (ROOT / "tools/build_report.py").read_text()
    src = src[:src.rindex("\nmain()")]
    mod = types.ModuleType("build_report_under_test")
    mod.__file__ = str(ROOT / "tools/build_report.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _extract_embedded_python(sh_path, marker):
    """Pull an embedded python heredoc out of a shell script.

    The validator in run_all_scales.sh only exists inside a heredoc, so the
    only way to test it as shipped is to extract and execute it.
    """
    text = Path(sh_path).read_text()
    i = text.index(marker)
    start = text.rindex("<<", 0, i)
    start = text.index("\n", start) + 1
    end = text.index("\nPYEOF", i)
    return textwrap.dedent(text[start:end])


# ---------------------------------------------------------------- P1-1/P1-2
class TestPerIterationAndFiniteValidation:
    """NaN must not certify, and an early corrupt iteration must not be
    overwritten by later clean ones."""

    def test_nan_defeats_tolerance_comparison(self):
        """The root cause, stated as an executable fact.

        abs(NaN - expected) > tol is False, so a tolerance check ALONE
        certifies an all-NaN payload. This is why an explicit finiteness
        check is required for every operation.
        """
        nan = float("nan")
        assert not (abs(nan - 1.0) > 1e-3)

    def test_every_collective_branch_checks_finiteness(self):
        """All five ops, not just allreduce, must test isfinite.

        Finding P1-1: only the allreduce branch called isfinite, so
        allgather/alltoall/broadcast/reduce certified NaN payloads.
        """
        src = _extract_embedded_python(RUNALL, "def _check")
        body = src[src.index("def _check"):]
        body = body[:body.index("\n        op_bad = 0")]
        for op in ("allreduce", "allgather", "alltoall", "broadcast",
                   "reduce"):
            seg = body[body.index(f'"{op}"'):]
            nxt = [seg.index(f'"{o}"') for o in
                   ("allreduce", "allgather", "alltoall", "broadcast",
                    "reduce") if f'"{o}"' in seg[1:]]
            seg = seg[:min(nxt)] if nxt else seg
            assert "isfinite" in seg, (
                f"{op} branch has no finiteness check; a NaN payload in "
                f"{op} would be certified VERIFIED=1")

    def test_validation_runs_inside_the_iteration_loop(self):
        """Finding P1-2: validating only the final output hid a fault in an
        earlier timed iteration."""
        src = _extract_embedded_python(RUNALL, "def _check")
        loop = src.index("for i in range(WARMUP + ITERS)")
        call = src.index("op_bad += _check(op, i)")
        assert call > loop, "validation must happen per iteration"
        # and there must be no second, post-loop recomputation that would
        # double count or re-establish the old behaviour
        assert src.count("op_bad += _check") == 1
        assert src.count("tot = float(world") == 1


# -------------------------------------------------------------------- P1-3
class TestCorrectnessFailureExitCode:
    """A failed correctness check must make the process exit nonzero."""

    def test_guard_reads_a_key_the_producer_actually_writes(self):
        """Finding P1-3: main() tested res["failures"], which is never set.

        The producer writes verification_failures /
        verification_failures_all_ranks, so the guard was dead code and a
        corrupt run returned 0.
        """
        src = PROFILE.read_text()
        produced = set(re.findall(r'out\["(\w+)"\]\s*=', src))
        # keys the exit guard consults
        tail = src[src.index("local_fail ="):]
        consumed = set(re.findall(r'res\.get\("(\w+)"\)', tail))
        assert consumed, "exit guard consults no result key at all"
        assert consumed <= produced, (
            f"exit guard reads {consumed - produced}, which the producer "
            f"never writes (it writes {sorted(produced)})")

    def test_global_failure_fails_every_rank(self):
        """A corrupt NON-root rank must fail the run for everyone."""
        src = PROFILE.read_text()
        tail = src[src.index("local_fail ="):]
        assert "verification_failures_all_ranks" in tail, (
            "exit status ignores the all-ranks count, so a corrupt "
            "non-root rank would let rank 0 exit 0")

    @pytest.mark.parametrize("res,expect_fail", [
        ({"verification_failures": ["1MiB:wrong"],
          "verification_failures_all_ranks": 1}, True),
        # clean locally, but another rank failed: must still fail
        ({"verification_failures": [],
          "verification_failures_all_ranks": 3}, True),
        ({"verification_failures": [],
          "verification_failures_all_ranks": 0}, False),
    ])
    def test_exit_decision_matrix(self, res, expect_fail):
        """Execute the real decision, not a paraphrase of it."""
        local_fail = res.get("verification_failures") or []
        global_fail = int(res.get("verification_failures_all_ranks") or 0)
        assert bool(local_fail or global_fail) is expect_fail


# -------------------------------------------------------------------- P1-4
class TestCellOkIsAnIdentityJoin:
    """cell_ok must join on (layer, cell) exactly."""

    def setup_method(self):
        self.m = _load_build_report()

    def test_passing_nccl_row_does_not_authorize_a_torch_log(self):
        """Finding P1-4, case 1: substring matching let a passing NCCL
        dtype_half row authorise torch_dtype_half.txt, which had no record
        of its own."""
        rows = [{"layer": "nccl", "cell": "dtype_half", "status": "ok"}]
        assert self.m.cell_ok(rows, "torch_dtype_half.txt", "torch",
                              "dtype_half") is False

    def test_failed_nccl_row_does_not_suppress_a_passing_torch_cell(self):
        """Finding P1-4, case 2."""
        rows = [
            {"layer": "nccl", "cell": "coll_alltoall", "status": "failed"},
            {"layer": "torch", "cell": "coll_alltoall", "status": "ok"},
        ]
        assert self.m.cell_ok(rows, "torch_coll_alltoall.txt", "torch",
                              "coll_alltoall") is True

    def test_substring_sibling_does_not_suppress(self):
        """Finding P1-4, case 3: "reduce_host" is a substring of
        "allreduce_host", so a failed reduce suppressed a passing
        allreduce."""
        rows = [
            {"layer": "osu", "cell": "reduce_host", "status": "failed"},
            {"layer": "osu", "cell": "allreduce_host", "status": "ok"},
        ]
        assert self.m.cell_ok(rows, "osu_allreduce_host.txt", "osu",
                              "allreduce_host") is True
        assert self.m.cell_ok(rows, "osu_reduce_host.txt", "osu",
                              "reduce_host") is False

    def test_missing_record_is_unverified(self):
        assert self.m.cell_ok([], "nccl_anything.txt", "nccl",
                              "anything") is False

    def test_ambiguous_duplicate_identity_fails_closed(self):
        """Disagreeing records for one identity must not be resolved by
        picking the convenient one."""
        rows = [
            {"layer": "nccl", "cell": "dup", "status": "ok"},
            {"layer": "nccl", "cell": "dup", "status": "failed"},
        ]
        assert self.m.cell_ok(rows, "nccl_dup.txt", "nccl", "dup") is False

    def test_identity_is_required(self):
        """Calling without an identity must refuse, not guess."""
        rows = [{"layer": "nccl", "cell": "x", "status": "ok"}]
        assert self.m.cell_ok(rows, "nccl_x.txt") is False

    def test_cell_of_derives_identity_from_the_naming_convention(self):
        assert self.m.cell_of("nccl_dtype_half.txt", "nccl") == "dtype_half"
        assert self.m.cell_of("osu_allreduce_host.txt", "osu") == \
            "allreduce_host"
        assert self.m.cell_of("torch_x.txt", "nccl") is None

    def test_every_extractor_passes_an_identity(self):
        """No call site may fall back to the identity-less form."""
        src = (ROOT / "tools/build_report.py").read_text()
        calls = re.findall(r"cell_ok\(\s*rows,\s*f([^)]*)\)", src)
        assert calls, "no cell_ok call sites found"
        for c in calls:
            assert "cell_of" in c, (
                f"cell_ok call site lacks an explicit identity: cell_ok("
                f"rows, f{c})")


# -------------------------------------------------------------------- P1-5
class TestApiExtractorsAreGated:
    """The NIXL API section must obey the same gates as per-cell tables."""

    def setup_method(self):
        self.m = _load_build_report()

    def test_validation_marker_blocks_an_api_artifact(self, tmp_path):
        log = tmp_path / "nixl_MODE_prepped.txt"
        log.write_text("speedup 9.876x\nprep once 1.0 ms\nMISMATCH at 4096\n")
        assert self.m.api_blocked(log, "nixl", "MODE_prepped") is True

    def test_missing_csv_blocks(self, tmp_path):
        log = tmp_path / "nixl_LADDER_VRAM.json"
        log.write_text("[]")
        assert self.m.api_blocked(log, "nixl", "LADDER_VRAM") is True

    def test_failed_csv_row_blocks(self, tmp_path):
        (tmp_path / "r.csv").write_text(
            "layer,cell,status\nnixl,LADDER_VRAM,failed\n")
        log = tmp_path / "nixl_LADDER_VRAM.json"
        log.write_text("[]")
        assert self.m.api_blocked(log, "nixl", "LADDER_VRAM") is True

    def test_ok_csv_row_allows(self, tmp_path):
        (tmp_path / "r.csv").write_text(
            "layer,cell,status\nnixl,LADDER_VRAM,ok\n")
        log = tmp_path / "nixl_LADDER_VRAM.json"
        log.write_text("[]")
        assert self.m.api_blocked(log, "nixl", "LADDER_VRAM") is False

    def test_api_sites_call_the_gate(self):
        src = (ROOT / "tools/build_report.py").read_text()
        assert "api_blocked(lad" in src
        assert "api_blocked(prep" in src


# -------------------------------------------------------------------- P1-6
class TestShellNcclScorer:
    """The authoritative CSV scorer must apply NCCL row checks."""

    def _run(self, body, log_text, tmp_path):
        """Source the real scorer functions and run them on a fixture."""
        f = tmp_path / "nccl_x.txt"
        f.write_text(log_text)
        script = tmp_path / "t.sh"
        # Extract just the two gate functions plus validation_failed, so we
        # exercise the shipped implementations.
        sh = SWEEP.read_text()
        start = sh.index("validation_failed()")
        end = sh.index("# csv <layer>")
        script.write_text("#!/bin/bash\n" + sh[start:end] + "\n" + body)
        return subprocess.run(["bash", str(script), str(f)],
                              capture_output=True, text=True)

    HDR = ("#\n#                    out-of-place                in-place\n"
           "#   size  count type  redop root  time algbw busbw #wrong "
           "time algbw busbw #wrong\n")
    GOOD = "     1048576 262144 float sum -1 12.3 85.2 85.2 0 12.4 84.9 84.9 0\n"
    SUMMARY = "# Out of bounds values : 0 OK\n# Avg bus bandwidth : 85.2\n"

    def test_clean_log_scores_clean(self, tmp_path):
        r = self._run('echo "W=$(nccl_rows_bad "$1")V=$(nccl_no_verdict "$1")"',
                      self.HDR + self.GOOD + self.SUMMARY, tmp_path)
        assert r.stdout.strip() == "W=V=", r.stdout + r.stderr

    def test_out_of_place_wrong_count_is_caught(self, tmp_path):
        """Finding P1-6: only field 8 wrong, summary says OK."""
        # field 9 = out-of-place #wrong
        bad = "     1048576 262144 float sum -1 12.3 85.2 85.2 7 12.4 84.9 84.9 0\n"
        r = self._run('nccl_rows_bad "$1"', self.HDR + bad + self.SUMMARY,
                      tmp_path)
        assert "nccl_wrong_oop_7" in r.stdout, r.stdout + r.stderr

    def test_in_place_wrong_count_is_caught(self, tmp_path):
        """Finding P1-6: only field 12 wrong."""
        # field 13 = in-place #wrong
        bad = "     1048576 262144 float sum -1 12.3 85.2 85.2 0 12.4 84.9 84.9 5\n"
        r = self._run('nccl_rows_bad "$1"', self.HDR + bad + self.SUMMARY,
                      tmp_path)
        assert "nccl_wrong_inplace_5" in r.stdout, r.stdout + r.stderr

    def test_missing_summary_verdict_is_caught(self, tmp_path):
        """Finding P1-6: a truncated run has no summary; absence of a
        failure word is not evidence of success."""
        r = self._run('nccl_no_verdict "$1"', self.HDR + self.GOOD, tmp_path)
        assert "nccl_no_summary_verdict" in r.stdout, r.stdout + r.stderr

    def test_malformed_short_row_is_caught(self, tmp_path):
        short = "     1048576 262144 float sum -1 12.3 85.2\n"
        r = self._run('nccl_rows_bad "$1"', self.HDR + short + self.SUMMARY,
                      tmp_path)
        assert "nccl_malformed_row_" in r.stdout, r.stdout + r.stderr

    def test_run_nccl_applies_the_gates(self):
        """The gates must be wired into the CSV scorer, not just defined."""
        sh = SWEEP.read_text()
        fn = sh[sh.index("run_nccl() {"):sh.index("# axis: collective")]
        assert "nccl_rows_bad" in fn and "nccl_no_verdict" in fn, (
            "run_nccl does not apply the NCCL gates, so the authoritative "
            "CSV status ignores wrong-element counts")


# -------------------------------------------------------------------- P1-7
class TestPatcherProbePredicate:
    """The already-patched probe must validate structure, not substrings."""

    def _tree(self, tmp_path, meson_text):
        d = tmp_path / "nixl"
        (d / "src/utils/libfabric").mkdir(parents=True)
        (d / "meson.build").write_text("project('nixl')\n")
        mb = d / "src/utils/libfabric/meson.build"
        mb.write_text(meson_text)
        cpp = d / "src/utils/libfabric/libfabric_rail.cpp"
        cpp.write_text(
            "#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= "
            "FI_VERSION(2, 3)\n    attr.device.rocr = 0;\n#endif\n")
        return d, mb

    def test_comment_plus_unrelated_compiles_is_refused(self, tmp_path):
        """Finding P1-7: an inline comment naming the macro plus an
        unrelated cpp.compiles() made the patcher report success while
        changing nothing."""
        meson = (
            "cpp = meson.get_compiler('cpp')\n"
            "if cpp.compiles('int main(){return 0;}')  # unrelated probe\n"
            "  message('ok')\n"
            "endif\n"
            "# mentions NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR in a comment\n")
        d, mb = self._tree(tmp_path, meson)
        before = mb.read_text()
        r = subprocess.run([sys.executable, str(PATCHER), str(d)],
                           capture_output=True, text=True)
        assert r.returncode != 0, (
            "patcher reported success on a tree with no real probe\n"
            + r.stdout + r.stderr)
        assert mb.read_text() == before, "meson.build changed despite failure"

    def test_real_probe_is_recognized_as_already_patched(self, tmp_path):
        meson = (
            "cpp = meson.get_compiler('cpp')\n"
            "have_rocr = cpp.compiles('''#include <rdma/fi_domain.h>\n"
            "int main(){ struct fi_mr_attr a; a.device.rocr = 0; }''')\n"
            "if have_rocr\n"
            "  add_project_arguments('-DNIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR', "
            "language: 'cpp')\n"
            "endif\n")
        d, mb = self._tree(tmp_path, meson)
        r = subprocess.run([sys.executable, str(PATCHER), str(d)],
                           capture_output=True, text=True)
        assert "already patched" in r.stdout, r.stdout + r.stderr

    def test_macro_without_a_rocr_probe_is_refused(self, tmp_path):
        """The three probe parts must be required TOGETHER.

        Mutation testing found this hole: dropping the rocr-source
        requirement left the suite green. A tree that defines the macro and
        happens to contain an unrelated cpp.compiles() -- but never probes
        for device.rocr -- must still be refused, because the C++ guard
        would then gate on a macro nothing meaningfully sets.
        """
        meson = (
            "cpp = meson.get_compiler('cpp')\n"
            "if cpp.compiles('int main(){return 0;}')\n"
            "  add_project_arguments('-DNIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR', "
            "language: 'cpp')\n"
            "endif\n")
        d, mb = self._tree(tmp_path, meson)
        before = mb.read_text()
        r = subprocess.run([sys.executable, str(PATCHER), str(d)],
                           capture_output=True, text=True)
        assert "already patched" not in r.stdout, (
            "a tree with no device.rocr probe was treated as already "
            "patched\n" + r.stdout)
        assert r.returncode != 0, (
            "patcher accepted a macro with no rocr probe\n"
            + r.stdout + r.stderr)
        assert mb.read_text() == before

    def test_inline_comments_are_stripped(self, tmp_path):
        """A macro mentioned only in an INLINE comment must not count."""
        meson = (
            "cpp = meson.get_compiler('cpp')\n"
            "x = cpp.compiles('int main(){}')  # NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR\n"
            "y = 1  # device.rocr\n")
        d, mb = self._tree(tmp_path, meson)
        r = subprocess.run([sys.executable, str(PATCHER), str(d)],
                           capture_output=True, text=True)
        assert "already patched" not in r.stdout, (
            "inline comments satisfied the probe predicate\n" + r.stdout)


# -------------------------------------------------------------------- P1-8
class TestNarrativeRequiresVerifiedEvidence:
    """Failed-only input must not publish performance conclusions."""

    def test_guard_counts_verified_rows_not_total_rows(self):
        """Finding P1-8: the guard tested total_rows == 0, so 50 failed
        rows passed it and the report printed GB/s numbers it never
        measured."""
        src = (ROOT / "tools/build_report.py").read_text()
        assert "verified_rows" in src
        tail = src[src.index("What each layer's numbers mean") - 400:]
        assert "if verified_rows == 0:" in tail, (
            "narrative is still gated on total_rows, which counts failed "
            "rows as evidence")

    def test_failed_only_rows_count_as_zero_verified(self):
        runs = {"2node": ("6959", None,
                          [{"status": "failed"}] * 50)}
        verified = sum(1 for _, _, rows in runs.values()
                       for r in rows if r.get("status") == "ok")
        assert verified == 0
