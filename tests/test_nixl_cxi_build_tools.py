"""The NIXL Cray patcher must never leave a half-patched tree.

Review finding: the script wrote meson.build, then validated
libfabric_rail.cpp. If the second anchor was missing -- which is exactly what
happens when upstream moves -- it exited after the first write, leaving a tree
with the probe defined but the guard still on the version test. Worse, a
re-run reported "already patched" for the first file and could not recover.

The build scripts these tests cover are driven on a cluster, so the tests
exercise the patcher against a synthetic tree rather than a real checkout.
"""
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "tools" / "nixl_cxi_build" / "patch_nixl_cray.py"
BUILDER = REPO / "tools" / "nixl_cxi_build" / "build_nixl_cxi.sh"

MESON_WITH_ANCHOR = (
    "    if something\n"
    "        message('libfabric does not support "
    "FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV')\n"
    "    endif\n")
RAIL_WITH_ANCHOR = (
    "void f() {\n"
    "#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= FI_VERSION(2, 3)\n"
    "                mr_attr.device.rocr = device_id;\n"
    "#endif\n}\n")
RAIL_UPSTREAM_MOVED = "void f() {\n    /* guard refactored away upstream */\n}\n"

PROBE = "NIXL_HAVE_FI_MR_ATTR_DEVICE_ROCR"


def _tree(tmp_path, rail_text, with_meson=True):
    d = tmp_path / "src/utils/libfabric"
    d.mkdir(parents=True)
    if with_meson:
        (d / "meson.build").write_text(MESON_WITH_ANCHOR)
    (d / "libfabric_rail.cpp").write_text(rail_text)
    return tmp_path


def _run(root):
    return subprocess.run([sys.executable, str(PATCHER), str(root)],
                          capture_output=True, text=True)


class TestAllOrNothing:
    def test_missing_rail_anchor_leaves_meson_untouched(self, tmp_path):
        root = _tree(tmp_path, RAIL_UPSTREAM_MOVED)
        meson = root / "src/utils/libfabric/meson.build"
        before = meson.read_text()
        proc = _run(root)
        assert proc.returncode != 0, "a tree it cannot patch must not exit 0"
        assert meson.read_text() == before, (
            "meson.build was modified before the second anchor was validated, "
            "leaving a half-patched tree")
        assert PROBE not in meson.read_text()

    def test_missing_file_is_refused_before_any_write(self, tmp_path):
        d = tmp_path / "src/utils/libfabric"
        d.mkdir(parents=True)
        (d / "meson.build").write_text(MESON_WITH_ANCHOR)
        proc = _run(tmp_path)
        out = proc.stdout + proc.stderr
        assert proc.returncode != 0
        assert "libfabric_rail.cpp" in out
        # It must be the script's own pre-flight check that refuses, not an
        # unhandled FileNotFoundError from deep inside the edit logic -- a
        # traceback means the check was removed and we got lucky.
        assert "Traceback" not in out, (
            "the missing file surfaced as a crash, not a checked refusal")
        assert "FATAL" in out


class TestHappyPath:
    def test_both_files_patched(self, tmp_path):
        root = _tree(tmp_path, RAIL_WITH_ANCHOR)
        proc = _run(root)
        assert proc.returncode == 0, proc.stderr
        meson = (root / "src/utils/libfabric/meson.build").read_text()
        rail = (root / "src/utils/libfabric/libfabric_rail.cpp").read_text()
        assert PROBE in meson
        assert f"#ifdef {PROBE}" in rail
        assert "FI_VERSION(2, 3)" not in rail, (
            "the version test is the wrong predicate and must be replaced")

    def test_rerun_is_idempotent(self, tmp_path):
        root = _tree(tmp_path, RAIL_WITH_ANCHOR)
        assert _run(root).returncode == 0
        meson_once = (root / "src/utils/libfabric/meson.build").read_text()
        second = _run(root)
        assert second.returncode == 0
        assert (root / "src/utils/libfabric/meson.build").read_text() == meson_once
        assert "already patched" in second.stdout


class TestBuilderFailsLoudly:
    """A build script that exits 0 on a failed install is a trap."""

    def test_install_failure_is_checked(self):
        body = BUILDER.read_text()
        assert "INSTALL_RC=$?" in body
        assert "FATAL: install failed" in body, (
            "ninja install's exit code was printed but never acted on")

    def test_missing_plugin_is_fatal(self):
        body = BUILDER.read_text()
        idx = body.index("LIBFABRIC_PLUGIN_MISSING")
        assert "exit 1" in body[idx:idx + 200], (
            "the whole point of this script is to produce the plugin the "
            "pip wheel advertises but does not ship; missing it is a failure")
