"""Guard the PBS nodefile node-count idiom used by every jobscript.

Background
----------
`wc -l < $PBS_NODEFILE` counts newline characters, not lines. On Tara the PBS
nodefile carries no trailing newline on its final entry, so a single-node
allocation reports 0 nodes and the jobscript builds `mpiexec --np 0`, which
dies with:

    Invalid -n/--np value 0: outside of valid range 1-2147483647

Measured on tara-n with a real 1-node nodefile:

    wc -l     => 0
    grep -c . => 1
    bytes: 45, last byte 0x76 ('v')  -- no 0x0a terminator

These tests fail if any jobscript regresses to `wc -l`, and they execute the
real block extracted from each script to prove it counts correctly.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = ("examples", "tools")

# `NNODES=` fed from a nodefile via wc -l, in backtick or $() form.
WC_NODEFILE = re.compile(r"NNODES=.*wc\s+-l[^\n]*PBS_NODEFILE")
BLOCK_START = re.compile(r'^\s*if \[ -n "\$\{PBS_NODEFILE:-\}" \]; then\s*$')


def _shell_scripts() -> list[Path]:
    out: list[Path] = []
    for d in SEARCH_DIRS:
        out.extend(sorted((REPO_ROOT / d).rglob("*.sh")))
    return out


def _scripts_counting_nodes() -> list[Path]:
    return [p for p in _shell_scripts() if "NNODES=" in p.read_text()]


def _extract_block(path: Path) -> str:
    """Pull the `if [ -n "${PBS_NODEFILE:-}" ]` ... `fi` block out of a script.

    Skips the calling test when the script does not use the shared block.
    """
    lines = path.read_text().split("\n")
    start = next((i for i, l in enumerate(lines) if BLOCK_START.match(l)), None)
    if start is not None:
        for j in range(start, len(lines)):
            if lines[j].rstrip() == "fi":
                return "\n".join(lines[start : j + 1])
    pytest.skip("script does not use the shared nodefile block")
    raise AssertionError("unreachable")  # pragma: no cover


def _run_block(block: str, nodefile: Path | None) -> tuple[int, str]:
    script = "set -u\n" + block + "\necho RESULT=$NNODES\n"
    env = {"PATH": "/usr/bin:/bin"}
    if nodefile is not None:
        env["PBS_NODEFILE"] = str(nodefile)
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=env
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


@pytest.mark.parametrize(
    "path", _scripts_counting_nodes(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_jobscript_counts_nodes_with_wc_l(path: Path) -> None:
    """`wc -l` on a nodefile undercounts and yields `mpiexec --np 0`."""
    offenders = [
        line.strip()
        for line in path.read_text().split("\n")
        if WC_NODEFILE.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} counts nodes with `wc -l`, which "
        f"undercounts a nodefile with no trailing newline: {offenders}"
    )


@pytest.mark.parametrize(
    "path", _scripts_counting_nodes(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_every_jobscript_has_a_syntactically_valid_nodefile_block(path: Path) -> None:
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"


@pytest.mark.parametrize(
    "path", _scripts_counting_nodes(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_node_count_block_handles_missing_trailing_newline(
    path: Path, tmp_path: Path
) -> None:
    """The exact Tara failure: one node, no trailing newline, must count 1."""
    block = _extract_block(path)

    nf = tmp_path / "nodefile_no_newline"
    nf.write_text("x4820c7s0b0n0.hsn4820.north.tara.alcf.anl.gov")  # no "\n"
    rc, out = _run_block(block, nf)
    assert rc == 0 and out.endswith("RESULT=1"), (
        f"{path.relative_to(REPO_ROOT)} miscounted a 1-node nodefile with no "
        f"trailing newline (rc={rc}, out={out!r})"
    )


@pytest.mark.parametrize(
    "path", _scripts_counting_nodes(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_node_count_block_counts_multiple_nodes(path: Path, tmp_path: Path) -> None:
    block = _extract_block(path)

    nf = tmp_path / "nodefile_three"
    nf.write_text("a\nb\nc\n")
    rc, out = _run_block(block, nf)
    assert rc == 0 and out.endswith("RESULT=3"), f"rc={rc}, out={out!r}"

    nf_nonl = tmp_path / "nodefile_three_no_newline"
    nf_nonl.write_text("a\nb\nc")
    rc, out = _run_block(block, nf_nonl)
    assert rc == 0 and out.endswith("RESULT=3"), f"rc={rc}, out={out!r}"


@pytest.mark.parametrize(
    "path", _scripts_counting_nodes(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_node_count_block_fails_loudly_on_empty_nodefile(
    path: Path, tmp_path: Path
) -> None:
    """An empty nodefile must abort, never silently launch `--np 0`."""
    block = _extract_block(path)

    nf = tmp_path / "nodefile_empty"
    nf.write_text("")
    rc, out = _run_block(block, nf)
    assert rc != 0, f"empty nodefile did not abort (out={out!r})"
    assert "RESULT=0" not in out, "empty nodefile produced a usable count of 0"


@pytest.mark.parametrize(
    "path", _scripts_counting_nodes(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_node_count_block_defaults_to_one_without_pbs(path: Path) -> None:
    """Interactive / non-PBS runs must still work."""
    block = _extract_block(path)

    rc, out = _run_block(block, None)
    assert rc == 0 and out.endswith("RESULT=1"), f"rc={rc}, out={out!r}"


def test_shared_nodefile_helper_exists_and_is_valid() -> None:
    helper = REPO_ROOT / "tools" / "nodefile.sh"
    assert helper.is_file(), "tools/nodefile.sh is missing"
    proc = subprocess.run(["bash", "-n", str(helper)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_shared_helper_matches_the_inlined_behaviour(tmp_path: Path) -> None:
    helper = REPO_ROOT / "tools" / "nodefile.sh"
    nf = tmp_path / "nf"
    nf.write_text("only.node.gov")  # no trailing newline

    proc = subprocess.run(
        ["bash", "-c", f'source "{helper}"; dlcomm_node_count "{nf}"'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1", proc.stdout
