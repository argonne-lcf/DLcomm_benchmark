"""Nodefile ordering must survive contact with PALS.

PALS assigns rank 0 to the first line of $PBS_NODEFILE as PBS wrote it. PBS
does not write that file sorted. Job 6935 was allocated

    x4820c7s6b1n0
    x4820c7s0b0n0

so `sort -u | head -1` yields s0b0n0 while rank 0 actually runs on s6b1n0.
Any address predicted from the sorted list points at a host where nothing is
listening; the resulting "Connection refused" reads as a fabric fault and
costs hours to chase (jobs 6931, 6932).

These tests pin the shell idioms used by the job scripts, so the mistake
cannot quietly return.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The real job-6935 allocation: deliberately NOT in alphabetical order.
NODEFILE_6935 = "x4820c7s6b1n0\nx4820c7s0b0n0\n"

SCRIPTS = [
    REPO / "tools" / "run_all_scales.sh",
    REPO / "examples" / "18_fabric_layers" / "jobscript_fabric_layers.sh",
]


def _sh(script: str, stdin: str = "") -> str:
    out = subprocess.run(
        ["bash", "-c", script], input=stdin,
        capture_output=True, text=True, timeout=30,
    )
    return out.stdout


def test_the_allocation_that_exposed_the_bug_is_not_sorted():
    """Guard the premise: if this fixture were sorted it would prove nothing."""
    lines = NODEFILE_6935.split()
    assert lines != sorted(lines), "fixture must be un-sorted to be meaningful"


def test_sort_u_picks_the_wrong_rank0_host(tmp_path):
    """Demonstrates the actual defect rather than asserting a rule abstractly."""
    nf = tmp_path / "nodefile"
    nf.write_text(NODEFILE_6935)

    sorted_first = _sh(f'sort -u "{nf}" | head -1').strip()
    natural_first = _sh(f'head -1 "{nf}"').strip()
    awk_first = _sh(f"awk '!seen[$0]++' \"{nf}\" | head -1").strip()

    # PALS puts rank 0 on the first line as written -> s6b1n0 (job 6935 B1).
    assert natural_first == "x4820c7s6b1n0"
    assert awk_first == "x4820c7s6b1n0"

    # ...and sort -u disagrees. That disagreement was the bug.
    assert sorted_first == "x4820c7s0b0n0"
    assert sorted_first != natural_first


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_job_scripts_never_sort_the_nodefile(script):
    """`sort -u $PBS_NODEFILE` must not reappear in a launcher."""
    assert script.exists(), f"missing {script}"
    text = script.read_text()

    offenders = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"sort\s+(-\w+\s+)*[\"']?\$(PBS_)?NODEFILE", line)
        and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "sort on $PBS_NODEFILE reorders hosts away from PALS rank order; "
        f"use head -1 or awk '!seen[$0]++' instead: {offenders}"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_nodefile_ordering_rationale_is_written_down(script):
    """A bare idiom invites someone to 'tidy' it into sort -u next year."""
    text = script.read_text().lower()
    assert "pals" in text and "sort -u" in text, (
        f"{script.name} should explain why the nodefile order is load-bearing"
    )
