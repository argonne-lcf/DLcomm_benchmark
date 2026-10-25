"""Guard the Python interpreter assumptions the examples actually rely on.

Background
----------
Run ``nixl_remote_20261024_170936`` on tara-n failed every rank with::

    File ".../examples/19_nixl_remote_rdma/mpi_fabric_baseline.py", line 13
        from __future__ import annotations
        ^
    SyntaxError: future feature annotations is not defined

That is not a bug in the source. ``from __future__ import annotations`` was
added in Python 3.7, and tara's *system* ``python3`` is 3.6.15::

    $ python3 --version
    Python 3.6.15

So a jobscript that resolves the interpreter as a bare ``python3`` -- rather
than the project venv, which is 3.12 -- fails at import time on every rank,
before a single measurement runs. The same sweep produced
``No module named 'mpi4py'`` for the same underlying reason: a non-venv
interpreter was picked up.

These tests pin the two invariants that keep that from recurring:

1. Every example module that uses modern syntax declares the minimum version
   it needs, and the interpreter running the suite satisfies it.
2. No jobscript invokes a bare ``python``/``python3`` for a benchmark payload.
   The interpreter must come from an explicitly resolved venv variable, so
   the system 3.6 can never be selected by accident.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

# `from __future__ import annotations` requires 3.7; the project targets 3.9+
# syntax elsewhere (PEP 585 builtin generics in annotations).
MIN_PYTHON = (3, 7)


def _python_files() -> list[Path]:
    return sorted(EXAMPLES.rglob("*.py"))


def _shell_files() -> list[Path]:
    return sorted(EXAMPLES.rglob("*.sh")) + sorted((REPO_ROOT / "tools").rglob("*.sh"))


def test_interpreter_is_new_enough() -> None:
    """The suite itself must not be run under an interpreter the code predates."""
    assert sys.version_info >= MIN_PYTHON, (
        f"examples require Python >= {'.'.join(map(str, MIN_PYTHON))}, "
        f"running {sys.version.split()[0]}. On tara the system python3 is "
        "3.6.15 -- use the project venv."
    )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.name))
def test_example_modules_parse(path: Path) -> None:
    """Every example parses under the current interpreter.

    This is the check that would have caught the 170936 failure locally
    instead of burning an allocation.
    """
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{path.relative_to(REPO_ROOT)} does not parse: {exc}")


# A benchmark payload launched through one of these must not use a bare
# interpreter name -- that resolves to the system python.
_LAUNCHER = re.compile(r"\bmpiexec\b[^\n]*", re.MULTILINE)
_BARE_PY = re.compile(r"(?<![\w/$.\"'{-])python3?(?![\w.])")


@pytest.mark.parametrize("path", _shell_files(), ids=lambda p: str(p.name))
def test_no_bare_python_in_launchers(path: Path) -> None:
    """mpiexec lines must use a resolved interpreter, not bare python/python3.

    tara's system python3 is 3.6.15 and has no mpi4py, so a bare name either
    raises SyntaxError on modern syntax or ModuleNotFoundError on import --
    both observed in the 2026-10-24 sweep.
    """
    text = path.read_text(encoding="utf-8")
    offenders = []
    for line in _LAUNCHER.findall(text):
        # Strip comments before matching; prose may mention python3 freely.
        code = line.split("#", 1)[0]
        if _BARE_PY.search(code):
            offenders.append(line.strip())

    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} launches a bare python/python3 "
        f"(resolves to system 3.6.15 on tara):\n  " + "\n  ".join(offenders)
    )
