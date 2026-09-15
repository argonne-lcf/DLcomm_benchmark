"""Global correctness-failure accounting for DLcomm.

The original benchmark reported a failed verification with ``log.output(...)``
and nothing else: the process still exited 0, so a silent data corruption at
8192 nodes was a line in a log nobody greps.

This module keeps a process-local tally that ``dl_comm_main`` reduces across
``MPI.COMM_WORLD`` at end of run to decide the exit status.
See ``docs/fixes/04-fail-loudly.md``.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()

_state = {
    "checks": 0,
    "failures": 0,
    "skipped": 0,
    "details": [],
}

MAX_DETAILS = 50


def reset() -> None:
    with _lock:
        _state["checks"] = 0
        _state["failures"] = 0
        _state["skipped"] = 0
        _state["details"] = []


def record_pass() -> None:
    with _lock:
        _state["checks"] += 1


def record_failure(detail: str) -> None:
    with _lock:
        _state["checks"] += 1
        _state["failures"] += 1
        if len(_state["details"]) < MAX_DETAILS:
            _state["details"].append(detail)


def record_skip(detail: str) -> None:
    """A check that could not be performed (e.g. collective returned no data).

    Skips are tracked separately from passes so that a verification path which
    silently never runs cannot masquerade as a clean result.
    """
    with _lock:
        _state["skipped"] += 1
        if len(_state["details"]) < MAX_DETAILS:
            _state["details"].append(f"SKIPPED: {detail}")


def snapshot() -> dict:
    with _lock:
        return {
            "checks": _state["checks"],
            "failures": _state["failures"],
            "skipped": _state["skipped"],
            "details": list(_state["details"]),
        }


def failure_count() -> int:
    with _lock:
        return _state["failures"]


def skip_count() -> int:
    with _lock:
        return _state["skipped"]


def check_count() -> int:
    with _lock:
        return _state["checks"]
