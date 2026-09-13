"""Static conformance check: does TorchCommsDist match what DLcomm calls?

Runs without torchcomms installed -- it compares the adapter's method names
against the dist.* surface the collectives actually use, so a missing entry
point is caught in CPU CI rather than 20 minutes into an Aurora queue.
"""

import inspect
import re
from pathlib import Path

import pytest

from dl_comm.comm import torchcomms_backend as tcb

SRC = Path(__file__).resolve().parents[1] / "dl_comm" / "comm" / "collectives.py"
MAIN = Path(__file__).resolve().parents[1] / "dl_comm" / "dl_comm_main.py"
PKG = Path(__file__).resolve().parents[1] / "dl_comm"


def _dist_calls_used():
    text = SRC.read_text()
    return sorted(set(re.findall(r"\bdist\.([a-z_]+)", text)))


def _main_dist_calls_used():
    text = MAIN.read_text()
    return sorted(set(re.findall(r"\bdist\.([a-z_]+)", text)))


def _all_dist_calls_in_package():
    """Every dist.<name> anywhere in dl_comm, excluding the adapter itself."""
    names = set()
    for py in PKG.rglob("*.py"):
        if py.name == "torchcomms_backend.py":
            continue
        names |= set(re.findall(r"\bdist\.([a-z_]+)", py.read_text()))
    return sorted(names)


def _dist_call_kwargs_in_package():
    """Map dist.<name> -> set of keyword args callers actually pass.

    Name-only coverage is not enough: job 8824643 failed on all 24 ranks
    because comm_setup.py calls new_group(use_local_synchronization=True) and
    the adapter's signature rejected it. Scanning call sites for kwargs catches
    that class of break in CPU CI.
    """
    calls = {}
    pattern = re.compile(r"\bdist\.([a-z_]+)\s*\(([^)]*)\)", re.DOTALL)
    for py in PKG.rglob("*.py"):
        if py.name == "torchcomms_backend.py":
            continue
        for name, args in pattern.findall(py.read_text()):
            kw = set(re.findall(r"(\w+)\s*=", args))
            calls.setdefault(name, set()).update(kw)
    return calls


def test_adapter_covers_every_dist_call_in_the_whole_package():
    """Scan all of dl_comm, not just the two obvious files.

    The first version of this test checked collectives.py and dl_comm_main.py
    only. comm_setup.py, system_info.py, validation.py and correctness.py also
    call dist.*, and missing one costs a full queue cycle to discover.
    """
    used = _all_dist_calls_in_package()
    assert used, "found no dist.* calls -- the scan is wrong"
    missing = [n for n in used if not hasattr(tcb.TorchCommsDist, n)]
    assert not missing, (
        f"TorchCommsDist is missing {missing}. Full surface used across "
        f"dl_comm: {used}"
    )


def test_adapter_accepts_every_keyword_callers_pass():
    """Signatures must accept the kwargs real call sites use."""
    offenders = []
    for name, kwargs in _dist_call_kwargs_in_package().items():
        fn = getattr(tcb.TorchCommsDist, name, None)
        if fn is None or not kwargs:
            continue
        sig = inspect.signature(fn)
        accepts_any = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if accepts_any:
            continue
        for kw in kwargs:
            if kw not in sig.parameters:
                offenders.append(f"{name}(...{kw}=...)")
    assert not offenders, (
        f"TorchCommsDist signatures reject keywords that callers pass: "
        f"{sorted(offenders)}"
    )


def test_adapter_covers_every_dist_call_the_main_loop_makes():
    """The driver calls dist.* too -- notably destroy_process_group at shutdown.

    Missing this crashes only at the very end of a run, after all the
    measurements, which is the most expensive possible place to fail.
    """
    used = _main_dist_calls_used()
    assert used, "found no dist.* calls in the main loop -- the grep is wrong"
    missing = [name for name in used if not hasattr(tcb.TorchCommsDist, name)]
    assert not missing, (
        f"TorchCommsDist is missing {missing}; dl_comm_main.py calls these. "
        f"Full surface used: {used}"
    )


def test_adapter_covers_every_dist_call_the_collectives_make():
    """Every dist.X used by a collective must exist on TorchCommsDist."""
    used = _dist_calls_used()
    assert used, "found no dist.* calls -- the grep is wrong"
    missing = [name for name in used if not hasattr(tcb.TorchCommsDist, name)]
    assert not missing, (
        f"TorchCommsDist is missing {missing}; collectives.py calls these via "
        f"dist. Full surface used: {used}"
    )


def test_module_imports_without_torchcomms_installed():
    """Importing the adapter must not require torchcomms to be present."""
    assert isinstance(tcb.is_available(), bool)


def test_requesting_backend_without_library_raises_actionable_error():
    if tcb.is_available():
        pytest.skip("torchcomms is installed here; the failure path needs it absent")
    with pytest.raises(RuntimeError) as e:
        tcb._require()
    msg = str(e.value)
    assert "torchcomms" in msg
    assert "ccl_backend" in msg, "error should name the config field to change"


def test_transport_defaults_by_device_type():
    assert tcb.resolve_transport("gpu") == "xccl"
    assert tcb.resolve_transport("xpu") == "xccl"
    assert tcb.resolve_transport("cpu") == "gloo"


def test_explicit_transport_is_validated_not_trusted():
    assert tcb.resolve_transport("gpu", "nccl") == "nccl"
    with pytest.raises(ValueError) as e:
        tcb.resolve_transport("gpu", "nccl-typo")
    assert "nccl-typo" in str(e.value)


def test_every_supported_transport_resolves():
    for t in tcb.SUPPORTED_TRANSPORTS:
        assert tcb.resolve_transport("gpu", t) == t


def test_group_records_its_global_ranks():
    g = tcb.TorchCommsGroup(object(), [4, 5, 6])
    assert g.ranks == [4, 5, 6]
    assert "4" in repr(g)


def test_work_wrapper_is_idempotent_and_safe_on_none():
    w = tcb._Work(None)
    assert w.wait() is None
    assert w.is_completed() is True


def test_recv_without_src_is_rejected():
    """torch.distributed allows a wildcard src; torchcomms does not.

    Silently receiving from the wrong peer would corrupt a correctness check,
    so the adapter must refuse rather than guess.
    """
    d = tcb.TorchCommsDist(comm=object())
    with pytest.raises(ValueError) as e:
        d.recv(tensor=object(), src=None)
    assert "src" in str(e.value)
    with pytest.raises(ValueError):
        d.irecv(tensor=object(), src=None)
