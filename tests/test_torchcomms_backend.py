"""Static conformance check: does TorchCommsDist match what DLcomm calls?

Runs without torchcomms installed -- it compares the adapter's method names
against the dist.* surface the collectives actually use, so a missing entry
point is caught in CPU CI rather than 20 minutes into an Aurora queue.
"""

import inspect
import os
import re
import types
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

# ---------------------------------------------------------------------------
# new_group semantics: non-members, and transports without split.
# Job 8824653 died on "XCCL split is not supported now and will be added
# later", so the fallback path is exercised with a fake comm that raises the
# exact upstream message.
# ---------------------------------------------------------------------------


class _FakeComm:
    """Minimal stand-in for torchcomms.TorchComm."""

    def __init__(self, rank=0, size=4, split_error=None):
        self._rank, self._size = rank, size
        self._split_error = split_error
        self.split_calls = []

    def get_rank(self):
        return self._rank

    def get_size(self):
        return self._size

    def split(self, ranks, name, hints=None, timeout=None):
        self.split_calls.append((tuple(ranks), name))
        if self._split_error:
            raise RuntimeError(self._split_error)
        return _FakeComm(rank=list(ranks).index(self._rank), size=len(ranks))


def test_non_member_never_calls_split():
    """dist.new_group is collective over the parent; split throws for non-members."""
    fake = _FakeComm(rank=3, size=4)
    d = tcb.TorchCommsDist(fake)
    grp = d.new_group(ranks=[0, 1])
    assert fake.split_calls == [], "split must not be called by a non-member"
    assert grp is not None
    assert list(grp.ranks) == [0, 1]


def test_member_uses_native_split_when_available():
    fake = _FakeComm(rank=1, size=4)
    d = tcb.TorchCommsDist(fake)
    d.new_group(ranks=[0, 1])
    assert fake.split_calls, "member should use native split"


def test_empty_rank_list_returns_none():
    d = tcb.TorchCommsDist(_FakeComm())
    assert d.new_group(ranks=[]) is None


def test_group_creation_is_cached():
    fake = _FakeComm(rank=0, size=4)
    d = tcb.TorchCommsDist(fake)
    a = d.new_group(ranks=[0, 1])
    b = d.new_group(ranks=[1, 0])   # same set, different order
    assert a is b, "identical member sets must reuse one communicator"
    assert len(fake.split_calls) == 1


def test_unrelated_split_error_is_not_swallowed():
    """Only the documented 'not supported' case may trigger the fallback."""
    fake = _FakeComm(rank=0, size=4, split_error="network unreachable")
    d = tcb.TorchCommsDist(fake)
    with pytest.raises(RuntimeError, match="network unreachable"):
        d.new_group(ranks=[0, 1])


def test_xccl_split_unsupported_triggers_new_comm_fallback(monkeypatch):
    """The exact XCCL message from job 8824653 must route to new_comm."""
    fake = _FakeComm(rank=0, size=4,
                     split_error="XCCL split is not supported now and will be added later")
    d = tcb.TorchCommsDist(fake, backend="xccl", device="xpu:0")

    captured = {}

    def _fake_new_comm_for(key, label):
        captured["key"] = key
        captured["rank_env"] = os.environ.get("RANK")
        return _FakeComm(rank=0, size=len(key))

    monkeypatch.setattr(d, "_new_comm_for", _fake_new_comm_for)
    grp = d.new_group(ranks=[0, 1])
    assert captured["key"] == (0, 1), "fallback did not receive the member set"
    assert grp is not None
    assert d._split_unsupported, "the upstream limitation should be recorded"


def test_fallback_restores_rank_env_vars(monkeypatch):
    """new_comm reads RANK/WORLD_SIZE from env; they must be put back."""
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "24")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")

    fake = _FakeComm(rank=7, size=24,
                     split_error="XCCL split is not supported now and will be added later")
    d = tcb.TorchCommsDist(fake, backend="xccl", device="cpu")

    seen = {}

    class _Store:
        def __init__(self, *a, **k):
            pass

    def _fake_new_comm(backend, device, name, store=None, **kw):
        seen["rank"] = os.environ["RANK"]
        seen["world"] = os.environ["WORLD_SIZE"]
        return _FakeComm(rank=0, size=2)

    fake_tc = types.SimpleNamespace(new_comm=_fake_new_comm)
    monkeypatch.setattr(tcb, "_require", lambda: fake_tc)
    monkeypatch.setattr(tcb, "_tc", fake_tc, raising=False)
    d._root_store = _Store()
    monkeypatch.setattr("torch.distributed.PrefixStore", _Store, raising=False)

    d.new_group(ranks=[6, 7])

    # inside the call: member-local values
    assert seen["rank"] == "1", "rank should be member-local index of 7 in [6,7]"
    assert seen["world"] == "2"
    # after the call: globals restored
    assert os.environ["RANK"] == "7"
    assert os.environ["WORLD_SIZE"] == "24"
