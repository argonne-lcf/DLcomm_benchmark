"""Reduction ops must not silently degrade to SUM in the torchcomms backend.

Job 8826159 measured, on 12 XPUs through the torchcomms adapter:

    int32 MIN of all-ones   -> 12   (expected 1)
    int32 MIN with rank3=0  -> 11   (expected 0)
    fp32  MIN of all-ones   -> 12.0 (expected 1.0)

Every reduction was executing as a SUM. The cause was the fallback in
TorchCommsDist._op:

    return getattr(tc.ReduceOp, text, tc.ReduceOp.SUM)

A name that did not line up between torch.distributed's ReduceOp and the
torchcomms enum produced a sum instead of an error, so the collective ran,
reported a bandwidth number, and silently computed the wrong thing. That is
the worst failure mode available: a benchmark that looks healthy and measures
something other than what it claims.

These tests pin the behaviour to raising.
"""

import types

import pytest

from dl_comm.comm import torchcomms_backend as tcb


class _RO:
    """Stand-in for torchcomms.ReduceOp carrying only the ops it really has."""

    SUM = "TC_SUM"
    MIN = "TC_MIN"
    MAX = "TC_MAX"
    PRODUCT = "TC_PRODUCT"


class _PartialRO:
    """A build that lacks MIN, to exercise the mismatch path."""

    SUM = "TC_SUM"
    MAX = "TC_MAX"


class _FakeComm:
    ranks = [0, 1]

    def get_rank(self):
        return 0


@pytest.fixture
def dist(monkeypatch):
    monkeypatch.setattr(tcb, "_require", lambda: types.SimpleNamespace(ReduceOp=_RO))
    return tcb.TorchCommsDist(_FakeComm(), backend="xccl", device="xpu")


@pytest.fixture
def partial_dist(monkeypatch):
    monkeypatch.setattr(
        tcb, "_require", lambda: types.SimpleNamespace(ReduceOp=_PartialRO)
    )
    return tcb.TorchCommsDist(_FakeComm(), backend="xccl", device="xpu")


def test_torch_min_maps_to_torchcomms_min(dist):
    """The regression: MIN must not come back as SUM."""
    import torch.distributed as tdist

    assert dist._op(tdist.ReduceOp.MIN) == _RO.MIN


def test_every_torch_op_maps_to_its_own_counterpart(dist):
    import torch.distributed as tdist

    for tname, expected in (
        ("SUM", _RO.SUM),
        ("MIN", _RO.MIN),
        ("MAX", _RO.MAX),
        ("PRODUCT", _RO.PRODUCT),
    ):
        got = dist._op(getattr(tdist.ReduceOp, tname))
        assert got == expected, f"torch {tname} mapped to {got}, expected {expected}"


def test_string_ops_map_too(dist):
    assert dist._op("min") == _RO.MIN
    assert dist._op("sum") == _RO.SUM
    assert dist._op("prod") == _RO.PRODUCT
    assert dist._op("product") == _RO.PRODUCT


def test_none_still_defaults_to_sum(dist):
    """An absent op is a genuine default, unlike an unmatched name."""
    assert dist._op(None) == _RO.SUM


def test_unmatched_torch_op_raises_instead_of_summing(partial_dist):
    """The whole point: a build without MIN must fail, not quietly sum."""
    import torch.distributed as tdist

    with pytest.raises(ValueError) as exc:
        partial_dist._op(tdist.ReduceOp.MIN)
    assert "MIN" in str(exc.value)
    # The available ops are named so the failure is actionable.
    assert "SUM" in str(exc.value)


def test_unmatched_string_op_raises(partial_dist):
    with pytest.raises(ValueError):
        partial_dist._op("min")


def test_no_silent_sum_fallback_in_source():
    """Guard the specific construct that caused this.

    The bug was a two-argument getattr whose default was SUM:
        getattr(tc.ReduceOp, text, tc.ReduceOp.SUM)
    A bare `getattr(tc.ReduceOp, name)` is fine -- it raises on a bad name.
    """
    import inspect
    import re

    src = inspect.getsource(tcb.TorchCommsDist._op)
    # Strip comments so the explanatory note quoting the old line is ignored.
    code = "\n".join(
        line.split("#", 1)[0] for line in src.splitlines()
    ).replace(" ", "").replace("\n", "")
    assert not re.search(r"getattr\([^)]*,[^)]*,[^)]*ReduceOp\.SUM\)", code), (
        "_op has a getattr default again; an unmatched op name would silently "
        "execute as a sum"
    )
