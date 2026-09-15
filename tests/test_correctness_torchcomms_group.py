"""Correctness checking must work under the torchcomms backend.

Job 8826041 ran every collective successfully and then died in the correctness
checker with `AttributeError: 'TorchCommsGroup' object has no attribute
'size'`. The checker imports `torch.distributed` directly and assumed any
`group` it received was a ProcessGroup, but under `ccl_backend: torchcomms`
the groups are `TorchCommsGroup` objects belonging to the adapter.

The failure is worth a test rather than a fix alone: a verification path that
crashes reports nothing, so the run looked like a backend failure when the
data movement had in fact succeeded.
"""

import sys
import types

import pytest

import torch

from dl_comm.analysis import correctness as C
from dl_comm.comm import torchcomms_backend as tcb


class _FakeComm:
    def __init__(self, ranks):
        self.ranks = list(ranks)


class _AdapterDist:
    """Stands in for TorchCommsDist: records calls, exposes its own ReduceOp."""

    ReduceOp = types.SimpleNamespace(MIN="MIN_SENTINEL", SUM="SUM_SENTINEL")

    def __init__(self):
        self.calls = []

    def all_reduce(self, tensor, op=None, group=None, async_op=False):
        self.calls.append(("all_reduce", op))
        return None

    def get_rank(self, group=None):
        return 0


class _ProcessGroupDist:
    """Stands in for torch.distributed."""

    ReduceOp = types.SimpleNamespace(MIN="MIN_SENTINEL")

    def __init__(self):
        self.calls = []

    def all_reduce(self, tensor, op=None, group=None, async_op=False):
        self.calls.append(op)

    def get_rank(self, group=None):
        return 0

    def get_world_size(self, group=None):
        return 2

    def get_process_group_ranks(self, group):
        return [0, 1]


@pytest.fixture
def tc_group():
    group = tcb.TorchCommsGroup.__new__(tcb.TorchCommsGroup)
    group.comm = _FakeComm([0, 1, 2, 3])
    group.ranks = [0, 1, 2, 3]
    return group


@pytest.fixture
def context():
    log = types.SimpleNamespace(output=lambda *a, **k: None)
    return {"log": log, "iteration": 1}


@pytest.fixture(autouse=True)
def _reset_active_dist():
    saved = tcb._ACTIVE_DIST
    yield
    tcb._ACTIVE_DIST = saved


def test_group_info_reads_ranks_without_calling_get_world_size(tc_group):
    """The original crash: dist.get_world_size reaches group.size()."""
    dist = _AdapterDist()  # deliberately has no get_world_size
    ranks, world_size, root, my_index = C._group_info(dist, tc_group)
    assert ranks == [0, 1, 2, 3]
    assert world_size == 4
    assert root == 0
    assert my_index == 0


def test_group_info_still_uses_torch_distributed_for_process_groups():
    """The ProcessGroup path must be unchanged."""
    dist = _ProcessGroupDist()
    ranks, world_size, root, _ = C._group_info(dist, object())
    assert ranks == [0, 1]
    assert world_size == 2
    assert root == 0


def test_verdict_reduction_uses_the_adapters_reduceop(context, tc_group):
    """The adapter exposes ReduceOp, so the call is identical for both."""
    dist = _AdapterDist()
    ok = C._reduce_verdict(
        context, dist, torch, torch.zeros(1), tc_group, [0, 1, 2, 3], 0, True, "t"
    )
    assert ok is True
    assert dist.calls == [("all_reduce", "MIN_SENTINEL")]


def test_adapter_exposes_reduceop_for_op_name_lookup():
    """_op_to_name reads dist.ReduceOp.SUM; job 8826084 died here."""
    from dl_comm.comm.torchcomms_backend import TorchCommsDist

    assert hasattr(TorchCommsDist, "ReduceOp"), (
        "correctness._op_to_name reads dist.ReduceOp off the facade"
    )


def test_verdict_reduction_keeps_reduceop_for_process_groups(context):
    dist = _ProcessGroupDist()
    C._reduce_verdict(
        context, dist, torch, torch.zeros(1), object(), [0, 1], 0, True, "t"
    )
    assert dist.calls == ["MIN_SENTINEL"]


def test_active_dist_exposes_the_adapter_the_run_built():
    """The checker resolves the adapter through this, not a call-site argument."""
    sentinel = _AdapterDist()
    tcb._ACTIVE_DIST = sentinel
    assert tcb.active_dist() is sentinel


def test_active_dist_is_none_before_any_build():
    tcb._ACTIVE_DIST = None
    assert tcb.active_dist() is None
