"""barrier must be verified, not skipped.

``check_collective_correctness`` used to return immediately for barrier, so
every barrier task reported ``[NO-CHECKS]``. By this project's standard an
unfired check is untested, not passing -- a barrier that silently failed to
synchronise would have been indistinguishable from a working one.

A barrier moves no payload, but it has a checkable property: it is a
rendezvous, so when it returns every rank in the group must have arrived at the
*same* barrier. Each rank contributes its iteration number; the group takes the
MIN and the MAX. Disagreement means the ranks were at different barriers.

What this cannot catch: a barrier that is a no-op on every rank at once. No
collective-level check can, because the observable state is identical.
"""

import types

import pytest

from dl_comm.analysis import correctness as C


class _Failures:
    def __init__(self):
        self.skips = []
        self.fails = []
        self.passes = []


@pytest.fixture(autouse=True)
def capture(monkeypatch):
    f = _Failures()
    monkeypatch.setattr(C.failures, "record_skip", lambda d: f.skips.append(d))
    monkeypatch.setattr(C.failures, "record_failure", lambda d: f.fails.append(d))
    monkeypatch.setattr(C.failures, "record_pass", lambda: f.passes.append(True))
    return f


class _Log:
    def __init__(self):
        self.lines = []

    def output(self, msg):
        self.lines.append(msg)

    info = output


class _Group:
    """Plain group object exposing a rank list, like TorchCommsGroup."""

    def __init__(self, ranks):
        self.ranks = list(ranks)


class _Dist:
    """Fake collective layer.

    ``peers`` models the iteration numbers the *other* ranks contributed.

    The verdict reduction in ``_reduce_verdict`` is a separate concern: it
    MIN-reduces a 1/0 flag, not an iteration number. Feeding it the peer
    iterations would corrupt the verdict (min([1, 5, 5]) == 1 passes by luck,
    min([1, 0, 0]) == 0 fails spuriously). Flag-shaped reductions -- values
    already restricted to 0/1 -- are therefore treated as the verdict and
    reduced against simulated peers that all agree, so these tests exercise
    the rendezvous logic rather than the verdict plumbing, which
    test_correctness_torchcomms_group.py already covers.
    """

    def __init__(self, peers, peer_verdicts=None):
        self.peers = list(peers)
        self.peer_verdicts = list(peer_verdicts) if peer_verdicts else None
        self.ReduceOp = types.SimpleNamespace(MIN="MIN", MAX="MAX", SUM="SUM")
        self.calls = []
        self.rendezvous_calls = 0

    def all_reduce(self, tensor, op=None, group=None):
        mine = int(tensor.item())
        is_verdict = self.rendezvous_calls >= 2
        if is_verdict:
            pool = (self.peer_verdicts or [1, 1]) + [mine]
        else:
            self.calls.append(op)
            self.rendezvous_calls += 1
            pool = self.peers + [mine]

        if op == "MIN":
            tensor.fill_(min(pool))
        elif op == "MAX":
            tensor.fill_(max(pool))
        else:
            tensor.fill_(sum(pool))
        return tensor

    def get_rank(self, group=None):
        return 0


def _context(iteration, log=None):
    return {
        "cfg": types.SimpleNamespace(framework="pytorch"),
        "log": log or _Log(),
        "iteration": iteration,
    }


def _run(iteration, peers, log=None):
    """Drive the real barrier check with a fake dist layer.

    Giving the group a ``.comm`` attribute makes _check_barrier consult
    ``active_dist()``, which is the injection point the production code
    already uses for the torchcomms backend.
    """
    import dl_comm.analysis.correctness as mod
    from dl_comm.comm import torchcomms_backend as tcb

    ctx = _context(iteration, log)
    dist = _Dist(peers)
    group = _Group([0, 1, 2])
    group.comm = object()

    prev = tcb._ACTIVE_DIST
    tcb._ACTIVE_DIST = dist
    try:
        mod._check_barrier(ctx, group=group, group_type="FLATVIEW", group_id=0)
    finally:
        tcb._ACTIVE_DIST = prev
    return ctx["log"]


class TestBarrierIsNoLongerSkipped:
    def test_barrier_is_not_skipped_by_the_dispatcher(self, monkeypatch):
        """The early `return` for barrier is gone."""
        called = {}

        def spy(context, group=None, group_type=None, group_id=None):
            called["yes"] = True

        monkeypatch.setattr(C, "_check_barrier", spy)
        ctx = _context(0)
        C.check_collective_correctness(ctx, None, "barrier", group=None,
                                       group_type="FLATVIEW", group_id=0)
        assert called.get("yes"), "barrier no longer reaches a check"

    def test_barrier_records_a_check_not_a_skip(self, capture):
        _run(iteration=5, peers=[5, 5])
        assert capture.skips == [], f"barrier was skipped: {capture.skips}"


class TestSynchronisedBarrierPasses:
    def test_all_ranks_at_the_same_iteration_passes(self, capture):
        _run(iteration=7, peers=[7, 7])
        assert capture.fails == [], f"a synchronised barrier failed: {capture.fails}"
        # _reduce_verdict logs only on failure, so silence is the pass signal;
        # the recorded pass is what proves the check ran.
        assert capture.passes, "a passing barrier recorded no check"

    def test_iteration_zero_is_handled(self, capture):
        """0 is falsy; it must still be verified rather than skipped."""
        _run(iteration=0, peers=[0, 0])
        assert capture.skips == []
        assert capture.fails == []


class TestDesynchronisedBarrierFails:
    """The check must actually fire -- an unfired threshold is untested."""

    def test_a_lagging_rank_is_detected(self, capture):
        _run(iteration=5, peers=[4, 5])
        assert capture.fails, "a rank at a different barrier was not detected"

    def test_a_leading_rank_is_detected(self, capture):
        _run(iteration=5, peers=[5, 9])
        assert capture.fails, "a rank ahead at a later barrier was not detected"

    def test_failure_names_the_span(self, capture):
        _run(iteration=5, peers=[2, 8])
        assert capture.fails
        detail = " ".join(capture.fails)
        assert "2" in detail and "8" in detail, (
            f"failure detail does not report the observed span: {detail}"
        )


class TestBothBoundsAreChecked:
    def test_uses_min_and_max(self, capture):
        """A single bound would miss desync in one direction."""
        import torch

        ctx = _context(3)
        dist = _Dist([3, 3])
        group = _Group([0, 1, 2])
        group.comm = object()
        from dl_comm.comm import torchcomms_backend as tcb
        prev = tcb._ACTIVE_DIST
        tcb._ACTIVE_DIST = dist
        try:
            C._check_barrier(ctx, group=group, group_type="FLATVIEW", group_id=0)
        finally:
            tcb._ACTIVE_DIST = prev
        assert "MIN" in dist.calls and "MAX" in dist.calls, (
            f"barrier must bound the rendezvous from both sides, saw {dist.calls}"
        )


class TestBadInput:
    def test_non_integer_iteration_skips_explicitly(self, capture):
        _run(iteration="not-a-number", peers=[1, 1])
        assert capture.skips, "a non-integer iteration must skip visibly"
        assert capture.fails == []
