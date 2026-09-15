"""Pre-converted torchcomms ReduceOps must survive both seams.

A torchcomms ReduceOp is an opaque pybind11 object. Unlike
``torch.distributed.ReduceOp``, whose ``str()`` is ``"RedOpType.SUM"``, it
renders as ``"<torchcomms.ReduceOp object at 0x7f...>"`` -- the name is not in
the text at all. Any code that recovers an op's identity by parsing ``str()``
therefore gets garbage from it.

Two places did exactly that, and both were reached with an already-converted
op because the correctness checker calls ``dist.all_reduce(flag,
op=dist.ReduceOp.MIN)`` where ``dist`` is the torchcomms adapter:

* ``TorchCommsDist._op`` raised ValueError on the derived name
  ``'REDUCEOP OBJECT AT 0X...'``. Job 8826319 died on the first timed
  allreduce with precisely that message.
* ``correctness._op_to_name`` returned None, which is worse than a crash: the
  reduction check was skipped and reported as passing-by-absence.

These tests use a stand-in that mimics the opaque repr, so they run without
torchcomms installed.
"""

import sys
import types

import pytest

from dl_comm.comm import torchcomms_backend as tcb


class _OpaqueOp:
    """Stand-in for a pybind11 enum value: no name recoverable from str()."""

    def __init__(self, label):
        self._label = label

    def __repr__(self):
        return f"<torchcomms.ReduceOp object at 0x{id(self):012x}>"

    __str__ = __repr__


class _FakeReduceOpMeta(type):
    pass


def _make_fake_tc():
    """A module object shaped like torchcomms, with opaque ReduceOp members."""

    class ReduceOp(_OpaqueOp, metaclass=_FakeReduceOpMeta):
        pass

    mod = types.ModuleType("torchcomms")
    mod.ReduceOp = ReduceOp
    for name in ("SUM", "MIN", "MAX", "PRODUCT", "AVG", "BAND", "BOR", "BXOR",
                 "PREMUL_SUM"):
        setattr(ReduceOp, name, ReduceOp(name))
    return mod


@pytest.fixture
def fake_tc(monkeypatch):
    mod = _make_fake_tc()
    monkeypatch.setitem(sys.modules, "torchcomms", mod)
    monkeypatch.setattr(tcb, "_require", lambda: mod)
    return mod


class TestAdapterOpPassthrough:
    """_op must accept an op it already produced."""

    def test_opaque_op_is_returned_unchanged(self, fake_tc):
        d = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
        for name in ("SUM", "MIN", "MAX", "PRODUCT", "AVG"):
            member = getattr(fake_tc.ReduceOp, name)
            assert d._op(member) is member, (
                f"{name} was not passed through; this is the job 8826319 crash"
            )

    def test_opaque_op_does_not_raise(self, fake_tc):
        """The regression: a derived name of 'REDUCEOP OBJECT AT 0X...'."""
        d = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
        try:
            d._op(fake_tc.ReduceOp.MIN)
        except ValueError as exc:  # pragma: no cover - the bug
            pytest.fail(f"pre-converted op rejected: {exc}")

    def test_string_ops_still_work(self, fake_tc):
        d = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
        assert d._op("sum") is fake_tc.ReduceOp.SUM
        assert d._op("prod") is fake_tc.ReduceOp.PRODUCT

    def test_none_still_defaults_to_sum(self, fake_tc):
        d = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
        assert d._op(None) is fake_tc.ReduceOp.SUM

    def test_unknown_string_still_raises(self, fake_tc):
        """Hardening must not be undone: a genuinely bad op still fails loud."""
        d = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
        with pytest.raises(ValueError):
            d._op("not_a_real_op")


class TestCheckerOpNaming:
    """_op_to_name must recognise an opaque op instead of silently skipping."""

    def test_opaque_ops_resolve_to_canonical_names(self, fake_tc):
        from dl_comm.analysis import correctness as C

        expected = {"SUM": "sum", "MIN": "min", "MAX": "max",
                    "PRODUCT": "prod"}
        for member, want in expected.items():
            got = C._op_to_name(getattr(fake_tc.ReduceOp, member), None)
            assert got == want, (
                f"{member} resolved to {got!r}, so its correctness check "
                f"would have been skipped rather than run"
            )

    def test_avg_still_skips_rather_than_crashing(self, fake_tc):
        """_expected_reduced has no closed form for AVG; skip, never guess."""
        from dl_comm.analysis import correctness as C

        assert C._op_to_name(fake_tc.ReduceOp.AVG, None) is None

    def test_torch_distributed_ops_still_resolve(self):
        """The existing name-based path must keep working."""
        from dl_comm.analysis import correctness as C

        class _TorchOp:
            def __init__(self, t):
                self.t = t

            def __str__(self):
                return f"RedOpType.{self.t}"

        assert C._op_to_name(_TorchOp("SUM"), None) == "sum"
        assert C._op_to_name(_TorchOp("MIN"), None) == "min"
        assert C._op_to_name(_TorchOp("PRODUCT"), None) == "prod"

    def test_none_is_still_none(self):
        from dl_comm.analysis import correctness as C

        assert C._op_to_name(None, None) is None
