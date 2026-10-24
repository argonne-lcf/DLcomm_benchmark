"""Regression cover for global -> group-local rank translation.

DEFECT (Tara job 8481, 2-node torchcomms, 2026-10-24)
-----------------------------------------------------
torch.distributed takes GLOBAL ranks for the root/peer argument of
broadcast/reduce/scatter/gather/send/recv. torchcomms takes a GROUP-LOCAL
rank. The adapter cast with int() and passed the value through, so any group
not based at global rank 0 received a root it could not interpret:

    RuntimeError: root must be in range [0, 4), but got 4
    rank 4 died from signal 11 and dumped core

Reproduced exactly below without a cluster: within_node group [4,5,6,7] on
the second node, broadcast root = min(group) = 4.

Why it escaped the 1-node run: with one node the only group is [0,1,2,3],
min() is 0, and 0 is both the global and the group-local answer. The bug
needs >= 2 nodes. Job 8480 (1 node) passed all 18 layers on the same commit
that job 8481 (2 nodes) crashed on.

Why it looked like a flaky node: only root-bearing collectives were affected,
so allreduce/allgather/reducescatter/alltoall (TASKs 1-4) completed and the
crash landed on TASK 5 (broadcast).

Run: PYTHONPATH=<repo> python3 tests/test_torchcomms_rank_translation.py
"""

import os
import unittest
from unittest import mock

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_backend():
    """Import torchcomms_backend without triggering dl_comm.comm.__init__.

    Matches the loader in tests/test_torchcomms_transport.py. Importing
    ``dl_comm.comm.torchcomms_backend`` normally would execute the package
    __init__, which pulls in omegaconf and the rest of the runtime stack --
    none of which this module needs. Loading the file directly keeps the
    suite runnable on a bare interpreter (Tara's login nodes have only
    cray-python, with no omegaconf).
    """
    path = ROOT / "dl_comm" / "comm" / "torchcomms_backend.py"
    spec = importlib.util.spec_from_file_location("_tcb_rank_xlate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tcb = _load_backend()


class FakeComm:
    """Minimal torchcomms communicator that enforces the real rank contract.

    The point of the test is that torchcomms REJECTS an out-of-range root,
    so the fake must reject it too -- a permissive stub would pass against
    the buggy code and prove nothing.
    """

    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank
        self.calls = []

    def get_size(self):
        return self._size

    def get_rank(self):
        return self._rank

    def _check_root(self, kind, r):
        if not (0 <= r < self._size):
            raise RuntimeError(
                f"root must be in range [0, {self._size}), but got {r}")
        self.calls.append((kind, r))

    def broadcast(self, tensor, root, async_op=False):
        self._check_root("broadcast", root)

    def reduce(self, tensor, root, op=None, async_op=False):
        self._check_root("reduce", root)

    def scatter(self, tensor, lst, root, async_op=False):
        self._check_root("scatter", root)

    def gather(self, lst, tensor, root, async_op=False):
        self._check_root("gather", root)

    def send(self, tensor, peer, async_op=False):
        self._check_root("send", peer)

    def recv(self, tensor, peer, async_op=False):
        self._check_root("recv", peer)


class _FakeReduceOp:
    SUM = "SUM"
    PRODUCT = "PRODUCT"
    MIN = "MIN"
    MAX = "MAX"
    AVG = "AVG"


class _FakeTorchComms:
    """Stand-in for the optional ``torchcomms`` module.

    REVIEW FIX (20261024T071020Z-dlcomm3c): ``test_reduce_dst_translated``
    used to error on any host without torchcomms installed, because
    ``reduce()`` calls ``_op()`` which calls ``_require()``. The test is about
    rank translation, not about the op mapping, so the dependency is injected
    here instead of being required from the environment. The suite is now
    hermetic: it passes on a bare interpreter with only the stdlib.
    """

    ReduceOp = _FakeReduceOp


def _dist_with_group(global_ranks, world=8):
    """Build the adapter with a subcommunicator covering `global_ranks`."""
    world_comm = FakeComm(world)
    dist = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
    dist._comm = world_comm
    dist._backend = "nccl"
    dist._device = None
    group_comm = FakeComm(len(global_ranks))
    group = tcb.TorchCommsGroup(group_comm, global_ranks)
    return dist, group, group_comm


class TestSecondNodeGroup(unittest.TestCase):
    """The exact shape that crashed: 2 nodes x 4, second group based at 4."""

    RANKS = [4, 5, 6, 7]

    def setUp(self):
        self.dist, self.group, self.comm = _dist_with_group(self.RANKS)

    def test_broadcast_root_4_becomes_local_0(self):
        # THE REGRESSION: pre-fix this raised
        # "root must be in range [0, 4), but got 4".
        self.dist.broadcast(object(), src=4, group=self.group)
        self.assertEqual(self.comm.calls, [("broadcast", 0)])

    def test_every_member_maps_to_its_offset(self):
        for i, g in enumerate(self.RANKS):
            c = FakeComm(4)
            grp = tcb.TorchCommsGroup(c, self.RANKS)
            self.dist.broadcast(object(), src=g, group=grp)
            self.assertEqual(c.calls, [("broadcast", i)],
                             f"global {g} should map to local {i}")

    def test_reduce_dst_translated(self):
        # Patch the optional torchcomms module so this test exercises rank
        # translation rather than the presence of an optional dependency.
        with mock.patch.object(tcb, "_tc", _FakeTorchComms):
            self.dist.reduce(object(), dst=6, group=self.group)
        self.assertEqual(self.comm.calls, [("reduce", 2)])

    def test_scatter_src_translated(self):
        self.dist.scatter(object(), scatter_list=[], src=5, group=self.group)
        self.assertEqual(self.comm.calls, [("scatter", 1)])

    def test_gather_dst_translated(self):
        self.dist.gather(object(), gather_list=[], dst=7, group=self.group)
        self.assertEqual(self.comm.calls, [("gather", 3)])

    def test_send_recv_peers_translated(self):
        self.dist.send(object(), dst=7, group=self.group)
        self.dist.recv(object(), src=4, group=self.group)
        self.assertEqual(self.comm.calls, [("send", 3), ("recv", 0)])

    def test_non_member_rejected_with_a_readable_message(self):
        # Must not clamp. Broadcasting from the wrong source would still
        # "succeed" and report plausible numbers.
        with self.assertRaises(ValueError) as cm:
            self.dist.broadcast(object(), src=1, group=self.group)
        msg = str(cm.exception)
        self.assertIn("1", msg)
        self.assertIn("4", msg)  # lists the actual members


class TestFirstNodeGroupStillWorks(unittest.TestCase):
    """The 1-node case that masked the bug must keep working."""

    def test_group_based_at_zero_is_identity(self):
        dist, group, comm = _dist_with_group([0, 1, 2, 3])
        dist.broadcast(object(), src=0, group=group)
        self.assertEqual(comm.calls, [("broadcast", 0)])

    def test_world_group_is_untranslated(self):
        dist, _, _ = _dist_with_group([0, 1, 2, 3])
        dist.broadcast(object(), src=3, group=None)
        self.assertEqual(dist._comm.calls, [("broadcast", 3)])


class TestAcrossNodeGroup(unittest.TestCase):
    """across_node groups are strided, e.g. [1,5] -- also not 0-based."""

    def test_strided_group(self):
        dist, group, comm = _dist_with_group([1, 5])
        dist.broadcast(object(), src=1, group=group)
        dist.broadcast(object(), src=5, group=group)
        self.assertEqual(comm.calls, [("broadcast", 0), ("broadcast", 1)])

    def test_strided_group_rejects_outsider(self):
        dist, group, _ = _dist_with_group([1, 5])
        with self.assertRaises(ValueError):
            dist.broadcast(object(), src=3, group=group)


class TestCollectivesContract(unittest.TestCase):
    """collectives.py passes min(get_process_group_ranks(group)) as the root.

    That contract is what produces 4 on the second node, so pin it: the
    adapter must report GLOBAL ranks from get_process_group_ranks and accept
    a GLOBAL rank back.
    """

    def test_group_ranks_are_global_and_round_trip(self):
        dist, group, comm = _dist_with_group([4, 5, 6, 7])
        reported = dist.get_process_group_ranks(group)
        self.assertEqual(reported, [4, 5, 6, 7])
        dist.broadcast(object(), src=min(reported), group=group)
        self.assertEqual(comm.calls, [("broadcast", 0)])


class SubgroupStoreRendezvous(unittest.TestCase):
    """R2-3: the rendezvous store must exist for subgroups excluding rank 0.

    Round-2 review finding. The TCPStore was created lazily inside
    _new_comm_for(), which only members of the new subgroup ever reach, while
    is_master was (self.get_rank() == 0). For a subgroup like [4,5,6,7]:

      - rank 0 is not a member, so it returns at the non-member early-return
        and never creates the store server;
      - no member is rank 0, so every member builds a client.

    Result: all four members block connecting to a server that was never
    started -- a hang at group-creation time, not an error. This is the same
    subgroup shape that crashed job 6549.

    The fix moves store creation into new_group, before the non-member
    early-return, so it is collective over the parent group.
    """

    def _dist(self, rank, world=8):
        comm = FakeComm(world, rank=rank)

        # Members reach the native-split attempt before the new_comm
        # fallback. Raise the exact XCCL message so the fallback path -- the
        # one that needs the store -- is the path under test.
        def _split(ranks, label):
            raise RuntimeError(
                "split is not supported now and will be added later")

        comm.split = _split
        dist = tcb.TorchCommsDist.__new__(tcb.TorchCommsDist)
        dist._comm = comm
        dist._splits = {}
        dist._op_map = {}
        dist._backend = "xccl"
        dist._device = "xpu"
        dist._root_store = None
        dist._split_unsupported = None
        return dist

    def _run_new_group(self, dist, groups, store_factory):
        """Call new_group under stubbed torch.distributed + torchcomms.

        Member ranks fall through to _new_comm_for, which calls _require().
        Stubbing the optional torchcomms module keeps this a test of the
        store rendezvous rather than of whether torchcomms is installed.
        """
        fake_tc = mock.MagicMock()
        fake_tc.new_comm.return_value = object()
        # _require() reads the module-global _tc captured at import time, so
        # patching sys.modules['torchcomms'] after the fact has no effect.
        with mock.patch.object(tcb, "_tc", fake_tc), mock.patch.dict(
            os.environ, {"MASTER_ADDR": "h", "MASTER_PORT": "29500"}
        ), mock.patch.dict(
            "sys.modules",
            {"torch": mock.MagicMock(),
             "torch.distributed": mock.MagicMock(
                 TCPStore=store_factory,
                 PrefixStore=lambda prefix, store: object()),
             "torchcomms": fake_tc},
        ):
            for g in groups:
                dist.new_group(g)

    def test_rank0_creates_store_even_when_not_a_member(self):
        dist = self._dist(rank=0)
        made = []

        def fake_store(**kw):
            made.append(kw)
            return object()

        # rank 0 is NOT in this subgroup
        self._run_new_group(dist, [[4, 5, 6, 7]], fake_store)

        self.assertEqual(len(made), 1,
                         "rank 0 must still start the store server")
        self.assertTrue(made[0]["is_master"],
                        "rank 0 must be the store master")

    def test_member_excluding_rank0_does_not_claim_master(self):
        dist = self._dist(rank=4)
        made = []

        def fake_store(**kw):
            made.append(kw)
            return object()

        self._run_new_group(dist, [[4, 5, 6, 7]], fake_store)

        self.assertEqual(len(made), 1)
        self.assertFalse(made[0]["is_master"],
                         "only global rank 0 may be the store master")

    def test_store_created_once_and_reused(self):
        dist = self._dist(rank=0)
        made = []

        def fake_store(**kw):
            made.append(kw)
            return object()

        self._run_new_group(
            dist, [[4, 5, 6, 7], [0, 1, 2, 3], [2, 3]], fake_store)

        self.assertEqual(len(made), 1, "store must be idempotent")

    def test_missing_master_addr_raises_clearly(self):
        dist = self._dist(rank=0)
        env = {k: v for k, v in os.environ.items()
               if k not in ("MASTER_ADDR", "MASTER_PORT")}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.dict(
                 "sys.modules",
                 {"torch": mock.MagicMock(),
                  "torch.distributed": mock.MagicMock()}):
            with self.assertRaises(RuntimeError) as ctx:
                dist.new_group([4, 5, 6, 7])
        self.assertIn("MASTER_ADDR", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
