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

import unittest
from unittest import mock

import dl_comm.comm.torchcomms_backend as tcb


class FakeComm:
    """Minimal torchcomms communicator that enforces the real rank contract.

    The point of the test is that torchcomms REJECTS an out-of-range root,
    so the fake must reject it too -- a permissive stub would pass against
    the buggy code and prove nothing.
    """

    def __init__(self, size):
        self._size = size
        self.calls = []

    def get_size(self):
        return self._size

    def get_rank(self):
        return 0

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
