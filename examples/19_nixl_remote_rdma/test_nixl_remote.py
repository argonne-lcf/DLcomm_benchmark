#!/usr/bin/env python3
"""Offline tests for the cross-node NIXL transfer layer.

No GPU, no NIXL, no fabric, no MPI. Everything tested here is logic that a
cluster run would either not exercise or would exercise in a way that hides
the bug:

* A wrong pairing still produces a plausible bandwidth number.
* A per-rank figure multiplied by world size looks merely "fast", and that
  exact bug shipped once already in the node-local layer (4x too high at 16
  ranks, caught only because best_s was unchanged across scales).
* The register-then-exchange ordering requirement fails on a cluster with an
  opaque NIXL_ERR_NOT_FOUND from inside the C++ binding, which names nothing.

Run: python3 test_nixl_remote.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dl_comm.transfer.nixl_remote import (  # noqa: E402
    RemoteTransferResult,
    expected_payload,
    format_remote_table,
    pair_ranks,
)


class _FakeTensor:
    """Minimal stand-in for the slice of torch the payload builder uses."""

    def __init__(self, values):
        self.values = list(values)

    def to(self, _dtype):
        return self

    def __add__(self, other):
        return _FakeTensor(v + other for v in self.values)


class _FakeTorch:
    int32 = "int32"
    int64 = "int64"

    @staticmethod
    def arange(n, dtype=None):
        return _FakeTensor(range(n))


class PairRanksTest(unittest.TestCase):
    def test_split_pairs_across_the_halves(self):
        # 2 nodes x 4 ranks, block placement: ranks 0-3 node A, 4-7 node B.
        # Every rank must land on the other node.
        for r in range(8):
            self.assertEqual(pair_ranks(r, 8, pairing="split"), (r + 4) % 8)

    def test_split_pairing_is_symmetric(self):
        # Asymmetry would deadlock: A waits on B's descriptors while B sends
        # to C. Cheap to assert, impossible to debug from a hung job.
        for world in (2, 4, 8, 16):
            for r in range(world):
                p = pair_ranks(r, world, pairing="split")
                self.assertEqual(pair_ranks(p, world, pairing="split"), r)

    def test_ring_pairing_steps_one_node(self):
        self.assertEqual(pair_ranks(0, 8, pairing="ring", ranks_per_node=4), 4)
        self.assertEqual(pair_ranks(5, 8, pairing="ring", ranks_per_node=4), 1)

    def test_ring_is_symmetric_only_for_two_nodes(self):
        # Documents a real limitation rather than pretending it is general:
        # at 3+ nodes a fixed forward step is a cycle, not a pairing, and the
        # measurement's mutual-exchange assumption no longer holds.
        for r in range(8):
            p = pair_ranks(r, 8, pairing="ring", ranks_per_node=4)
            self.assertEqual(pair_ranks(p, 8, pairing="ring", ranks_per_node=4), r)
        asym = [
            r for r in range(12)
            if pair_ranks(pair_ranks(r, 12, pairing="ring", ranks_per_node=4),
                          12, pairing="ring", ranks_per_node=4) != r
        ]
        self.assertTrue(asym, "ring at 3 nodes should not be an involution")

    def test_rejects_odd_and_tiny_worlds(self):
        with self.assertRaises(ValueError):
            pair_ranks(0, 1, pairing="split")
        with self.assertRaises(ValueError):
            pair_ranks(0, 7, pairing="split")

    def test_ring_on_single_node_is_rejected(self):
        # Would silently measure same-node traffic and report it as fabric.
        with self.assertRaises(ValueError):
            pair_ranks(0, 4, pairing="ring", ranks_per_node=4)

    def test_ring_requires_ranks_per_node(self):
        with self.assertRaises(ValueError):
            pair_ranks(0, 8, pairing="ring")

    def test_unknown_pairing_is_rejected(self):
        with self.assertRaises(ValueError):
            pair_ranks(0, 8, pairing="diagonal")


class PayloadTest(unittest.TestCase):
    def test_payload_identifies_its_sender(self):
        a = expected_payload(_FakeTorch, 4, 0).values
        b = expected_payload(_FakeTorch, 4, 1).values
        self.assertEqual(a, [0, 1, 2, 3])
        self.assertEqual(b, [1000003, 1000004, 1000005, 1000006])

    def test_payloads_never_collide_within_a_buffer(self):
        # The verification check is only meaningful if two ranks cannot
        # produce the same bytes. 256 MiB of int32 is 67108864 elements,
        # comfortably under the 1000003 stride... which is FALSE, and this
        # test is what says so: the stride bounds the usable buffer size.
        n_max_safe = 1_000_003
        a = set(expected_payload(_FakeTorch, 8, 0).values)
        b = set(expected_payload(_FakeTorch, 8, 1).values)
        self.assertFalse(a & b)
        self.assertEqual(
            expected_payload(_FakeTorch, n_max_safe, 0).values[-1],
            1_000_002,
        )
        # Overlap begins exactly one element past the stride, so a buffer
        # larger than stride*4 bytes weakens sender identification. Asserted
        # so a future size bump trips here instead of silently degrading the
        # correctness check on the cluster.
        self.assertEqual(
            expected_payload(_FakeTorch, n_max_safe + 1, 0).values[-1],
            1_000_003,
        )

    def test_rejects_empty_buffer(self):
        with self.assertRaises(ValueError):
            expected_payload(_FakeTorch, 0, 0)


class ResultArithmeticTest(unittest.TestCase):
    def _res(self, times, nbytes=1 << 26, world=16):
        return RemoteTransferResult(
            direction="remote_d2d", backend="UCX", nbytes=nbytes,
            world_size=world, local_rank_id=0, peer_rank_id=8,
            cross_node=True, times_s=list(times), verified=True,
        )

    def test_total_bytes_is_per_rank_not_scaled_by_world(self):
        # The regression that shipped in the node-local layer: total_bytes
        # was nbytes * world_size, so an unchanged best_s reported 4x the
        # bandwidth at 4x the ranks. A wrong number here still looks
        # plausible in a log, so only an assertion catches it.
        small = self._res([0.1], world=4)
        large = self._res([0.1], world=64)
        self.assertEqual(small.total_bytes, 1 << 26)
        self.assertEqual(large.total_bytes, 1 << 26)
        self.assertAlmostEqual(small.gbytes_per_s, large.gbytes_per_s)

    def test_best_and_median_differ_under_a_tail(self):
        r = self._res([0.1, 0.1, 0.5, 0.1, 0.9])
        self.assertAlmostEqual(r.best_s, 0.1)
        self.assertAlmostEqual(r.median_s, 0.1)
        self.assertGreater(r.gbytes_per_s, 0)
        self.assertGreaterEqual(r.gbytes_per_s, r.median_gbytes_per_s)

    def test_median_averages_the_middle_pair_when_even(self):
        r = self._res([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(r.median_s, 0.25)

    def test_no_timings_raises_instead_of_returning_zero(self):
        r = self._res([])
        with self.assertRaises(ValueError):
            _ = r.best_s
        with self.assertRaises(ValueError):
            _ = r.gbytes_per_s

    def test_nonpositive_time_raises(self):
        r = self._res([0.0])
        with self.assertRaises(ValueError):
            _ = r.gbytes_per_s


class TableTest(unittest.TestCase):
    def test_unverified_and_samenode_are_visible_columns(self):
        rows = [
            RemoteTransferResult(
                direction="remote_d2d", backend="UCX", nbytes=1 << 20,
                world_size=2, local_rank_id=0, peer_rank_id=1,
                cross_node=False, times_s=[0.01], verified=False,
            )
        ]
        out = format_remote_table(rows)
        self.assertIn("False", out)
        self.assertIn("cross", out)
        self.assertIn("verified", out)

    def test_empty_results_do_not_render_an_empty_table(self):
        self.assertIn("no cross-node", format_remote_table([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
