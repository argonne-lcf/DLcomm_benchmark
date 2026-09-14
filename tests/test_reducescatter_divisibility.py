"""reducescatter must not silently drop the remainder of a non-divisible buffer.

``collectives._reduce_scatter`` splits the input with integer division::

    chunk_size = tensor.numel() // world_size

When the element count is not a multiple of the group size, the trailing
``numel % world_size`` elements are never placed in any chunk. The collective
still runs and still reports the *configured* buffer size, so the bandwidth
figure is computed from more bytes than were actually moved. Nothing fails and
nothing is logged -- the number is simply wrong.

The effect is small on the sizes this benchmark uses (16-64 bytes out of 1-4
MiB) but it is a wrong measurement rather than a noisy one, and it grows with
group size. ``adjust_buffer_size_for_group_divisibility`` now covers
reducescatter so the configured size matches the transferred size.

allgather is covered here too, as a negative case: it has no such requirement
and must be left alone.
"""

import pytest

from dl_comm.config.validation import adjust_buffer_size_for_group_divisibility as adjust

FP32 = 4
BF16 = 2


def _elems(nbytes, elem_size):
    return nbytes // elem_size


class TestReducescatterIsAdjusted:
    def test_non_divisible_buffer_is_adjusted(self):
        # 262144 elements over 12 ranks leaves 4 elements (16 bytes) stranded.
        out, msg = adjust(1048576, 12, "reducescatter", FP32)
        assert out != 1048576
        assert _elems(out, FP32) % 12 == 0
        assert "reducescatter" in msg

    def test_adjusted_size_is_divisible_at_every_scale(self):
        """Both scales under test, and the two dtypes the examples use."""
        for nbytes in (1048576, 2097152, 4194304, 1000000):
            for elem_size in (FP32, BF16):
                for world in (12, 24):
                    out, _ = adjust(nbytes, world, "reducescatter", elem_size)
                    assert _elems(out, elem_size) % world == 0, (
                        f"{nbytes} B / {elem_size} B elems over {world} ranks "
                        f"-> {out} B still leaves a remainder"
                    )

    def test_already_divisible_buffer_is_untouched(self):
        # 750000 elements is divisible by both 12 and 24.
        for world in (12, 24):
            out, msg = adjust(3000000, world, "reducescatter", FP32)
            assert out == 3000000
            assert msg == ""

    def test_adjustment_is_to_the_nearest_valid_size(self):
        """Rounding must not quietly change the measurement scale."""
        for world in (12, 24):
            out, _ = adjust(4194304, world, "reducescatter", FP32)
            assert abs(out - 4194304) < 4194304 * 0.001

    def test_name_matching_is_case_insensitive(self):
        a, _ = adjust(1048576, 12, "reducescatter", FP32)
        b, _ = adjust(1048576, 12, "ReduceScatter", FP32)
        c, _ = adjust(1048576, 12, "REDUCESCATTER", FP32)
        assert a == b == c


class TestAllgatherIsNotAdjusted:
    """allgather has no divisibility requirement; adjusting it would be a bug."""

    def test_allgather_is_left_alone(self):
        for world in (12, 24):
            out, msg = adjust(1048576, world, "allgather", FP32)
            assert out == 1048576
            assert msg == ""

    def test_allgather_untouched_even_when_not_divisible(self):
        out, msg = adjust(1000001, 12, "allgather", FP32)
        assert out == 1000001
        assert msg == ""


class TestExistingBehaviourPreserved:
    """The alltoall cases that already worked must keep working."""

    @pytest.mark.parametrize("name", ["alltoallsingle", "alltoallv"])
    def test_alltoall_still_adjusted(self, name):
        for world in (12, 24):
            out, _ = adjust(1048576, world, name, FP32)
            assert _elems(out, FP32) % world == 0

    @pytest.mark.parametrize("name", ["allreduce", "broadcast", "reduce", "barrier"])
    def test_unrelated_collectives_untouched(self, name):
        out, msg = adjust(1048577, 12, name, FP32)
        assert out == 1048577
        assert msg == ""
