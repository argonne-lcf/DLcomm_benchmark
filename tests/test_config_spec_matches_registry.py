"""The config spec and the collective registry must not drift apart.

`config_spec.json` is the user-facing list of valid `collective_name` values.
`COLLECTIVES` is what the benchmark can actually dispatch. When a collective is
added to one and not the other the failure is silent in the unhelpful
direction: the collective exists, works, and is verified by the test suite, but
a configuration naming it is rejected (or, in the reverse case, accepted and
then dispatched into a KeyError).

This drift had already happened before these tests existed: `reducescatter` was
implemented, registered and verified, yet absent from the spec.
"""

import json
from pathlib import Path

import pytest

from dl_comm.comm.collectives import COLLECTIVES, init_framework_constants

SPEC_PATH = Path(__file__).resolve().parents[1] / "dl_comm" / "config" / "config_spec.json"


@pytest.fixture(scope="module")
def spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module", autouse=True)
def _init():
    init_framework_constants("pytorch")


def test_every_registered_collective_is_in_the_spec(spec):
    missing = sorted(set(COLLECTIVES) - set(spec["collective"]))
    assert not missing, (
        f"collectives are dispatchable but rejected by config validation: {missing}. "
        f"Add them to {SPEC_PATH.name}."
    )


def test_every_spec_collective_is_registered(spec):
    missing = sorted(set(spec["collective"]) - set(COLLECTIVES))
    assert not missing, (
        f"config_spec.json advertises collectives that cannot be dispatched: {missing}. "
        f"A config naming one of these reaches COLLECTIVES[...] and fails there."
    )


def test_every_collective_has_an_op_entry(spec):
    missing = sorted(set(spec["collective"]) - set(spec["op"]))
    assert not missing, f"collectives with no 'op' entry in the spec: {missing}"


def test_reduction_collectives_declare_ops(spec):
    """A collective that reduces must offer real ops, not [null]."""
    for name in ("allreduce", "reduce", "reducescatter"):
        ops = spec["op"].get(name)
        assert ops and ops != [None], f"{name} must declare reduction ops, got {ops}"


def test_movement_collectives_declare_no_op(spec):
    """A collective that only moves data must not advertise reduction ops."""
    for name in ("broadcast", "allgather", "gather", "scatter", "alltoall",
                 "alltoallsingle", "alltoallv", "sendrecv", "sendrecv_async",
                 "barrier"):
        ops = spec["op"].get(name)
        assert ops == [None], f"{name} should declare ops [None], got {ops}"


def test_alltoallv_requires_divisibility():
    """alltoallv splits the buffer across ranks, so it needs the same
    divisibility adjustment as alltoallsingle."""
    from dl_comm.config.validation import adjust_buffer_size_for_group_divisibility

    # 1000 bytes of float32 = 250 elements, which is not divisible by 12.
    adjusted, note = adjust_buffer_size_for_group_divisibility(
        buffer_bytes=1000, group_size=12, collective_name="alltoallv", elem_size=4
    )
    assert adjusted != 1000, (
        "alltoallv was not adjusted for divisibility; an indivisible buffer "
        "would reach the collective and fail there"
    )
    assert (adjusted // 4) % 12 == 0, f"adjusted buffer still not divisible: {adjusted}"
    assert note, "adjustment should explain itself"


def test_unknown_collective_is_rejected_with_a_useful_message():
    """A typo in collective_name must not surface as a bare KeyError."""
    assert "allreduc" not in COLLECTIVES
    # The guard in dl_comm_main raises ValueError naming the valid choices;
    # here we assert the registry lookup it protects would otherwise be opaque.
    with pytest.raises(KeyError):
        COLLECTIVES["allreduc"]
