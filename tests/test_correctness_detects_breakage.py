"""Verification must detect a broken collective (fix #1).

The original all-ones payload made 12 of 15 collective/op configurations
impossible to verify: a collective that transferred nothing produced output
identical to its input, so the check passed.

Each test below runs the SAME collective twice against real gloo processes:

  healthy      -> verification must report zero failures
  broken_noop  -> the collective is skipped entirely; verification MUST fail

A verifier that passes both runs is vacuous, and the ``broken_noop`` assertion
is what makes these tests non-trivial. See
``docs/fixes/01-rank-dependent-verification.md``.
"""

import pytest

from gloo_harness import run_gloo, total_failures, total_checks, errors

WORLD = 4

# (collective, op) pairs covering every configuration the benchmark supports.
REDUCTIONS = [
    ("allreduce", "sum"),
    ("allreduce", "max"),
    ("allreduce", "min"),
    ("allreduce", "prod"),
    ("reduce", "sum"),
    ("reduce", "max"),
    ("reduce", "min"),
    ("reduce", "prod"),
    ("reducescatter", "sum"),
    ("reducescatter", "max"),
    ("reducescatter", "min"),
]

MOVEMENT = [
    ("broadcast", None),
    ("allgather", None),
    ("alltoall", None),
    ("gather", None),
    ("scatter", None),
    ("alltoallsingle", None),
]

ALL_CONFIGS = REDUCTIONS + MOVEMENT


@pytest.mark.gloo
@pytest.mark.parametrize("collective,op", ALL_CONFIGS,
                         ids=[f"{c}-{o}" for c, o in ALL_CONFIGS])
def test_healthy_collective_passes(collective, op):
    """A correctly functioning collective must verify clean."""
    results = run_gloo(collective, op, world_size=WORLD, num_elems=16,
                       mode="healthy")
    assert not errors(results), errors(results)[:1]
    assert len(results) == WORLD, f"only {len(results)} of {WORLD} ranks reported"
    assert total_checks(results) > 0, "no verification actually ran"
    assert total_failures(results) == 0, (
        f"{collective}/{op}: healthy run reported "
        f"{total_failures(results)} failures: "
        f"{[r.get('details') for r in results]}")


@pytest.mark.gloo
@pytest.mark.parametrize("collective,op", ALL_CONFIGS,
                         ids=[f"{c}-{o}" for c, o in ALL_CONFIGS])
def test_noop_collective_is_detected(collective, op):
    """THE REGRESSION TEST: skipping the collective entirely must be caught.

    This is the exact defect the all-ones payload could not see. If this test
    ever passes with zero failures, verification has become vacuous again.
    """
    results = run_gloo(collective, op, world_size=WORLD, num_elems=16,
                       mode="broken_noop")
    assert not errors(results), errors(results)[:1]
    assert len(results) == WORLD, f"only {len(results)} of {WORLD} ranks reported"
    assert total_failures(results) > 0, (
        f"{collective}/{op}: a collective that did NOTHING was reported as "
        f"correct -- verification is vacuous for this configuration")


@pytest.mark.gloo
@pytest.mark.parametrize("collective,op", [
    ("allreduce", "sum"),
    ("allgather", None),
    ("alltoall", None),
])
def test_corrupted_rank_is_detected(collective, op):
    """Corrupting one rank's data after a real collective must be caught."""
    results = run_gloo(collective, op, world_size=WORLD, num_elems=16,
                       mode="broken_corrupt")
    assert not errors(results), errors(results)[:1]
    assert total_failures(results) > 0, (
        f"{collective}/{op}: corruption on one rank went undetected")


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 3, 8])
def test_detection_holds_at_other_group_sizes(world_size):
    """Detection must not depend on a particular world size."""
    healthy = run_gloo("allreduce", "max", world_size=world_size, num_elems=16,
                       mode="healthy")
    broken = run_gloo("allreduce", "max", world_size=world_size, num_elems=16,
                      mode="broken_noop")
    assert not errors(healthy), errors(healthy)[:1]
    assert not errors(broken), errors(broken)[:1]
    assert total_failures(healthy) == 0
    assert total_failures(broken) > 0


@pytest.mark.gloo
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_detection_holds_for_dtypes(dtype_name):
    healthy = run_gloo("allreduce", "sum", world_size=WORLD, num_elems=16,
                       mode="healthy", dtype_name=dtype_name)
    broken = run_gloo("allreduce", "sum", world_size=WORLD, num_elems=16,
                      mode="broken_noop", dtype_name=dtype_name)
    assert not errors(healthy), errors(healthy)[:1]
    assert total_failures(healthy) == 0
    assert total_failures(broken) > 0
