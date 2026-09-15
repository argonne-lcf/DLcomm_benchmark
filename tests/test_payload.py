"""Payload construction properties (fix #1).

These are the invariants the correctness checks depend on. See
``docs/fixes/01-rank-dependent-verification.md``.
"""

import pytest

torch = pytest.importorskip("torch")

from dl_comm.verify.payload import (
    build_payload,
    choose_moduli,
    expected_reduction,
    rank_signature,
    scatter_source,
)


@pytest.mark.parametrize("group_size", [2, 4, 8, 12, 1024])
@pytest.mark.parametrize("op_name", ["sum", "max", "min"])
def test_ranks_produce_distinguishable_payloads(group_size, op_name):
    """The defect in one sentence: all-ones made every rank identical."""
    payloads = [
        build_payload(torch, 32, torch.float32, r, group_size, op_name)
        for r in range(min(group_size, 8))
    ]
    distinct = {tuple(p.tolist()) for p in payloads}
    assert len(distinct) > 1, (
        f"{op_name} at group_size={group_size}: every rank built the same "
        f"buffer, so a no-op collective would be indistinguishable")


@pytest.mark.parametrize("group_size", [2, 4, 8, 12, 1024])
def test_prod_payload_differs_from_its_own_reduction(group_size):
    """prod cannot give every rank a distinct value without overflowing.

    Detection instead rests on the weaker but sufficient property that a rank's
    own buffer never equals the expected product, so a collective that never
    ran leaves a value the checker rejects.
    """
    rank_mod, _ = choose_moduli(torch.float32, group_size, "prod")
    expected = expected_reduction("prod", group_size, rank_mod)
    for r in range(min(group_size, 8)):
        x = build_payload(torch, 8, torch.float32, r, group_size, "prod")
        assert not torch.allclose(x, torch.full_like(x, expected)), (
            f"rank {r} already holds the expected product; a no-op prod "
            f"would pass verification")


@pytest.mark.parametrize("op_name", ["sum", "max", "min", "prod"])
def test_payload_is_not_all_ones(op_name):
    x = build_payload(torch, 32, torch.float32, 1, 8, op_name)
    assert not torch.all(x == 1.0), "payload degenerated to the original all-ones"


def test_min_extremum_is_not_on_rank_zero():
    """reduce/min lands on root 0; if rank 0 held the min it could not fail.

    Regression guard for the vacuous case found by
    test_noop_collective_is_detected[reduce-min].
    """
    group_size, mod = 4, 4
    sigs = [rank_signature(r, group_size, mod, "min") for r in range(group_size)]
    assert sigs[0] != min(sigs), (
        "rank 0 holds the group minimum, making reduce/min unverifiable")


def test_max_extremum_is_not_on_rank_zero_only():
    group_size, mod = 4, 4
    sigs = [rank_signature(r, group_size, mod, "max") for r in range(group_size)]
    assert sigs[0] != max(sigs) or group_size == 1


@pytest.mark.parametrize("group_size", [2, 4, 8, 64, 1024, 8192])
def test_prod_does_not_overflow_float32(group_size):
    """Naive rank+1 products overflow float32 past ~34 ranks."""
    _, mod = choose_moduli(torch.float32, group_size, "prod")
    value = expected_reduction("prod", group_size, mod)
    assert value == value, "product overflowed to NaN"
    assert value not in (float("inf"), float("-inf")), "product overflowed to inf"
    assert value < 3.4e38


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "bfloat16", "float16"])
def test_payload_builds_for_supported_dtypes(dtype_name):
    dtype = getattr(torch, dtype_name)
    x = build_payload(torch, 16, dtype, 2, 8, "sum")
    assert x.dtype == dtype
    assert x.numel() == 16
    assert torch.isfinite(x.float()).all()


@pytest.mark.parametrize("op_name", ["sum", "max", "min", "prod"])
def test_expected_reduction_matches_bruteforce(op_name):
    group_size = 8
    _, mod = choose_moduli(torch.float32, group_size, op_name)
    sigs = [rank_signature(r, group_size, mod, op_name) for r in range(group_size)]
    if op_name == "sum":
        want = sum(sigs)
    elif op_name == "max":
        want = max(sigs)
    elif op_name == "min":
        want = min(sigs)
    else:
        want = 1.0
        for s in sigs:
            want *= s
    assert expected_reduction(op_name, group_size, mod) == pytest.approx(want)


def test_scatter_source_chunks_are_distinct():
    """scatter must send a different chunk to each rank to be verifiable."""
    world, per_rank = 4, 8
    rank_mod, pos_mod = choose_moduli(torch.float32, world, None)
    chunks = []
    for dest in range(world):
        chunk = scatter_source(torch, per_rank, torch.float32, dest, world,
                               rank_mod, pos_mod)
        chunks.append(tuple(chunk.tolist()))
    assert len(set(chunks)) == world, "scatter chunks are not distinguishable"


def test_payload_positional_term_varies_within_buffer():
    """A buffer constant along its length cannot detect element reordering."""
    x = build_payload(torch, 64, torch.float32, 1, 8, "sum")
    assert len(set(x.tolist())) > 1, "payload is constant across elements"
