"""Rank-dependent verification payloads for DLcomm.

Rationale
---------
The original benchmark filled every input buffer with ``torch.ones()``. For every
non-SUM reduction and every data-movement collective the expected output of an
all-ones input is *also* all-ones, so a collective that moved no data at all
still passed verification. See ``docs/fixes/01-rank-dependent-verification.md``.

This module builds payloads whose value depends on **which rank** produced them
and on **which element position** within the buffer, so that:

  * a no-op / dropped transfer is detected (values differ per rank),
  * a mis-ordered gather or all-to-all is detected (position matters),
  * a wrong-chunk reduce-scatter is detected (position matters).

Payload shape
-------------
For a rank at index ``i`` within its process group::

    x[idx] = rank_signature(i) + position_signature(idx)

``rank_signature`` is a small bounded integer (1..K) and ``position_signature``
is ``idx % Q``. Both are kept deliberately small so that reduction results stay
exactly representable in the benchmark's floating point dtypes: the largest
value any check has to represent is about ``group_size * (K + Q)``.

For PRODUCT reductions the signature is restricted to ``{1, 2}`` over a bounded
number of contributing ranks, because a product over thousands of ranks
overflows any supported dtype.
"""

from __future__ import annotations

# Default moduli. Kept small on purpose: the reduction of a rank signature over
# `n` ranks must stay exactly representable in the payload dtype.
DEFAULT_RANK_MODULUS = 8
DEFAULT_POSITION_MODULUS = 4

# Number of ranks that contribute a factor of 2 to a PRODUCT reduction.
# 2**PROD_CONTRIBUTORS must stay far inside float32/float16 range.
PROD_CONTRIBUTORS = 10

# Offset applied to scatter source data so a receiving rank cannot accidentally
# match its own pre-existing buffer contents.
SCATTER_OFFSET = DEFAULT_RANK_MODULUS + 1

# (rtol, atol) per dtype name. Low-precision dtypes get loose tolerances because
# the hardware reduction itself is lossy and order-dependent.
_TOLERANCE_BY_NAME = {
    "torch.float64": (1e-12, 0.0),
    "torch.float32": (1e-6, 0.0),
    "torch.float16": (5e-3, 1e-2),
    "torch.bfloat16": (4e-2, 1e-1),
    "torch.int32": (0.0, 0.0),
    "torch.int64": (0.0, 0.0),
}

# Largest finite value per dtype, used to keep expected results from overflowing.
_MAX_FINITE_BY_NAME = {
    "torch.float64": 1.7e308,
    "torch.float32": 3.4e38,
    "torch.float16": 65504.0,
    "torch.bfloat16": 3.3e38,
    "torch.int32": 2**31 - 1,
    "torch.int64": 2**63 - 1,
}

# Largest integer that is exactly representable (2**mantissa_bits).
_EXACT_INT_BY_NAME = {
    "torch.float64": 2**53,
    "torch.float32": 2**24,
    "torch.float16": 2**11,
    "torch.bfloat16": 2**8,
    "torch.int32": 2**31 - 1,
    "torch.int64": 2**63 - 1,
}


def _dtype_key(dtype) -> str:
    return str(dtype)


def tolerance_for(dtype) -> tuple[float, float]:
    """Return ``(rtol, atol)`` appropriate for comparing results of this dtype."""
    return _TOLERANCE_BY_NAME.get(_dtype_key(dtype), (1e-6, 0.0))


def is_exact_dtype(dtype) -> bool:
    return _dtype_key(dtype) in ("torch.int32", "torch.int64")


def choose_moduli(dtype, group_size: int, op_name: str | None) -> tuple[int, int]:
    """Pick ``(rank_modulus, position_modulus)`` that cannot overflow ``dtype``.

    The worst case a checker must represent is a SUM reduction, whose magnitude
    is bounded by ``group_size * (rank_modulus + position_modulus)``. We shrink
    the moduli until that bound fits comfortably inside the dtype, while never
    collapsing the rank modulus below 2 (a rank modulus of 1 would make every
    rank identical and bring back the vacuous-verification bug).
    """
    if op_name == "prod":
        # Product mode uses the {1, 2} signature and no positional term.
        return 2, 1

    name = _dtype_key(dtype)
    max_finite = _MAX_FINITE_BY_NAME.get(name, 3.4e38)
    # Stay an order of magnitude away from both the overflow limit and the
    # exact-integer limit where we can.
    budget = min(max_finite / 8.0, float(_EXACT_INT_BY_NAME.get(name, 2**24)))

    k, q = DEFAULT_RANK_MODULUS, DEFAULT_POSITION_MODULUS
    group_size = max(1, int(group_size))
    while k > 2 and group_size * (k + q) > budget:
        k //= 2
        q = max(1, q // 2)
    if group_size * (k + q) > budget:
        q = 1
    return max(2, k), max(1, q)


def rank_signature(rank_index: int, group_size: int, rank_modulus: int,
                   op_name: str | None) -> float:
    """Scalar contributed by the rank at ``rank_index`` within its group.

    For ``min`` the ordering is deliberately inverted so that the extremum does
    NOT live on rank 0. ``reduce`` lands its result on root rank 0, so if rank 0
    already held the minimum, a ``reduce/min`` that never ran would leave rank 0
    holding the correct answer and the check could not fail. This was caught by
    ``test_noop_collective_is_detected[reduce-min]``; see
    docs/fixes/01-rank-dependent-verification.md.
    """
    if op_name == "prod":
        contributors = min(group_size, PROD_CONTRIBUTORS)
        return 2.0 if rank_index < contributors else 1.0
    if op_name == "min":
        # Descending: rank 0 holds the LARGEST value, so the group minimum is
        # only reachable by actually communicating with a higher rank.
        return float(1 + ((group_size - 1 - rank_index) % rank_modulus))
    return float(1 + (rank_index % rank_modulus))


def expected_reduction(op_name: str, group_size: int, rank_modulus: int) -> float:
    """Reduction of ``rank_signature`` over the whole group (positional term excluded)."""
    sigs = [rank_signature(i, group_size, rank_modulus, op_name)
            for i in range(group_size)]
    if op_name == "sum":
        return float(sum(sigs))
    if op_name == "max":
        return float(max(sigs))
    if op_name == "min":
        return float(min(sigs))
    if op_name == "prod":
        out = 1.0
        for s in sigs:
            out *= s
        return out
    raise ValueError(f"unsupported reduction op '{op_name}'")


def build_payload(torch, num_elems, dtype, rank_index, group_size, op_name,
                  device=None, rank_modulus=None, position_modulus=None):
    """Create the input buffer this rank contributes to the collective."""
    if rank_modulus is None or position_modulus is None:
        rank_modulus, position_modulus = choose_moduli(dtype, group_size, op_name)

    sig = rank_signature(rank_index, group_size, rank_modulus, op_name)
    base = torch.arange(num_elems, dtype=torch.float64)
    pos = base % position_modulus if position_modulus > 1 else torch.zeros_like(base)
    values = pos + sig
    out = values.to(dtype)
    if device is not None:
        out = out.to(device, non_blocking=True)
    return out


def position_term(torch, num_elems, position_modulus, offset=0):
    """``(idx + offset) % position_modulus`` as a float64 tensor."""
    base = torch.arange(num_elems, dtype=torch.float64) + offset
    if position_modulus <= 1:
        return torch.zeros_like(base)
    return base % position_modulus


def scatter_source(torch, num_elems, dtype, dest_index, group_size, rank_modulus,
                   position_modulus, device=None):
    """Buffer the scatter root sends to the group member at ``dest_index``.

    Offset by :data:`SCATTER_OFFSET` so that a receiver that never actually
    received anything cannot pass by holding its own original payload.
    """
    sig = rank_signature(dest_index, group_size, rank_modulus, None) + SCATTER_OFFSET
    pos = position_term(torch, num_elems, position_modulus)
    out = (pos + sig).to(dtype)
    if device is not None:
        out = out.to(device, non_blocking=True)
    return out
