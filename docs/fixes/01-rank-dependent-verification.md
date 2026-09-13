# Fix 1 — Rank-dependent verification payloads

**Severity:** critical — verification reported broken collectives as correct
**Files:** `dl_comm/verify/payload.py` (new), `dl_comm/analysis/correctness.py`, `dl_comm/comm/collectives.py`, `dl_comm/dl_comm_main.py`
**Tests:** `tests/test_payload.py`, `tests/test_correctness_detects_breakage.py`

## The defect

Every input buffer was filled with `torch.ones()`:

```python
x = torch.ones(num_elems, dtype=_dtype).to(device, non_blocking=True)
```

For a buffer of all ones, most collectives produce output numerically identical
to their input:

- `max`, `min` of all-ones across any number of ranks is one.
- `prod` of all-ones is one.
- `broadcast` of all-ones sends all-ones.
- `allgather`, `alltoall`, `gather`, `scatter` of all-ones move all-ones.

Only `sum` changes the value, and only because the rank count multiplies it.

The consequence is that a collective which transferred no data at all left
every buffer holding exactly the value verification expected. The check could
not fail.

## Measured scope

Each shipped checker was run against a collective replaced by the identity
function — data movement removed entirely — across four ranks:

| collective | op | original verdict on a no-op |
|---|---|---|
| allreduce | sum | detected |
| allreduce | max / min / prod | **reported correct** |
| reduce | sum | detected |
| reduce | max / min / prod | **reported correct** |
| reducescatter | sum | detected |
| reducescatter | max / min | **reported correct** |
| broadcast | — | **reported correct** |
| allgather | — | **reported correct** |
| alltoall | — | **reported correct** |
| gather | — | **reported correct** |
| scatter | — | **reported correct** |
| alltoallsingle | — | **reported correct** |

12 of 15 configurations were unverifiable. `examples/13_for_paper/` runs
`verify_correctness: on` across all ten collectives at 1024–8192 nodes; only
the three `sum` rows carried information.

## The fix

`dl_comm/verify/payload.py` builds a buffer whose value depends on both the
rank and the position within the buffer:

```
value[i] = rank_signature(rank) + position_term(i)
```

`rank_signature` varies per rank so cross-rank movement is observable;
`position_term` varies along the buffer so element reordering and partial
transfers are observable. The expected result is derived analytically per
collective in `dl_comm/analysis/correctness.py`.

Three details are load-bearing:

**Overflow.** A naive `rank + 1` payload overflows float32 under `prod` past
roughly 34 ranks. `choose_moduli()` caps the number of ranks contributing a
factor other than 1.0 and bounds values by dtype, so the expected product stays
finite at 8192 ranks. Verified by
`test_prod_does_not_overflow_float32[8192]`.

**The `min` ordering.** The first implementation assigned rank 0 the smallest
value. `reduce` lands its result on root rank 0, so a `reduce/min` that never
ran left rank 0 already holding the correct minimum — still vacuous. This was
caught by `test_noop_collective_is_detected[reduce-min]` failing during
development, not by inspection. `rank_signature` now inverts the ordering for
`min` so the extremum lives on the highest rank.

**`scatter` sent clones.** The shipped `_scatter` built its send list as
`[tensor] * world_size`, so every destination received an identical chunk and
positional verification was impossible. It now slices a distinct chunk per
destination, offset by `SCATTER_OFFSET` so a receiver that never received
cannot pass by holding its own original payload.

## Verification

`tests/test_correctness_detects_breakage.py` runs every collective twice
against real spawned `gloo` processes:

- `healthy` — the collective runs normally; verification must report zero failures.
- `broken_noop` — the collective is removed; verification **must** fail.

Both assertions must hold. The `broken_noop` half is what distinguishes a real
check from a decorative one.

```
$ python -m pytest tests/test_correctness_detects_breakage.py -q
42 passed in 116.24s
```

All 17 no-op cases are now detected, including the 12 that previously passed.

## Residual limitation

`prod` cannot give every rank a distinct value without overflowing at scale.
Detection there rests on the weaker property that no rank's own buffer equals
the expected product, which is sufficient to catch a no-op but does not
localise which rank failed. This is asserted by
`test_prod_payload_differs_from_its_own_reduction`.
