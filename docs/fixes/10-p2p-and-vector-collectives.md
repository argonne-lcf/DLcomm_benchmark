# Feature 10 — Point-to-point and vector collectives

**Status:** implemented; sendrecv and alltoallv run on hardware

**Type:** feature
**Files:** `dl_comm/comm/collectives.py`, `dl_comm/analysis/correctness.py`,
`dl_comm/analysis/bandwidth.py`, `tests/gloo_harness.py`,
`tests/test_correctness_detects_breakage.py`

## Gap

The registry covered ten collectives. Two classes of operation that
communication benchmarks are normally expected to measure were absent:

1. **Vector (uneven) exchange.** `alltoall` and `alltoallsingle` both move an
   identical element count between every pair of ranks. Real workloads —
   mixture-of-experts routing, unbalanced sharding, ragged batches — send a
   different count to each peer. That path uses different code inside the
   backend and has different performance characteristics.

2. **Point-to-point.** `send`/`recv` did not appear anywhere in the source.
   Pairwise latency and bandwidth, the measurement `osu_latency` and `osu_bw`
   exist to provide, could not be expressed by any configuration.

## What was added

| Name | Operation | busbw factor |
|---|---|---|
| `alltoallv` | `all_to_all_single` with uneven `input_split_sizes` / `output_split_sizes` | `(n-1)/n` |
| `sendrecv` | blocking pairwise `dist.send` / `dist.recv` | `1.0` |
| `sendrecv_async` | non-blocking `dist.isend` / `dist.irecv`, both directions in flight | `2.0` |

### Split table

`_uneven_splits` produces a deterministic, deliberately skewed split that all
ranks compute identically, so send and receive sides agree without extra
communication:

```
ws=2   [341, 683]                                            sum=1024
ws=4   [102, 204, 307, 411]                                  sum=1024
ws=12  [13, 26, 39, 52, 65, 78, 91, 105, 118, 131, 144, 162] sum=1024
```

The skew matters. An equal split would let an implementation that ignores the
split arguments pass by coincidence; with unequal shares it produces wrong
sizes instead.

### Pairing and the odd-rank case

Ranks pair as (0,1), (2,3), … The even member sends first, the odd member
receives first — symmetric blocking sends would deadlock. When the group has an
odd rank count the final rank has no partner. It records a **skip**, not a
pass, so a configuration in which nothing is exchanged cannot report clean.

### busbw for point-to-point

Group size does not enter the p2p factor: the measurement involves exactly two
ranks regardless of how large the group is. `sendrecv` moves the buffer once, so
busbw equals algbw. `sendrecv_async` has both directions in flight over the same
elapsed time, so the bus carries twice the payload.

## Verification

All three are verified against a computed expectation, not merely executed:

- `alltoallv` — checks the received element count equals
  `my_share × world_size` and that each chunk holds the exact slice the
  corresponding peer assigned to this rank.
- `sendrecv` / `sendrecv_async` — checks the rank holds **its partner's**
  payload, which is what distinguishes a real exchange from a no-op.

## Pitfall found while adding these

The first implementation returned `None` from the test harness's
`_shape_only_result` for the three new operations. The no-op regression test
failed:

```
sendrecv_async/None: a collective that did NOTHING was reported as correct
  -- verification is vacuous for this configuration
assert 0 > 0
```

`None` made the checker record a *skip* rather than a *failure*, so a dead
collective looked clean — exactly the defect documented in
`01-payload-verification.md`, reintroduced in new code. The harness now returns
a correctly shaped buffer holding the rank's own data, and all three no-op
cases are detected.

This is why the no-op test exists: it caught a vacuous verifier in code written
by the same person who wrote the rule against vacuous verifiers.

## Result

Local suite: **142 passed** (was 136). The six added cases are the three new
operations in both healthy and no-op mode.
