# DLcomm hardening — overview

Six defects were found by executing the shipped modules against instrumented
stand-ins, not by reading them. Each has its own document, a fix, and tests
that fail if the fix is reverted.

| # | Defect | Severity | Document |
|---|---|---|---|
| 1 | Verification vacuous for 12 of 15 collective/op configs | critical | [01](01-rank-dependent-verification.md) |
| 2 | Across-node bandwidth wrong by `nodes/devices_per_node` | high | [02](02-bandwidth-group-size.md) |
| 3 | Surplus ranks silently excluded from all groups | high | [03](03-rank-topology-validation.md) |
| 4 | Correctness failure did not fail the job | high | [04](04-fail-loudly.md) |
| 5 | Timed region measured enqueue, not completion | medium-high | [05](05-timing-and-statistics.md) |
| 6 | Dependency and license metadata | low | [06](06-packaging.md) |

## Impact on published results

Fixes 1, 2, and 5 change what the benchmark reports.

- **Fix 2 changes numbers directly.** Across-node bandwidth was scaled by
  `num_compute_nodes / num_devices_per_node` — 256× at the 1024-node,
  4-device-per-node geometry in `examples/13_for_paper/`. Those results need
  regenerating.
- **Fix 1 changes confidence, not numbers.** Runs that reported correct may
  simply never have been checked. A clean verification result from before this
  change carries no information for any collective other than the three `sum`
  cases.
- **Fix 5 changes numbers when `CCL_OP_SYNC=1` was not set**, and separates
  iteration 0 from steady-state statistics in all cases.

## Testing

```
python -m pytest tests -q                 # full suite
python -m pytest tests -q -m "not gloo"   # unit only, ~2 s
```

The suite needs no GPU, no allocation, and no MPI launcher — it runs on a login
node or in CI. `tests/test_correctness_detects_breakage.py` spawns real
`torch.distributed` gloo processes and runs each collective twice: once
working, once with the collective removed. The second run asserts verification
**fails**. That assertion is the point of the suite; a check that passes both
runs is decorative.

Local result: **136 passed**.

## Two bugs the tests found in the fixes themselves

Worth recording because both were invisible to inspection.

1. **`reduce/min` was still vacuous** after the first payload implementation.
   `reduce` lands on root rank 0, and the payload gave rank 0 the smallest
   value, so a `reduce/min` that never ran left rank 0 holding the correct
   answer. Caught by `test_noop_collective_is_detected[reduce-min]`.

2. **`config_hash()` was non-deterministic.** Its fallback used `repr(cfg)`,
   which embeds the object's memory address, so the same config hashed twice
   gave different digests. Caught by
   `test_config_hash_is_stable_and_sensitive`.

## On-hardware validation

`validate_on_aurora.sh` runs on 2 Aurora nodes (24 XPU ranks) and proves three
things the CPU suite cannot:

- **A** — the suite passes under the Aurora `frameworks` module stack;
- **B** — a real XCCL run completes and verification passes, with
  `CCL_OP_SYNC=0` so the tool must synchronize itself;
- **C** — with `allreduce` replaced by a no-op at runtime, verification
  **fails** and the job exits non-zero.

(C) is the load-bearing case. (B) passing on its own would not distinguish a
working check from one that cannot fail.

The sabotage in (C) is injected via `sitecustomize.py` on `PYTHONPATH`, so the
repository tree under test stays unmodified.
