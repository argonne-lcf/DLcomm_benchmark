# DLcomm hardening — overview

Nine defects were found by executing the shipped modules — six against
instrumented stand-ins, and three more that appeared only when the benchmark
was run on Aurora hardware. Each has its own document, a fix, and tests that
fail if the fix is reverted.

| # | Defect | Severity | Document |
|---|---|---|---|
| 1 | Verification vacuous for 12 of 15 collective/op configs | critical | [01](01-rank-dependent-verification.md) |
| 2 | Across-node bandwidth wrong by `nodes/devices_per_node` | high | [02](02-bandwidth-group-size.md) |
| 3 | Surplus ranks silently excluded from all groups | high | [03](03-rank-topology-validation.md) |
| 4 | Correctness failure did not fail the job | high | [04](04-fail-loudly.md) |
| 5 | Timed region measured enqueue, not completion | medium-high | [05](05-timing-and-statistics.md) |
| 6 | Dependency and license metadata | low | [06](06-packaging.md) |
| 7 | `RUN_LOG_DIR` KeyError outside the bundled jobscripts | medium | [07](07-log-dir-fallback.md) |
| 8 | Unconditional `oneccl_bindings_for_pytorch` import | critical | [08](08-conditional-ccl-import.md) |
| 9 | Teardown segfault overrode the exit status | high | [09](09-teardown-exit-status.md) |

Defects 7, 8, and 9 share a property worth noting: none was findable by
reading the source. Each is a mismatch between correct-looking code and the
machine it runs on, and each surfaced only on an actual execution attempt.
Defect 8 alone made the benchmark unable to start on the current Aurora
module stack.

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

Local result: **136 passed**. Under the Aurora `frameworks/2025.3.1` stack
(torch 2.10, Python 3.12): **134 passed, 2 skipped** — the two skips are
`alltoall`, which that build's gloo backend refuses at call time. Note that
`dist.all_to_all` exists as a symbol on both builds, so the capability must be
probed by result rather than by `hasattr`. The skip triggers only on the
backend's own "does not support" message; any other error still fails, and a
genuine no-op regression raises no error at all, so
`test_noop_collective_is_detected` cannot be skipped into a false pass.

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
repository tree under test stays unmodified. Both the sabotage directory and
the repository root must be on `PYTHONPATH`, and it must be propagated to the
ranks with `mpiexec -genv`; otherwise every rank fails `import dl_comm` and
the run proves nothing while still exiting non-zero.

### Measured on 2 nodes, 24 XPU ranks (job 8824234)

Within-node allreduce, 4 MiB buffer, 20 iterations, `CCL_OP_SYNC=0`:

```
group              n     bytes     t_min     t_med    t_mean    t_std     t_p99   algbw_med   busbw_med
(Within-Group-0)  12   4194304  0.000393  0.000419  0.000419  0.000013  0.000441  1.000e+10   1.834e+10
(Within-Group-1)  12   4194304  0.000386  0.000421  0.000419  0.000013  0.000451  9.961e+09   1.826e+10
busbw factor for allreduce at n=12: 1.833333
```

This confirms on hardware that algbw and busbw are reported separately, that
the busbw factor is the standard `2(n-1)/n` (1.8333 at n=12), that iteration 0
is excluded from the summary, and that the min/median/mean/stddev/p99
statistics are populated. With `CCL_OP_SYNC=0` the run depends on the
benchmark's own synchronization, exercising fix 5.

### Validation-harness failures worth recording

Four successive jobs failed before any of the above was measurable, each for a
different environmental reason. They are listed because the harness reported
misleading status in three of the four cases:

1. **8824212** — `module load frameworks` aborted under `set -u` (Aurora's
   lmod init references unbound variables), leaving no Python. The job
   reported `Exit_status=0` with a one-second walltime while running nothing.
2. **8824221** — defect 7. The harness `RUN_C_EXIT=143` looked like the
   expected non-zero sabotage result, but the process had died on a
   `KeyError` before any collective ran. Only the evidence check distinguished
   the two.
3. **8824229** — defect 8. All ranks died at import.
4. **8824234** — defect 9. Measurement succeeded, teardown segfaulted, exit
   143.

The recurring lesson is that an exit code is not evidence. The harness now
requires positive proof that each phase did its work: run B must emit both
benchmark output and a `[CORRECTNESS]` block, and run C must show the sabotage
hook firing. Without those gates, cases 1, 2 and 4 would each have been
reported as a pass.
