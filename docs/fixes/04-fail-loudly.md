# Fix 4 — Correctness failures did not fail the job

**Status:** fixed; verified on hardware (job 8824490 run C exits non-zero)

**Severity:** high — a silent corruption at 8192 nodes was a log line nobody greps
**Files:** `dl_comm/verify/failures.py` (new), `dl_comm/analysis/results.py` (new), `dl_comm/analysis/correctness.py`, `dl_comm/dl_comm_main.py`
**Tests:** `tests/test_failures_and_results.py`

## The defect

A verification mismatch called `log.output(...)` and nothing else. The process
still exited 0. `sys.exit(1)` appeared in the codebase only for duplicate task
names and configuration validation — never for a correctness failure.

A job script, CI pipeline, or scaling sweep therefore had no way to detect that
a collective returned wrong data. At 1024–8192 nodes the log is large and
nobody reads it line by line.

Compounding this, results were emitted only as formatted log text, so
downstream plots were produced by scraping the log and no provenance travelled
with the numbers.

## The fix

**Failure accounting.** `dl_comm/verify/failures.py` maintains a process-local
tally of checks, failures, and skips. Every checker in `correctness.py` records
its outcome there instead of only printing.

The detail list is capped at `MAX_DETAILS = 50` entries while the *count* stays
exact — an 8192-rank failure must not produce an 8192-entry list in memory or
in the results file.

**Job-level verdict.** At the end of the run `dl_comm_main.py` reduces the
failure count across all ranks with `MPI.SUM` and exits non-zero if it is
positive. The reduction matters: a failure on one rank of 98304 must fail the
job, and only rank 0 writes the result files.

**Structured results.** `dl_comm/analysis/results.py` writes `results.json` and
`results.csv` alongside the log:

- the pass/fail verdict and failure count;
- per-group measurements with group size, buffer bytes, and the full timing
  distribution;
- provenance: a SHA256 of the resolved config, library versions, rank count,
  and timestamp.

## A bug found by the tests

`config_hash()` originally fell back to `repr(cfg)` when the config was not an
OmegaConf object. The default object `repr` embeds the instance's memory
address, so hashing the same configuration twice in one process produced
different digests and the provenance field was worthless.

`test_config_hash_is_stable_and_sensitive` asserts
`config_hash(A()) == config_hash(A())` and caught this. The fallback is now
`_stable_repr()`, a deterministic sorted traversal.

## Verification

```
$ python -m pytest tests/test_failures_and_results.py -q
```

`test_details_are_bounded` records 5000 failures and asserts the count stays
exact while the detail list stays capped.

On-hardware validation additionally runs a deliberately sabotaged collective
and asserts the job exits **non-zero** — see
`validation/` and `docs/INDEX.md`.
