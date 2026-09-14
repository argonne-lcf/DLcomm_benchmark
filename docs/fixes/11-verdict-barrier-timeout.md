# Fix 11 — Silent deadlock at the correctness verdict

**Status:** fixed; found by job 8824276

**Severity:** high — a hang consumes the full walltime and reports nothing
**File:** `dl_comm/dl_comm_main.py`
**Found by:** Aurora validation job 8824276

## Symptom

Run B produced complete bandwidth output, logged `[MPI] Job complete` and
`[EXIT] All Done.`, then stopped. The job sat in `job_state=R` for 38 minutes
against a 3-minute workload until it was killed manually. No `[CORRECTNESS]`
block was ever printed and no `results.json` was written.

Output frozen at 16:21:26; walltime at kill 00:38:20.

## Cause

The verdict block added by fix 4 reduces per-rank tallies across
`MPI_COMM_WORLD`:

```python
total_failures = MPI.COMM_WORLD.allreduce(local_verify["failures"], op=MPI.SUM)
```

`allreduce` is collective: it returns only when **every** rank in the
communicator calls it. The task loop above contains seven `continue`
statements. Any rank that takes one — a missing task key, a validation
failure, a rank-count shortfall — leaves the loop at a different point than
its peers. If the set of ranks reaching the verdict differs from the set that
entered, the reduce blocks forever.

This is the same class as the segfault in fix 9: a failure in the reporting
path is more damaging than a failure in the measured path, because it destroys
the evidence rather than producing a wrong number.

## Fix

A collective call cannot report who failed to arrive — by the time it blocks,
the information is gone. The fix replaces the blind reduce with a non-blocking
barrier and a bounded wait:

```python
_req = MPI.COMM_WORLD.Ibarrier()
_deadline = time.time() + _VERDICT_TIMEOUT_S
while not _req.Test():
    if time.time() > _deadline:
        sys.stderr.write(f"[CORRECTNESS] rank {mpi_rank} timed out ...")
        MPI.COMM_WORLD.Abort(3)
    time.sleep(0.05)
```

`Ibarrier` returns immediately and is polled, so a rank that waits too long can
still execute code — it prints which rank it is and how many were expected,
then aborts the job with status 3 instead of hanging.

The timeout is `DLCOMM_VERDICT_TIMEOUT` seconds, default 120.

## Why abort rather than skip the verdict

Skipping the reduce on timeout would let the job exit 0 with no correctness
result, which is the false pass this work exists to remove. A job that cannot
determine whether verification passed must not report success.

## Verification

Local suite: 142 passed. The timeout path is exercised on hardware; a run in
which all ranks arrive normally passes through the barrier in well under a
second and the added cost is not measurable against the benchmark's own
iteration timings.
