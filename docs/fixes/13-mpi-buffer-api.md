# Fix 13 — Pickle-based MPI reduce deadlocks after XCCL initialisation

**Status:** fixed; found by job 8824457

**Severity:** critical — the correctness verdict never executed on hardware
**File:** `dl_comm/dl_comm_main.py`
**Found by:** Aurora validation job 8824457, located by the fix 12 watchdog

## Symptom

Three consecutive validation jobs (8824276, 8824405, 8824457) printed complete
bandwidth tables, logged `[EXIT] All Done.`, and then hung until killed. No
`[CORRECTNESS]` verdict was ever produced and no `results.json` was written.

## How it was located

The first two hangs were diagnosed by inspection and the conclusion was wrong.
The hypothesis on record — that ranks were diverging at the `continue`
statements in the task loop and so a different set of ranks reached the
collective — was disproved by this run.

The watchdog added in fix 12 printed the stack of every rank:

```
Timeout (0:04:00)!
Thread 0x0000146b79dc2280 (most recent call first):
  File ".../dl_comm/dl_comm_main.py", line 791 in main
```

Counting the dumps gives the decisive evidence:

```
$ grep -cE "^Timeout \(.*\)!" run_b.out          # ranks that hung
22
$ grep -oE 'dl_comm_main.py", line [0-9]+ in main' run_b.out | sort | uniq -c
     23 dl_comm_main.py", line 791 in main
```

Every hung rank was at the *same* line, and the fix 11 barrier immediately
above it had been passed by all of them (`grep -c "timed out after"` → 0). The
ranks were not diverging. They all arrived, and the collective itself never
completed.

## Cause

Line 791 was the lowercase, pickle-based mpi4py call:

```python
total_failures = MPI.COMM_WORLD.allreduce(local_verify["failures"], op=MPI.SUM)
```

mpi4py exposes two APIs. The lowercase methods (`allreduce`, `gather`, `bcast`)
accept arbitrary Python objects: they pickle the payload and negotiate its size
at runtime with additional probe and receive traffic. The uppercase methods
(`Allreduce`, `Gather`, `Bcast`) move fixed-size buffers with no negotiation.

Once the XCCL backend is initialised on the same ranks, the dynamic negotiation
path stops making progress and the call blocks indefinitely.

Note what this does *not* say. Pickle-based calls are not uniformly broken here:
`MPI.COMM_WORLD.gather` at `analysis/bandwidth.py:294` runs after XCCL
initialisation and completes — its output is the `[BANDWIDTH] SUMMARY` table
printed immediately before the hang. The failure depends on the position in the
teardown sequence, not merely on pickling. The correct engineering response is
to avoid the object API on the shutdown path rather than to explain exactly
which interaction stalls.

## Fix

The three tallies are packed into one fixed-size buffer and reduced with the
uppercase call:

```python
_counts = np.array([local_verify["failures"],
                    local_verify["checks"],
                    local_verify["skipped"]], dtype=np.int64)
_totals = np.zeros(3, dtype=np.int64)
MPI.COMM_WORLD.Allreduce(_counts, _totals, op=MPI.SUM)
```

This also reduces three collectives to one.

The `gather` of per-rank detail strings is removed rather than converted.
Variable-length strings have no fixed-size representation, so gathering them
would reintroduce the object path. Every rank already logs its own failures as
they occur — run C's evidence is 40 such lines — so the aggregate counts plus
per-rank logs carry the same information.

## Verification

Run C of job 8824457 exercised the sabotage hook for the first time and the
verification caught it on real XPUs:

```
[SABOTAGE] allreduce replaced with a no-op
[CORRECTNESS][Within-Group-0] allreduce iteration 0 [FAILED] - op=max, group_size=12
...
$ grep -c FAILED run_c.out
40
```

That is the load-bearing result for the whole effort: a deliberately broken
collective is detected rather than reported clean. It was produced by the
per-rank checks, which run before the reduce and were unaffected by this defect.
