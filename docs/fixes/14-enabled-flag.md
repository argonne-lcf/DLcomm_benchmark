# Fix 14 — `results.json` reported `"enabled": false` on verified runs

**Status:** fixed; verified by job 8824490

**Severity:** medium — the structured record contradicted the run, and a
safety check could never fire
**File:** `dl_comm/dl_comm_main.py`
**Found by:** inspecting the `results.json` produced by Aurora job 8824490

## Symptom

The validating run wrote a record whose two fields disagree:

```json
"correctness": {
  "enabled": false,
  "total_checks": 480,
  "total_failures": 0,
  "passed": true
}
```

480 checks were executed and the flag describing whether verification was
enabled says it was not. The configuration driving the run contained
`verify_correctness: on`.

## Cause

The verdict block introduced by fix 4 read the flag from the top-level config:

```python
"enabled": bool(getattr(cfg, "verify_correctness", False)),
```

`verify_correctness` is not a top-level key. It is a per-mode setting read
inside the task loop as `mode_cfg.verify_correctness` (line 366). `getattr` on
the wrong object with a default silently returned `False` on every run,
including runs that verified.

`getattr(obj, name, default)` cannot distinguish "absent" from "present and
false", so the mistake produced a plausible value instead of an error.

## Second effect

The same flag gates the guard that catches a run which requested verification
but performed none:

```python
elif correctness_summary["enabled"] and total_checks == 0:
```

With `enabled` permanently `False`, this branch was unreachable. A configuration
that silently verified nothing would not have been reported — the precise
failure mode the guard exists to catch, and the one that motivated this work.

## Fix

Whether verification ran is a property of the run, not of a config object read
afterwards. A flag is set inside the task loop where the per-mode value is in
scope:

```python
enable_correctness = mode_cfg.verify_correctness
if enable_correctness:
    correctness_was_enabled = True
```

and reduced across ranks alongside the tallies, in the same buffer:

```python
_counts = np.array([failures, checks, skipped,
                    1 if correctness_was_enabled else 0], dtype=np.int64)
MPI.COMM_WORLD.Allreduce(_counts, _totals, op=MPI.SUM)
any_enabled = bool(_totals[3] > 0)
```

Reducing rather than reading rank 0's local value also covers a rank that sits
outside every communication group and therefore runs no tasks.

## Note

This defect was in code added by this hardening effort, not in the original
benchmark, and it disabled one of the checks this effort introduced. It was
found by reading the artefact the run produced rather than the run's console
output — the console reported `checks=480 failures=0`, which looked correct.
