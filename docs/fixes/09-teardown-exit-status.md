# Fix 9 — Teardown segfault overrode the exit status

**Severity:** high — a signal death can mask a correctness failure
**File:** `dl_comm/dl_comm_main.py`
**Found by:** Aurora validation job 8824234

## Defect

A clean 24-rank run on two Aurora nodes completed all measurement work and
printed its final banner, then died:

```
[2026-09-13 15:12:01,569][DL_COMM][INFO] - [EXIT] All Done.
x4217c5s2b0n0: rank 5 died from signal 11
x4217c5s2b0n0: rank 0 died from signal 15
RUN_B_EXIT=143   (expected 0)
```

Signal 11 is a segfault inside XCCL teardown, triggered by
`dist.destroy_process_group()` on subgroups created with
`use_local_synchronization=True`. The crash occurs during interpreter
finalization, where the oneCCL/XCCL destructors run.

The visible symptom is a successful run reporting exit 143. The dangerous
symptom is the general case: **a process killed by a signal reports the
signal, not its intended exit status.** A run that had detected a real
correctness failure would call `sys.exit(1)` from fix 4, then segfault during
teardown, and the launcher would report 143 instead. Downstream tooling that
distinguishes "failed verification" (1) from "crashed" (143) sees the wrong
answer, and any harness checking only for nonzero sees a correctness failure
and an unrelated crash as identical.

This is the same defect class as fix 4 — a failure that does not reliably
reach the caller — relocated into the shutdown path.

## Fix

Latch the verdict before teardown, make teardown non-fatal, and leave through
`os._exit` so interpreter finalization never runs:

```python
exit_code = 1 if total_failures else 0

if framework == "pytorch":
    try:
        dist.destroy_process_group()
    except Exception as exc:
        print(f"[dl_comm] destroy_process_group raised (ignored): {exc}", flush=True)
...
sys.stdout.flush()
sys.stderr.flush()
os._exit(exit_code)
```

`os._exit` bypasses the destructors that raise SIGSEGV. Both streams are
flushed first, because `os._exit` skips `atexit` handlers and buffered output.

## Verification

The exit-status logic is covered by `tests/test_failures_and_results.py`. The
segfault itself is hardware- and backend-specific and does not reproduce under
the CPU gloo suite; it is confirmed only by the Aurora job output quoted above.

## Notes

The underlying XCCL teardown crash is not fixed here — it is upstream, in the
backend's subgroup destructors. This change ensures the crash cannot corrupt
the benchmark's reported result.
