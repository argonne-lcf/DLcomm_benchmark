# Fix 7 — `RUN_LOG_DIR` KeyError outside the bundled jobscripts

**Status:** fixed; found by job 8824221

**Severity:** medium — the benchmark was unrunnable except via its own jobscripts
**File:** `dl_comm/dl_comm_main.py`
**Found by:** Aurora validation job 8824221, not by code review

## Defect

Rank 0 read the log directory with a bare subscript:

```python
log_dir = os.environ["RUN_LOG_DIR"]
```

`RUN_LOG_DIR` is exported only by the bundled `examples/*/jobscript_*.sh`. Any
other launch path — a plain `mpiexec`, a CI harness, an interactive debug
session — died with an unhandled `KeyError` before a single collective ran:

```
File "dl_comm/dl_comm_main.py", line 94, in main
  log_dir = os.environ["RUN_LOG_DIR"]
KeyError: 'RUN_LOG_DIR'
```

All 24 ranks aborted at startup.

## Fix

Accept `RUN_LOG_DIR` or `DL_COMM_LOG_DIR`, and otherwise fall back to a
timestamped directory under the current working directory, announcing the
choice rather than failing:

```python
log_dir = os.environ.get("RUN_LOG_DIR") or os.environ.get("DL_COMM_LOG_DIR")
if not log_dir:
    log_dir = os.path.join(os.getcwd(), "logs",
                           f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    print(f"[dl_comm] RUN_LOG_DIR not set; logging to {log_dir}", flush=True)
os.makedirs(log_dir, exist_ok=True)
```

## Notes

The environment-variable sweep that followed this fix confirmed every other
`os.environ[...]` in the benchmark path is either a write or an already
guarded read. Unguarded reads remain in `tools/examples_dl_scaling/` and the
legacy `tests/*.py` standalone scripts, which are separate utilities outside
the benchmark entry point and were left unmodified.
