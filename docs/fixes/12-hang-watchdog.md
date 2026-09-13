# Fix 12 — Hang watchdog

**Severity:** high — a hang destroys the evidence a benchmark run exists to produce
**File:** `dl_comm/dl_comm_main.py`, `validate_on_aurora.sh`
**Found by:** Aurora validation jobs 8824276 and 8824405

## Problem

Two consecutive validation jobs hung after printing `[EXIT] All Done.`, each
consuming its full walltime allocation and producing no correctness result. A
hang is worse than a wrong number: a wrong number can be read and disputed,
whereas a hang leaves nothing to inspect.

Diagnosing the first one took an hour and several wrong hypotheses. The compute
nodes are air-gapped and carry neither `gdb` nor `py-spy`, so there was no way
to ask a stuck rank where it was:

```
$ gdb -p 208485
timeout: failed to run command 'gdb': No such file or directory

$ pip install --user py-spy
Failed to establish a new connection: [Errno 101] Network is unreachable
```

The only evidence obtainable was `/proc/<pid>/wchan` and per-thread state, which
narrowed the location but never named a line of Python.

## Fix

The program now diagnoses itself. `faulthandler.dump_traceback_later` runs on a
dedicated thread and, if the timer is not cancelled, prints the Python stack of
every thread on every rank and exits:

```python
_watchdog_s = float(os.environ.get("DLCOMM_WATCHDOG", "600"))
if _watchdog_s > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(_watchdog_s, exit=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
```

`faulthandler.enable()` also installs handlers for `SIGSEGV`, `SIGABRT` and
`SIGBUS`, so the teardown segfault of fix 9 will now print a stack rather than
only a signal number. `SIGUSR1` allows dumping a stack on demand from a running
job without killing it.

Default 600 s; the validation harness sets `DLCOMM_WATCHDOG=240` so a hang
surfaces in four minutes instead of at the walltime limit.

## Verification

An unfired threshold is untested, so the watchdog was made to fire against a
deliberate deadlock:

```
$ DLCOMM_WATCHDOG=5 python wd_test.py
entering hang
Timeout (0:00:05)!
Thread 0x0000f44e26805020 (most recent call first):
  File "/tmp/wd_test.py", line 10 in deadlocked_function
  File "/tmp/wd_test.py", line 13 in <module>
EXIT=1
```

It names the hung function, the exact line, and exits non-zero.

## Pitfall: the detection pattern did not match

The harness clause that surfaces the dump was first written as:

```bash
grep -q "Timeout (.*) !" "$RESULTS/run_b.out"
```

Tested against real faulthandler output, it matched nothing — the emitted text
is `Timeout (0:00:05)!` with no space before the exclamation mark. The clause
would have stayed silent on every hang while appearing to be a working check.
Corrected to:

```bash
grep -qE "^Timeout \(.*\)!" "$RESULTS/run_b.out"
```

and confirmed against captured output. This is the same failure mode the
project's own rule warns about: a check that has never been observed to fire is
not known to work.
