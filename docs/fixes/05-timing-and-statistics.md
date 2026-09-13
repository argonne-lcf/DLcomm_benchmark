# Fix 5 — Timing measured enqueue, not completion

**Severity:** medium-high — correctness of the measurement depended on an environment variable set outside the tool
**Files:** `dl_comm/timer/timer.py`, `dl_comm/analysis/bandwidth.py`, `dl_comm/dl_comm_main.py`
**Tests:** `tests/test_timer_sync.py`, `tests/test_bandwidth.py`

## The defect

`grep -rn "synchronize|\.wait\(\)|async_op" dl_comm/` returned no hits outside
`dummy_mxm_compute`. The timed region was:

```python
with timer(label):
    collective(...)
```

On a GPU, launching a collective enqueues work on a device queue and returns.
Without a device synchronize before the stop timestamp, the measured interval
is the *enqueue* cost, not the time the collective took to complete.

The Aurora job scripts set `CCL_OP_SYNC=1`, which makes oneCCL operations
synchronous and hides the problem. That is the right result obtained for the
wrong reason: the validity of every measurement depended on an environment
variable set outside the tool, and a run without it silently produced
enqueue-latency numbers presented as collective bandwidth.

## Secondary defects

**Barrier inside the timed region.** `time_barrier()` was called *inside* the
`with timer(...)` block, so barrier cost was attributed to the collective.

**Iteration 0 kept as a sample.** The first iteration carries connection
establishment and lazy allocation. A representative observed run showed
iteration 0 at 0.000731 s against a settled ~0.00117 s — 37 % low, dragging the
mean. `examples/13_for_paper/` runs with `warmup_iterations: 0`, so this
outlier is in the published averages.

**Mean only.** No min, median, p99, or standard deviation was reported, so
run-to-run variance and stragglers were invisible.

## The fix

**Explicit synchronization.** `set_sync_device(device)` registers the device
whose queue must drain. The `timer` context manager calls `sync_device()`
immediately before the start timestamp and immediately before the stop
timestamp, so the interval brackets actual completion. It dispatches to
`torch.xpu.synchronize()` or `torch.cuda.synchronize()` by device type and is a
no-op on CPU.

`sync=False` opts a host-only region out.

The timer also records its sample in a `finally` block, so a collective that
raises still contributes its measurement rather than silently vanishing.

**Barrier moved out.** `time_barrier()` now runs outside the timed region.

**Statistics.** `summarize()` in `bandwidth.py` returns count, min, max, mean,
median, p99, and standard deviation, and reports iteration 0 separately as
`first` rather than folding it into the summary. Bandwidth is derived from the
median rather than the mean, so a single straggler does not move the headline
number.

## Verification

```
$ python -m pytest tests/test_timer_sync.py tests/test_bandwidth.py -q
```

`test_sync_is_invoked_at_both_ends` monkeypatches `sync_device` and asserts it
is called exactly twice per timed region — before the start stamp and before
the stop stamp. `test_timer_records_even_when_body_raises` asserts the sample
survives an exception. `test_summarize_excludes_first_iteration_outlier` uses
the observed 0.000731 / 0.00117 numbers.

On hardware, the validation job deliberately sets `CCL_OP_SYNC=0` so the tool
must synchronize itself; a run that only appeared correct because of the
environment variable would show it there.

## Note on the test

`dl_comm.timer.timer` as an attribute resolves to the re-exported *function*,
not the submodule, because `dl_comm/timer/__init__.py` does
`from .timer import timer`. Monkeypatching requires
`importlib.import_module("dl_comm.timer.timer")`. This shadowing cost a
debugging cycle and is worth knowing before adding tests here.
