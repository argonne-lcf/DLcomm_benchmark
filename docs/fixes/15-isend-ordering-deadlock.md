# Fix 15 — `sendrecv_async` deadlocks at `isend` on XCCL

**Severity:** critical — hung the job; the collective never completed one call
**File:** `dl_comm/comm/collectives.py`
**Found by:** Aurora job 8824532, the first run of these collectives on XPUs

## Symptom

`alltoallv` and `sendrecv` completed and reported bandwidth. The third task,
`sendrecv_async`, never produced a single timed iteration. The watchdog fired
on 22 of 24 ranks with identical stacks:

```
Timeout (0:05:00)!
  File ".../torch/distributed/distributed_c10d.py", line 2491 in isend
  File ".../dl_comm/comm/collectives.py", line 409 in _send_recv_async
  File ".../dl_comm/dl_comm_main.py", line 618 in main
```

Line 618 is the **warmup** loop, so the hang occurred on the very first call —
before any timing or verification ran.

22 rather than 24 is expected: the two ranks that are the odd member of an
odd-sized group return early and never enter the exchange.

## Cause

The original implementation posted both requests send-first:

```python
reqs = [dist.isend(tensor, dst=partner, group=group),
        dist.irecv(recv, src=partner, group=group)]
```

Every rank in the pair executes the same line, so both call `isend` before
either calls `irecv`. XCCL matches a point-to-point pair at enqueue time; with
no posted receive to match, both peers block inside `isend` and neither ever
reaches its `irecv`. The deadlock is symmetric and total.

The blocking `sendrecv` in the same file avoids this by ordering on rank parity
— even ranks send first, odd ranks receive first. The async path had no
equivalent ordering.

## Fix

Post the receive before the send:

```python
reqs = [dist.irecv(recv, src=partner, group=group),
        dist.isend(tensor, dst=partner, group=group)]
for r in reqs:
    r.wait()
```

Both operations are still non-blocking and both are in flight before either
`wait()`, so this remains the bidirectional-bandwidth case and does not
degenerate into a re-spelling of the blocking version.

## Why the test suite did not catch this

The gloo harness passes with **either** ordering — gloo buffers the send rather
than requiring a matched receive at enqueue time. The 149-test suite therefore
reported `sendrecv_async` as working, on both the broken and the fixed code.

This is a backend-behaviour difference that no CPU-side test reproduces. It is
the concrete case for the rule that gloo results are a development convenience
and only the XCCL run on hardware is evidence.

## Status

Fixed and resubmitted. Not yet confirmed on hardware — the claim that
`sendrecv_async` works on XPUs remains unproven until a job completes with a
correctness verdict.
