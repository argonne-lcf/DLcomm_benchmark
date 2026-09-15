# DLcomm benchmark test suite

The suite is split by what it needs to run. Everything here runs on a login
node or in CI: no GPU, no job allocation, no MPI launcher.

## Layout

| File | Kind | Covers |
|---|---|---|
| `test_payload.py` | unit | rank-dependent payload construction (fix 1) |
| `test_correctness_detects_breakage.py` | integration, gloo | verification detects broken collectives (fix 1) |
| `test_bandwidth.py` | unit | group size, algbw/busbw, iteration statistics (fixes 2, 5) |
| `test_topology_validation.py` | unit | orphaned-rank detection (fix 3) |
| `test_failures_and_results.py` | unit | failure accounting, results.json/csv (fix 4) |
| `test_timer_sync.py` | unit | device synchronization and barrier placement (fix 5) |
| `gloo_harness.py` | helper | spawns real `torch.distributed` gloo processes |
| `conftest.py` | helper | fixtures |

## Running

```
python -m pytest tests -q                 # everything
python -m pytest tests -q -m "not gloo"   # unit tests only, ~2 s
python -m pytest tests/test_correctness_detects_breakage.py -q
```

## Why the gloo tests matter

`test_correctness_detects_breakage.py` runs each collective twice against real
spawned processes: once working, once with the collective removed entirely. The
second run asserts that verification **fails**. A verification routine that
passes both runs is vacuous, which is exactly the defect these tests were
written for: with the original all-ones payload, 12 of 15 collective/operation
combinations reported a no-op collective as correct.

The `broken_noop` assertions are therefore the load-bearing part of the suite.
If one of them ever starts passing with zero failures, verification has
regressed to being decorative.

## Dependencies

`torch` (CPU build is sufficient) and `pytest`. `mpi4py` with a working MPI
runtime is required because `dl_comm.timer` imports it at module scope. On a
machine without a system MPI, `pip install mpi4py openmpi` provides both.
