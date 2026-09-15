# Fix 21 — C++ oneCCL benchmark layer

**Status:** builds and runs at 12 and 24 ranks. Collective figures below are
measured; `reduce_scatter` is **retracted pending a re-run** (see "Retracted
numbers").
**File:** `dl_comm/ccl/ccl_bench.cpp`
**Jobs:** 8824748 (first compile), 8825050, 8825296 (measured)

## Why the layer exists

DLcomm measured two layers: MPI through OSU, and PyTorch. A slow `allreduce` in
PyTorch could be the framework, the communication library, or the fabric, and
nothing in the benchmark could separate them.

`ccl_bench.cpp` calls the vendor CCL directly from C++, with no Python
interpreter and no torch in the path. Against the PyTorch layer it isolates
framework overhead; against OSU it isolates the CCL from MPI.

## Backends

Selected at compile time, exactly one:

| Flag | Library | Runs on Aurora |
|---|---|---|
| `-DDLCOMM_XCCL` | oneCCL | yes (default) |
| `-DDLCOMM_NCCL` | NCCL | no — no NVIDIA device |
| `-DDLCOMM_RCCL` | RCCL | no — no AMD device |

NCCL and RCCL sit behind compile guards rather than being omitted, so they stay
compilable and reviewable for a machine that can run them.

## Operations

Collectives: `allreduce`, `allgather`, `alltoall`, `broadcast`, `reduce`,
`reduce_scatter`, `barrier`. Point-to-point: `sendrecv` between ranks 0 and 1.

## Build

```
mpicxx -fsycl -O2 -DDLCOMM_XCCL ccl_bench.cpp -o ccl_bench \
    -I$CCL_ROOT/include -L$CCL_ROOT/lib -lccl
```

## Measured bus bandwidth, 4 MiB (GB/s)

Job 8825296, Aurora, `frameworks/2025.3.1`, oneCCL 26.26.0.

| op | 12 ranks | 24 ranks |
|---|---|---|
| allgather | 80.74 | 44.58 |
| sendrecv | 25.10 | 26.20 |
| allreduce | 22.41 | 15.37 |
| reduce | 16.18 | 9.38 |
| alltoall | 11.67 | 2.75 |
| broadcast | 10.62 | 1.72 |
| reduce_scatter | retracted | retracted |

`alltoall` falls 4.2× across the node boundary while `allreduce` falls 1.5×.
This is the collapse recorded in `docs/findings/02`, which four layers now show
independently.

## Retracted numbers

**`reduce_scatter` 220.71 GB/s at 12 ranks and 152.74 GB/s at 24 ranks are
withdrawn. Do not cite them.**

The value is inflated by the rank count. `traffic_bytes()` expanded the
`reduce_scatter` numerator by `world`, but the call site passes `count/world`
(the per-rank output count), so the reported buffer size is already the total
input volume. The raw record shows the error directly:

```
OP=allgather      BYTES=4194304 MOVED_BYTES=50331648 T_MED=0.000571413 BUSBW=8.07e+10
OP=reduce_scatter BYTES=4194304 MOVED_BYTES=50331648 T_MED=0.000209043 BUSBW=2.21e+11
```

`MOVED_BYTES` is identical for both, which is correct — they are duals and move
the same bytes — but that is exactly why the result is impossible.
`reduce_scatter` cannot report 2.73× the bandwidth of `allgather` for the same
traffic.

Commit `9de202e` corrected `traffic_bytes()` to expand `allgather` only. The
correct figures require a re-run; they are not estimated here.

## Why the wrong numbers were published anyway

Two failures, both worth recording because each defeats the other's safeguard.

**The fix never reached the machine.** Commit `9de202e` was made locally and
the deployed tree kept the pre-fix source:

```
local  f02b35e11aec54f14c96934b753de322   if (op == "allgather")
remote 38f4ee99713a007d1e2f0e6dbf95c62b   if (op == "allgather" || op == "reduce_scatter")
```

Every C++ CCL run after that commit used a binary compiled from pre-fix source.
A commit is not a deployment, and only an md5 comparison of the file the build
actually compiles establishes which code ran.

**The guard written for this defect could not fire.** The same commit added
`test_no_collective_exceeds_hardware_ceiling`, intended to catch exactly this
class of error. It failed twice over:

1. It runs in pytest against synthetic values. It never sees job output, so no
   value produced by the deployed binary can reach it.
2. Its threshold is `CEILING_BPS = 1.5e12`. The wrong value was 220.71 GB/s —
   roughly 7× below the trip point. Even had the guard seen the number, it
   would have passed.

A threshold that never fires is untested. The check that would have caught this
is a duality assertion — `reduce_scatter` must not exceed `allgather` for the
same message size and rank count, since the two move identical bytes — and it
needs to run against emitted records, not synthetic ones.

## Comparability

Bus bandwidth uses the same per-collective factors as
`dl_comm/analysis/bandwidth.py`:

```
allreduce                      2(n-1)/n
allgather, alltoall,
alltoallv, reduce_scatter        (n-1)/n
broadcast, reduce, barrier,
sendrecv                         1
```

Output is one `LAYER=cpp_ccl` line per (collective, size), parsed by
`dl_comm.analysis.parse_layers`. Buffers are device-resident, so these figures
are same-path with the torch layers and not with OSU (see the README section on
cross-layer comparison).
