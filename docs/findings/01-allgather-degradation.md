# Finding — allgather bandwidth degrades with message size

**Status:** reproduced at two layers, in three independent jobs.
**Jobs:** 8824700 (OSU/MPI), 8824748 and 8824754 (C++ oneCCL).
**Scale:** 1 node, 12 ranks, 1 rank per tile, Intel Data Center GPU Max 1550.

## Observation

Every collective measured gains bus bandwidth as the message grows, which is
the expected shape: fixed per-operation cost is amortised over more bytes.

`allgather` does the opposite.

C++ oneCCL layer, bus bandwidth in MB/s (job 8824748 / job 8824754):

| bytes | allgather | allreduce | alltoall | sendrecv |
|---|---|---|---|---|
| 1 MiB | 172 / 174 | 1007 / 976 | 559 / 562 | 8667 / 7669 |
| 2 MiB | 155 / 152 | 1499 / 1445 | 875 / 974 | 14716 / 14756 |
| 4 MiB | 132 / 124 | 1691 / 1838 | 1174 / 1247 | 22928 / 23023 |

allgather falls from ~172 to ~128 MB/s across a 4x size increase while
allreduce rises by 1.7x over the same range.

At 4 MiB, allgather is roughly 180x slower than sendrecv on the same ranks,
the same devices and the same fabric.

## Corroboration

The MPI layer shows the same outlier independently. OSU 7.1 latency at 4 MiB,
12 ranks (job 8824700):

```
allgather   72108.32 us
alltoall    20319.48 us
allreduce    9166.51 us
bcast        1080.72 us
```

allgather is 7.9x slower than allreduce there, against a payload ratio that
does not justify it.

Two separate implementations -- MPICH's allgather and oneCCL's -- exhibit the
same degradation on the same hardware. The benchmark code is common to
neither, so the measurement is unlikely to be at fault.

## What is not yet known

The cause has not been established. Candidates, none of them confirmed:

- an algorithm switch (ring to Bruck, or similar) at a size threshold;
- per-tile buffer growth: allgather output scales with rank count, so at 12
  ranks a 4 MiB input produces a 48 MiB output per rank, which may exceed a
  cache or staging buffer;
- contention between the 12 tiles for a shared host or fabric resource.

Distinguishing these requires a rank-count sweep (does the knee move with
world size?) and a finer size sweep to locate the threshold. Neither has been
run.

## Why this is recorded now

The number is reproducible and cross-validated, and it is the largest
performance anomaly the benchmark has surfaced. The explanation is not, so it
is stated as an open question rather than attributed to a cause.
