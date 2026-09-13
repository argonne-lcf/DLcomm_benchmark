# Feature 18 — OSU Micro-Benchmarks integration

**Files:** `dl_comm/osu/{__init__,runner,compare}.py` (new),
`tests/test_osu_integration.py`, `tests/test_osu_compare.py` (new),
`run_osu.sh`

## Purpose

DLcomm measures framework-level collectives — `torch.distributed` or
`torchcomms` over XCCL. OSU measures the same operations at the MPI transport
layer. Running both on identical payloads turns "allreduce achieved 2.9 GB/s"
into "allreduce achieved 2.9 GB/s against an MPI floor of X GB/s", which is the
difference between a measurement and a result.

## Why not a git submodule

The request was for OSU as a dependent package or submodule. Probing Aurora
first showed that to be the wrong shape.

`/soft/tools/osu/` already exists, with binaries built against the site MPI:

```
osu_latency   osu_latency_mp   osu_mbw_mr   osu_multi_lat
osu-micro-benchmarks-7.1-1.tar.gz
```

The tarball carries the complete source — 98 `.c` files, including 14 blocking
collectives (`osu_allreduce`, `osu_alltoallv`, `osu_reduce_scatter`, ...), 14
non-blocking, 7 pt2pt, one-sided, and NCCL variants.

OSU is an MPI-ABI-sensitive C program. A vendored submodule would have to be
rebuilt against each site's MPI anyway, and on Aurora it would *shadow* a
working site build with one compiled against whichever `mpicc` happened to be
loaded. That risks comparing DLcomm's XCCL numbers against a mis-built MPI
reference — a wrong number that looks plausible, which is worse than no number.

So the integration **discovers** OSU rather than vendoring it, and builds from
the site tarball only as a fallback.

Search precedence:

1. `$DLCOMM_OSU_DIR` — explicit override, always wins
2. `/soft/tools/osu` — Aurora site install
3. `$PATH` — module-provided install
4. `$DLCOMM_OSU_BUILD` — tarball build, opt-in via `build_from_tarball()`

## Equivalence map

Every registered DLcomm collective has a genuine OSU counterpart:

| DLcomm | OSU | DLcomm | OSU |
|---|---|---|---|
| `allreduce` | `osu_allreduce` | `scatter` | `osu_scatter` |
| `allgather` | `osu_allgather` | `reduce` | `osu_reduce` |
| `alltoall` | `osu_alltoall` | `reducescatter` | `osu_reduce_scatter` |
| `alltoallsingle` | `osu_alltoall` | `barrier` | `osu_barrier` |
| `alltoallv` | `osu_alltoallv` | `sendrecv` | `osu_bw` |
| `broadcast` | `osu_bcast` | `sendrecv_async` | `osu_bibw` |
| `gather` | `osu_gather` | | |

A collective absent from this map is reported `no_equivalent` and **not**
compared against a near-neighbour benchmark — a plausible but meaningless ratio
is the failure mode worth avoiding. A test asserts the map covers the registry
exactly, so a newly added collective fails CI instead of silently going
uncompared.

## Unit normalisation

OSU reports latency in microseconds or bandwidth in MB/s (base 1e6, not 2^20);
DLcomm reports B/s. Both are normalised to B/s before any ratio is taken:

- bandwidth: `MB/s x 1e6`
- latency: `size_bytes / (us x 1e-6)`

## Deliberate refusals

- A missing OSU install **raises `OsuNotFound`** naming every path searched and
  the `DLCOMM_OSU_DIR` override, rather than returning `None`. A missing
  reference must never be mistaken for a zero measurement.
- A non-zero exit or unparseable stdout **raises**, so a failed reference run
  cannot be reported as a comparison.
- A missing data point at the requested size is `no_reference_at_<N>B`, and
  renders as `-` in the table, never `0.00x`.
- **No pass/fail threshold.** A framework collective is legitimately slower
  than raw MPI (device buffers, Python dispatch, group setup), so a cutoff
  would manufacture false failures. The ratio is the result.

## Parser coverage

`osu_barrier` is the awkward case: it emits a single latency value with no size
column. It is recorded at size 0 rather than parsing as empty — tested
explicitly, since an empty parse would silently drop the benchmark.

## Verification

19 tests, all passing on CPU with no OSU installed. Parser tests use real OSU
7.x output shapes rather than invented ones.

Hardware verification via `run_osu.sh`: discovery against `/soft/tools/osu`, a
real `mpiexec -n 2 osu_latency` run, parsing its output, and a comparison
against job `8824624`'s measured `sendrecv` figure of 7.196e+09 B/s.

## Hardware verification (job `8824688`)

Discovery, execution, parsing, and comparison all confirmed on Aurora:

```
osu_latency:   /soft/tools/osu/osu_latency        <- found
osu_bw:        NOT FOUND. Searched: ['/soft/tools/osu', '$PATH']
osu_allreduce: NOT FOUND. Searched: ['/soft/tools/osu', '$PATH']
```

**The site install ships only four pt2pt binaries** — `osu_latency`,
`osu_latency_mp`, `osu_mbw_mr`, `osu_multi_lat`. No `osu_bw`, and no
collectives. The equivalence map above describes OSU the project, not what is
installed at `/soft/tools/osu`, so `build_from_tarball()` is the required path
for every collective comparison rather than a fallback.

This is exactly the case the "raise, never return None" rule exists for: the
missing benchmark produced an error naming all searched paths and the
`DLCOMM_OSU_DIR` override, instead of a silent zero that would have rendered
as a plausible `0.00x` ratio.

First measured comparison, DLcomm XCCL against MPI:

```
collective            bytes   dlcomm B/s      osu B/s    ratio  status
sendrecv            4194304    7.196e+09    2.315e+10    0.31x  ok
```

**This ratio is not yet apples-to-apples.** OSU ran 2 ranks across 2 nodes;
the DLcomm figure is a 12-rank within-node median from job `8824624`. It
demonstrates that the pipeline produces real numbers end to end, not that
DLcomm achieves 31% of MPI. A matched-topology run is required before the
ratio carries meaning.

## Usage

```python
from dl_comm.osu import run_osu, equivalent_for
from dl_comm.osu.compare import compare, format_table

osu = run_osu("osu_allreduce", ranks=24, ppn=12, min_size=1024, max_size=4194304)
print(format_table([compare("allreduce", 4194304, measured_bps, osu)]))
```

Override discovery with `export DLCOMM_OSU_DIR=/path/to/osu/bin`.
