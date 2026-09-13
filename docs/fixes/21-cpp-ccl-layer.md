# Feature 21 — C++-level CCL benchmark

**Status:** written; first compile attempted in job 8824748. Not yet proven to
build or run.

## Why

DLcomm measured two layers: MPI (via OSU) and PyTorch. The gap between them
was unattributable — a slow `allreduce` in PyTorch could be the framework, the
CCL, or the fabric, and nothing in the benchmark could tell them apart.

`dl_comm/ccl/ccl_bench.cpp` adds the middle layer: the vendor CCL called
directly from C++, with no Python interpreter and no torch in the path.
Comparing it against the PyTorch layer isolates framework overhead; comparing
it against OSU isolates the CCL against MPI.

## Backends

Selected at compile time, exactly one:

| Flag | Library | Runs on Aurora |
|---|---|---|
| `-DDLCOMM_XCCL` | oneCCL | yes (default) |
| `-DDLCOMM_NCCL` | NCCL | no — no NVIDIA device |
| `-DDLCOMM_RCCL` | RCCL | no — no AMD device |

NCCL and RCCL are behind compile guards rather than omitted. Aurora cannot
execute them, so they are kept compilable and reviewable for a machine that
can; claiming them as supported here would be false.

## Operations

Collectives: `allreduce`, `allgather`, `alltoall`, `broadcast`, `reduce`,
`reduce_scatter`, `barrier`.
Point-to-point: `sendrecv` between ranks 0 and 1.

The oneCCL header set was checked before writing the code — all ten primitives
including `send`/`recv` are present in the API, so p2p is measurable at this
layer and not only at the torch level.

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

Without this the C++ numbers would not be comparable with the Python layer's,
which is the entire purpose of adding the layer.

Output is one `LAYER=cpp_ccl` line per (collective, size), parseable by
`dl_comm.osu.compare`.

## Build

```
mpicxx -fsycl -O2 -DDLCOMM_XCCL ccl_bench.cpp -o ccl_bench \
    -I$CCL_ROOT/include -L$CCL_ROOT/lib -lccl
```

## Status and risk

The code has never been compiled. The oneCCL C++ API for communicator
construction (`create_main_kvs`, `create_communicator`, stream binding) is
easy to get wrong, and errors there appear only at compile or first run. Job
8824748 is the first test; results will be recorded here rather than assumed.
