# Instrumented NCCL build — transport fix + in-collective timers

Scripts that build an instrumented NCCL on Tara North (GH200, Slingshot 11)
and measure *where time goes inside a collective*, rather than inferring it
from `NCCL_DEBUG` text.

Measured result: switching NCCL off its TCP fallback onto CXI gave
**18.4× lower latency / 19.4× higher bus bandwidth** on a 256 MiB 8-rank
allreduce, verified in a single A/B job on the same two nodes.

## Why

Stock NCCL on this machine silently selects `NET/Socket` — the Slingshot
rails are visible but driven as kernel TCP with GPU Direct RDMA disabled.
NCCL has no native libfabric/CXI support; it needs the `aws-ofi-nccl` plugin
(`libnccl-net.so`). Without that file, every multi-node NCCL number is a
TCP-over-Slingshot number, with no error and exit 0.

## Files

| File | Purpose |
|---|---|
| `build_nccl_instr.sh` | PBS: build NCCL 2.27.5 `TRACE=1` (sm_90) + profiler plugin |
| `build_ofi_only.sh` | PBS: build aws-ofi-nccl v1.21.1 against CXI libfabric |
| `nccl_ab.sh` | PBS: stock vs instrumented, **one job, same nodes** |
| `nccl_prof.sh` | PBS: capture per-event timers to JSON traces |
| `parse_ab.py` | Extract transport + timings from interleaved A/B logs |
| `nccl_trace_stats.py` | Aggregate traces into a GPU/PROXY/NET phase split |

## Run order

```bash
qsub build_nccl_instr.sh    # 1 node
qsub build_ofi_only.sh      # 1 node
qsub nccl_ab.sh             # 2 nodes — proves the transport switch
qsub nccl_prof.sh           # 2 nodes — produces the traces
python parse_ab.py <run>/stock.txt <run>/instr.txt
python nccl_trace_stats.py <run>/trace_268435456_*.json
```

## Reading the result

The primary evidence is the **transport line**, not the speedup:

```
stock:  NET/Plugin: Could not find: libnccl-net.so.   -> Using network Socket
instr:  NET/Plugin: Loaded net plugin Libfabric (v10) -> Using network Libfabric
```

A faster number still reporting `Using network Socket` means the plugin did
not load and the delta came from something else. Check this first.

`nccl_trace_stats.py` reports `unmatched_begin_events`. It must be **0**;
anything else means begin/end pairing failed and the durations are not
trustworthy.

## Pitfalls these scripts already handle

- **`NCCL_PROFILE_EVENT_MASK` defaults to 0.** Without it the profiler plugin
  loads, logs that it loaded, and records nothing. These scripts set `90`
  (Coll | ProxyOp | ProxyStep | KernelCh).
- **`mpiexec --env LD_LIBRARY_PATH` replaces the module-set value**, it does
  not extend it. Omitting the CUDA dirs kills every rank with
  `libcudart.so.12: cannot open shared object file` (exit 127 → 143), which
  reads like a fabric timeout. Each arm greps for library-resolution failure
  before reporting any timing.
- **Two CUDA runtimes are needed in one process**: `.so.13` for
  `all_reduce_perf`, `.so.12` for our CXI libfabric and the plugin.
- **aws-ofi-nccl ≥1.21 needs C++20.** Default `g++` here is 7.5.0; the build
  pins `g++-14`.
- **`--disable-dependency-tracking` is required.** Upstream guards the GDAKI
  (AWS EFA) sources with `if HAVE_GDAKI` but references them across
  directories under `subdir-objects`, so automake emits a `.deps` rule for a
  file it never generates when GDAKI is off — as it is on CXI. Reproduces on
  a pristine tree.
- **Source must reach the cluster by rsync**; Tara has no GitHub egress.
  Stripping `.git` breaks `autogen.sh` version detection — a `.release_version`
  file is the supported substitute.
- **stdout/stderr interleave**, truncating result rows. `parse_ab.py` matches
  rows structurally rather than by leading byte count.
