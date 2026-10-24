# 18 — NIXL transfer layer and torchcomms on NVIDIA

Two things that previously did not run outside Aurora:

* **NIXL transfer** (`h2d` / `d2h` / `d2d`) — measures the three directions
  through NIXL's registered-memory path instead of the framework's own
  `copy_`, which is what a disaggregated serving job actually uses.
* **torchcomms (layer 5) on NVIDIA** — verifies DLcomm's torchcomms backend
  resolves to a transport that exists on the local machine and that a
  collective moves real data.

Nothing here is specific to one site. Backends and transports are resolved
from what the local build registered, with an optional explicit override.

## Files

| File | Purpose |
|---|---|
| `run_nixl_transfer.py` | NIXL h2d/d2h/d2d under MPI, one rank per GPU |
| `run_torchcomms_check.py` | torchcomms transport resolution + verified allreduce |
| `test_nixl_transfer.py` | Unit tests for the reporting arithmetic (no GPU, no NIXL) |
| `jobscript_nixl_transfer.sh` | PBS submission for the NIXL layer alone |

## Quick start

```bash
# unit tests — no GPU and no NIXL needed
python examples/18_nixl_transfer/test_nixl_transfer.py

# single rank
python examples/18_nixl_transfer/run_nixl_transfer.py \
    --bytes 67108864 --iterations 10 --no-barrier

# under PBS
qsub -A <account> -l select=4 examples/18_nixl_transfer/jobscript_nixl_transfer.sh
```

## Optional backend arguments

The NIXL backend is **not** hardcoded. With no flags it is resolved from the
plugins the local build registered, preferring `UCX`, then `GDS`, then
`POSIX`. Override when you need a specific path:

```bash
# force a backend; fails loudly if it is not registered
--backend UCX

# change the auto-resolution order
--backend-preference GDS,UCX,POSIX

# backend tuning applied before the agent is constructed
--backend-env UCX_IB_GDA_RETAIN_INACTIVE_CTX=yes
```

`--backend-env` exists because UCX reads its configuration **once**, when the
agent is created; exporting `UCX_*` after that point is silently ignored. The
variables are restored after the agent is built.

Similarly `run_torchcomms_check.py --transport nccl` overrides the resolved
transport; by default it asks DLcomm's own `resolve_transport("gpu")`.

## Reading the output

```
NIXL_MAP    <host> rank <r> dev <d>
NIXL_RESULT <direction> <backend> <bytes> <best_s> <gbytes_per_s>
```

`gbytes_per_s` is **per rank**, not job-wide. To get aggregate throughput,
sum it across ranks; to get per-node, divide that by the node count. The
jobscript checks `NIXL_MAP` host/device pairs are distinct — ranks sharing a
GPU would otherwise inflate the per-device numbers silently.

## Measured on Tara (GH200, 4 GPUs/node, NIXL 1.5.0 + UCX)

64 MiB buffers, best of 10 iterations, median across ranks:

| Direction | per-rank GB/s | 1 node (4 ranks) aggregate | 4 nodes (16 ranks) aggregate |
|---|---|---|---|
| h2d | 0.41–0.42 | 1.73 GB/s | 6.82 GB/s |
| d2h | 0.31–0.32 | 1.33 GB/s | 5.29 GB/s |
| d2d | 0.20 | 0.85 GB/s | 3.37 GB/s |

Per-node throughput is the same at both scales (1.73 vs 1.71 GB/s h2d), which
is the expected result for a node-local transfer and the sanity check that the
per-rank arithmetic is right.

torchcomms allreduce (1 Mi floats, transport `nccl`, all ranks verified):
63.7–65.2 µs within one node, 1.88–2.12 ms across four.

### These NIXL numbers are a local-loopback floor, not link bandwidth

All three directions land within a few percent of each other (~6.6 GB/s
single-rank at 16 MiB), while native `torch` copies on the same node reach
**365 GB/s h2d** and **2032 GB/s d2d**. A d2d path that is no faster than h2d
is not using NVLink: NIXL's local UCX path stages through host memory. This is
a property of this NIXL/UCX build, not a configuration error — restricting
`UCX_TLS` to `cuda_copy`/`cuda_ipc` makes the backend fail to initialise
entirely (`NIXL_ERR_BACKEND`), and forcing a primary CUDA context before agent
creation does not change the result.

Treat these as a **floor for NIXL's local path**, and compare against
`examples/15_transfer_h2d_d2h_d2d` for the framework copy path. The
interesting NIXL numbers are inter-node RDMA, which this example does not yet
exercise.

## Known upstream issue: NIXL 1.5.0 segfaults at exit

`run_nixl_transfer.py` ends with `os._exit(0)`. NIXL 1.5.0 faults when its
agent is destroyed during interpreter shutdown, reproducible with no torch and
no CUDA in the process:

```bash
python -c "from nixl._api import nixl_agent, nixl_agent_config; \
           a = nixl_agent('x', nixl_agent_config(backends=['UCX'])); del a"
# -> Segmentation fault (exit 139), after the script body completed
```

Results are printed and flushed before this point, so nothing is lost — but
the rank exits 139 and `mpiexec` fails the whole job with "died from signal
11" on an otherwise successful run. Remove the `os._exit` once upstream fixes
the teardown.

## Layer 4 (oneCCL/SYCL) is not runnable here

Not a DLcomm defect and not fixable in this repo: Tara has no SYCL compiler
(`icpx`, `dpcpp`, `sycl-ls` all absent), no oneCCL headers, and
`torch.xpu.is_available()` is `False`. Layer 4 needs an Intel GPU machine such
as Aurora.
