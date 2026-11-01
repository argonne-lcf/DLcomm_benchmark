# Example 19 — per-layer option sweeps (CUDA / aarch64)

Example 16 asks *"does every layer work?"* — one setting per collective, one
number per layer. This example asks the harder question: **which of the
options each layer exposes actually change performance, and by how much?**

Every cell varies **one option at a time** against a fixed baseline. A full
cross product of the axes below is thousands of runs and, when something
regresses, tells you nothing about which axis caused it. One-at-a-time means a
slow cell names its own cause.

Scale is a parameter, not a second script: the same driver runs 1 node / 4 GPUs
and 2 nodes / 8 GPUs, so the two results are directly comparable.

## Axes

| Layer | Axis | Values |
|---|---|---|
| `fi` | provider inventory | CXI providers + domains |
| | per-rail pingpong | `cxi0..cxi3` (2 nodes only) |
| `osu` | collective | allreduce, allgather, alltoall, reduce, bcast, reduce_scatter, barrier, gather, scatter |
| | buffer | `device` (`-d cuda`), `host` |
| `nccl` | collective | all_reduce, all_gather, alltoall, broadcast, reduce_scatter, reduce, sendrecv, scatter, gather |
| | dtype | float, half, bfloat16 |
| | redop | sum, max, min, prod |
| | `NCCL_ALGO` | default, Ring, Tree |
| | `NCCL_PROTO` | default, Simple, LL, LL128 |
| `torch_dist` | collective | allreduce, allgather, reducescatter, alltoall, broadcast, reduce, barrier, send_recv |
| | dtype | float32, float16, bfloat16 |
| | async | sync, `async_op=True` |
| `nixl` | op × memory | READ/WRITE × VRAM/DRAM |
| | size | 4 KiB … 1 GiB |

Message sizes sweep 4 KiB → 256 MiB (`DLCOMM_QUICK=1` shortens this to 1 MiB
for a smoke test).

### Cell count

Counted by executing the driver against the offline stubs, not by hand:

| Layer | 1 node | 2 nodes |
|---|---|---|
| `fi` | 2 | 5 |
| `osu` | 18 | 18 |
| `nccl` | 19 | 19 |
| `torch_dist` | 11 | 11 |
| `nixl` | 4 | 4 |
| **total** | **54** | **57** |

`fi` grows from 2 to 5 because the four per-rail pingpongs need a second node.
Regenerate with `bash tools/count_cells.sh`.

### Measured outcome

Final runs on Tara North (GH200, Slingshot/CXI):

| Scale | Job | ok | failed | unsupported | unavailable |
|---|---|---|---|---|---|
| 1 node / 4 GPUs | 6956 | 49 | 0 | 4 | 1 |
| 2 nodes / 8 GPUs | 6957 | 55 | 0 | 2 | 0 |

Nothing is left in a `failed` state. The non-ok cells are two real platform
limits, each recorded with its reason rather than as a bug to chase:

- **same-node NIXL** (4 cells at 1 node) — a second LIBFABRIC agent cannot
  construct on one node; CXI has no loopback. Same root cause makes the 1-node
  `fi_pingpong` unavailable (`fi_domain()` returns -38).
- **CXI RMA WRITE** (2 cells at 2 nodes) — `postXferReq` returns
  `NIXL_ERR_BACKEND`, matching the -260 "Flags not supported" from the
  standalone reference. READ over the identical path is healthy
  (22.8 GB/s VRAM, 38.4 GB/s DRAM at 1 GiB, byte-exact).

A test asserts that this WRITE exemption stays narrow: the same backend error
on a **READ** must still be reported as `failed`.

## Contents

| File | Purpose |
|---|---|
| `sweep_layers_cuda.sh` | the driver — all layers, all axes, one CSV |
| `jobscript_sweep_1node.sh` | 1 node, 4 GPUs |
| `jobscript_sweep_2node.sh` | 2 nodes, 8 GPUs |
| `nixl_putget_bench.py` | NIXL `registerMem → transfer → deregister`, timed separately |
| `torch_dist_bench.py` | torch.distributed collectives, algbw **and** busbw |
| `pals_*_env.sh` | per-layer PALS rank→GPU wrappers |
| `../../tests/test_sweep_layers.py` | 23 offline tests, no cluster needed |

## Running

```bash
qsub jobscript_sweep_1node.sh          # all layers
qsub jobscript_sweep_2node.sh

# exclude a known-broken layer without editing the driver
qsub -v DLCOMM_LAYERS=fi,osu,nccl,torch_dist jobscript_sweep_1node.sh

# fast smoke test
qsub -v DLCOMM_QUICK=1 jobscript_sweep_1node.sh
```

Results land in `$DLCOMM_RUN_DIR/sweep_results.csv`, one row per cell:

```
layer,cell,axis,value,collective,dtype,extra,nodes,ranks,exit,rows,status,detail
```

## Reading `status`

| Status | Meaning |
|---|---|
| `ok` | ran **and produced measured rows** |
| `failed` | ran and did not produce rows — a real problem |
| `unavailable` | could not run here; `detail` says why |
| `unsupported` | the platform rejects this option (e.g. CXI WRITE `-260`) |

`status` is derived from measured rows, never from reaching the end of a
stage. An earlier version of this harness printed `STATUS=ok` for five layers
while every stage exited 127; the tests below exist to keep that from
recurring.

## Tests

```bash
python -m pytest tests/test_sweep_layers.py -v    # 23 passed
```

They execute the driver for real against stub binaries — no cluster, no PBS.
They cover the bugs this harness actually had:

- a hardcoded `/opt/cray/pe/pals/` launcher path (Tara uses `/opt/cray/pals/`)
  → every cell exited 127
- the NCCL reduction op passed as an env var instead of `-o <op>` → four
  identical "max/min/prod" rows that were all really `sum`
- `STATUS=ok` with zero rows
- OSU printing usage text and exiting 0 — indistinguishable from a real run in
  a batch log

## Platform notes

- **GTL is mandatory for `-d cuda`.** A non-GTL OSU build segfaults the moment
  it touches a device buffer. The driver checks `ldd | grep gtl` and refuses
  the device cells rather than producing a crash log.
- **`MPICH_GPU_SUPPORT_ENABLED=1` is set per-cell**, never globally — leaking
  it into a non-GTL step aborts that step and looks like a second bug.
- **Never sort `$PBS_NODEFILE`.** PALS gives rank 0 to its first line as
  written; reordering points `MASTER_ADDR` at a node with nothing listening
  and yields a connection refused that mimics a fabric fault.
- **CXI has no loopback** (`fi_domain()` returns `-38`), so `fi_pingpong` is a
  2-node-only measurement and is recorded `unavailable` with that reason at 1
  node instead of a misleading `0`.
- **`TMPDIR`**: PALS sets `/var/run/palsd/<uuid>`, which the job cannot write.
  Both jobscripts override it.
- **NIXL DRAM vs VRAM is a diagnostic, not just a variant.** If VRAM fails and
  DRAM passes, the fault is in GPU memory registration (dmabuf/IOMMU), not the
  fabric.
