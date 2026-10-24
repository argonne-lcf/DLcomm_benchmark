# 19 — Cross-node NIXL RDMA

One rank writes directly into a **remote** rank's device memory over the
fabric, and the received bytes are checked against the sender's payload.

## Why this exists alongside `18_nixl_transfer`

Example 18 measures NIXL `h2d` / `d2h` / `d2d` between buffers that both
belong to the local agent. Every transfer there is node-local: a 4-node run
of 18 is four independent node-local measurements and the fabric never
appears in the result. That is a real path and worth measuring, but it is not
the path a disaggregated serving deployment depends on — a prefill node
handing KV cache to a decode node crosses the network.

This example measures that crossing. It is additive; neither example replaces
the other, and 18 is the right node-local floor to compare against.

## Reading the numbers on this cluster (measured 2026-10-24, Tara)

**Cross-node NIXL here runs over TCP, not RDMA, and is ~50x slower than the
fabric.** This is not a tuning gap and not a property of Slingshot:

| path | 64 MiB cross-node, same two nodes, same allocation |
|---|---|
| MPI (`mpi_fabric_baseline.py`) | **12.4 GB/s** |
| NIXL cross-node (this example) | **0.17 – 0.30 GB/s** |

Evidence that the cause is transport availability in the NIXL build:

* Tara's NIC is Slingshot: `/dev/cxi0..3` present, **no** `/sys/class/infiniband`,
  no Mellanox device.
* The pip-installed `nixl_cu12` bundles its own UCX, whose transport plugins
  are exactly `libuct_ib`, `libuct_ib_mlx5*`, `libuct_ib_efa`, `libuct_cma`,
  `libuct_cuda`. **There is no CXI/libfabric provider in that list.** A
  transport that is not present cannot be selected, whatever `UCX_TLS` says.
* `UCX_TLS=tcp,cuda_copy,self` reproduces the default numbers almost exactly
  (0.19–0.38 GB/s vs 0.20–0.32 GB/s default) — i.e. **the default path is
  already TCP**.
* `UCX_TLS=rc,ud,cuda_copy,self` fails outright with `NIXL_ERR_BACKEND`:
  UCX cannot build an RC/UD transport because there is no IB device.
* The curve is flat from 1 MiB to 256 MiB (~0.17 GB/s at 256 MiB), so this is
  bandwidth-limited, not latency-limited.

The system *does* ship `/opt/cray/libfabric/2.3.1`, so the fix is a NIXL/UCX
build with a libfabric-CXI (or OFI) transport rather than anything in this
benchmark. Until then, treat cross-node NIXL numbers on Slingshot as a
**transport-availability floor, not a fabric measurement**. The jobscript runs
the MPI baseline on every invocation so this comparison is always in the log.

## Run

```bash
# 2 nodes, 4 ranks/node
qsub -A <account> -l select=2 \
     -v PROJ=$HOME/<project-root>,RANKS_PER_NODE=4 \
     jobscript_nixl_remote.sh

# 4 nodes
qsub -A <account> -l select=4 \
     -v PROJ=$HOME/<project-root>,RANKS_PER_NODE=4 \
     jobscript_nixl_remote.sh
```

`PROJ` must contain `DLcomm_benchmark/` and the venv (`tcvenv/` by default;
override with `VENV=` or `PYBIN=`). Useful overrides: `SIZES`, `ITERATIONS`,
`PAIRING` (`split` | `ring`).

Single-node submission exits 1 with `VERDICT=SINGLE_NODE` — this example
measures the fabric, so there is nothing for it to do on one node.

## Files

| file | role |
|---|---|
| `run_nixl_remote.py` | MPI runner; emits `NIXL_RMAP` / `NIXL_REMOTE_RESULT` |
| `mpi_fabric_baseline.py` | MPI cross-node baseline, same nodes, same allocation |
| `jobscript_nixl_remote.sh` | PBS driver + verification gates |
| `test_nixl_remote.py` | 18 offline tests; no GPU, NIXL, MPI, or fabric needed |

Implementation lives in `dl_comm/transfer/nixl_remote.py`.

Run the offline tests anywhere, including a login node:

```bash
python3 examples/19_nixl_remote_rdma/test_nixl_remote.py   # 18 tests, ~0.1s
```

## What is verified, and why exit 0 is not enough

A NIXL `WRITE` to a wrong-but-valid remote address still reports `DONE`. A
bandwidth number from such a transfer is not a slow result, it is a
meaningless one. So each rank sends a sender-identifiable payload —
element *i* of rank *r* is `r * 1000003 + i` — into its partner's buffer, and
afterwards compares what it **received** against its partner's expected
payload. A destination that was never written fails (it is filled with `-1`,
not zeros, since zero is a value a real payload could hold), and a buffer
written by the wrong peer fails too. A mismatch raises
`NixlVerificationError` and the job exits non-zero; it is never downgraded to
a warning.

The jobscript additionally gates on:

* `REMOTE_RECORDS` > 0 — a run that skipped every measurement also exits 0.
* no `NIXL_REMOTE_CORRUPT` line.
* `UNVERIFIED_ROWS == 0` — every result row carries `verified=True`.
* `SAME_NODE_PAIRS == 0` — a pairing that degenerates to same-node traffic
  produces a healthy-looking number for the wrong thing.
* `DISTINCT_HOST_DEV == NRANKS` — ranks sharing a GPU inflate per-device
  bandwidth while every other check still passes.

These gates work: the first 2-node attempt exited 143 with
`VERDICT=NO_RECORDS` because the jobscript had not activated the venv and the
system `python3` rejected `from __future__ import annotations`. Nothing was
measured and nothing was reported as measured.

## Two things that will bite you

**1. Register memory *before* exchanging agent metadata.** NIXL agent
metadata carries the agent's registered-memory map. Exchange it first and
`initialize_xfer` fails with `NIXL_ERR_NOT_FOUND` and
`createXferReq: no potential backend found to be able to do the transfer`,
which names neither the cause nor the fix. A *new* registration needs a
*fresh* exchange, so the whole setup is redone per buffer size.

**2. A local `DONE` does not mean your buffer is filled.** It means *your*
write was delivered to the peer. Reading your own destination on that signal
is a race that passes on a quiet machine. Inbound completion is established
by the peer's notification, which is why every transfer carries one — and
why that wait sits **outside** the timed region. Folding a notification
round-trip into the timing turns latency into apparent bandwidth: the first
spike measured 0.198 GB/s at 64 MiB that way, with a median an order of
magnitude below the best.

## Pairing

`split` pairs rank *r* with `r + world/2`. With block placement
(`mpiexec -ppn`) both halves land on different nodes, and it is symmetric for
any even world size.

`ring` pairs *r* with `r + ranks_per_node`, giving every rank a partner one
node away. **It is only symmetric at exactly two nodes** — at three or more a
fixed forward step is a cycle, not a pairing, and the mutual-exchange
assumption breaks. `test_ring_is_symmetric_only_for_two_nodes` pins that
limitation rather than leaving it to be discovered on a hung 4-node job.

Pairings are never *assumed* to be cross-node: the runner compares actual
hostnames and records the answer in the result row.
