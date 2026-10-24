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

Cross-node NIXL on Slingshot works, **but only with the LIBFABRIC backend and
a correctly provisioned CXI environment.** Get any part of that wrong and it
still runs — over TCP, roughly 6x slower — with every correctness check
green. That failure mode is the main thing this example is built to catch.

| path | 64 MiB cross-node, same nodes, same allocation |
|---|---|
| MPI baseline (`mpi_fabric_baseline.py`), single pair | 12.4 – 24.2 GB/s |
| NIXL, LIBFABRIC backend, per rank (8–16 concurrent) | **2.8 – 5.5 GB/s** |
| NIXL, UCX backend (TCP fallback) | 0.17 – 0.67 GB/s |

Read the middle row as per-rank, not aggregate: 8 ranks at ~3–4 GB/s each is
~24 GB/s of concurrent traffic, against an MPI baseline that times **one**
pair. The two rows answer different questions and should not be divided.

### Why the UCX path is a floor, not a measurement

`nixl_cu12`'s bundled UCX has transports `ib`, `ib_mlx5`, `ib_efa`, `cma`,
`cuda` — no CXI provider. Tara has `/dev/cxi0..3` and no
`/sys/class/infiniband`, so UCX silently lands on TCP: `UCX_TLS=tcp`
reproduces the default numbers, and `UCX_TLS=rc,ud` fails outright with
`NIXL_ERR_BACKEND`. NIXL's separate **`libplugin_LIBFABRIC.so`** is the
backend that reaches the NIC, so `--backend-preference` defaults to
`LIBFABRIC,UCX,GDS,POSIX` and the jobscript **fails with
`VERDICT=WRONG_BACKEND`** rather than reporting TCP numbers as fabric ones.

### Four environment requirements, each with a misleading failure

| requirement | what you see if it is missing |
|---|---|
| `-l select=N:ngpus=4` | `Failed to find valid default rgroup and vni for cxi0`, then `fi_domain: Function not implemented`. PALS provisions a Slingshot VNI only for a GPU allocation; `fi_pingpong` fails the same way, so it is not a NIXL bug. |
| `NIXL_PLUGIN_DIR` + matching `libfabric` | agent constructs, then **every** transfer fails `NIXL_ERR_BACKEND` at `postXferReq`. Mixing one build's libfabric with another's plugins does this. |
| CUDA runtime on `LD_LIBRARY_PATH` | `prepareAndSubmitTransfer failed for descriptor 0 device N` — points at the fabric; the real cause is that NIXL cannot register device memory without `libcudart`. |
| `--op READ` on CXI | `fi_writedata failed on rail N: Function not implemented`. NIXL implements `WRITE` via `fi_writedata()`, which the cxi provider only offers with the `FI_RMA_EVENT` endpoint capability that NIXL does not request. `READ` (`fi_read`) has no such requirement and moves the same bytes. |

None of these name their own cause, which is why the jobscript prints
`SLINGSHOT_VNI`, `NIXL_IMPORTED_FROM` and `BACKENDS_USED` on every run and
gates on them. `VERDICT=NIXL_SHADOWED` exists because pointing
`EXTRA_PYTHONPATH` at a full site-packages drags that venv's NIXL in and
silently wins the import race.

## Run

```bash
# 2 nodes, 4 ranks/node. ngpus=4 is REQUIRED on Slingshot -- without it
# PALS provisions no VNI and the LIBFABRIC backend cannot open a domain.
qsub -A <account> -l select=2:ngpus=4 \
     -v PROJ=$HOME/<project-root>,RANKS_PER_NODE=4,\
VENV=<nixl-venv>,CUSTOM_LIBFABRIC=<cxi-libfabric>,\
NIXL_INSTALL_DIR=<nixl-install>,EXTRA_PYTHONPATH=<dir-with-only-mpi4py> \
     jobscript_nixl_remote.sh

# 4 nodes: same, with -l select=4:ngpus=4
```

`PROJ` must contain `DLcomm_benchmark/`. The fabric variables are all
optional and default to the system libfabric; set them when NIXL was built
against a site libfabric.

* `VENV` / `PYBIN` — the Python holding NIXL.
* `CUSTOM_LIBFABRIC` — CXI-capable libfabric (default
  `/opt/cray/libfabric/2.3.1`).
* `NIXL_INSTALL_DIR` — for a source-built NIXL; adds its `lib64` and
  `lib64/plugins`. Keep libfabric, `libnixl` and plugins from **one** build.
* `EXTRA_PYTHONPATH` — a directory containing **only** the missing module
  (typically `mpi4py`, needed for the out-of-band peer exchange). Never a
  full site-packages; see `VERDICT=NIXL_SHADOWED`.
* `NIXL_FABRIC=ucx` — skip the CXI block entirely on an IB/RoCE site.
* `NIXL_OP=WRITE` — opt back into RDMA write where `FI_RMA_EVENT` is
  available. Fails on CXI.

Other overrides: `SIZES`, `ITERATIONS`, `PAIRING` (`split` | `ring`),
`BACKEND_PREFERENCE`.

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

## Three things that will bite you

**1. Register memory *before* exchanging agent metadata.** NIXL agent
metadata carries the agent's registered-memory map. Exchange it first and
`initialize_xfer` fails with `NIXL_ERR_NOT_FOUND` and
`createXferReq: no potential backend found to be able to do the transfer`,
which names neither the cause nor the fix. A *new* registration needs a
*fresh* exchange, so the whole setup is redone per buffer size.

**2. A local `DONE` means different things for READ and WRITE.** For `READ`
(the default) your own completion does mean your destination is filled — you
pulled the bytes. For `WRITE` it means only that *your* write was delivered;
the buffer you are about to verify is filled by the **peer's** write, so
reading on local completion is a race that passes on a quiet machine. The
`WRITE` path therefore waits on the peer's notification, and that wait sits
**outside** the timed region. Folding a notification round-trip into the
timing turns latency into apparent bandwidth: the first spike measured
0.198 GB/s at 64 MiB that way, with a median an order of magnitude below the
best.

**3. A green correctness run can still be the wrong transport.** The first
working version of this example verified every byte across 2 and 4 nodes and
reported 0.67 GB/s — correct data, carried over TCP, on a 24 GB/s fabric.
Byte verification cannot detect that; only the backend gate and the MPI
baseline can. That is why both run on every invocation.

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
