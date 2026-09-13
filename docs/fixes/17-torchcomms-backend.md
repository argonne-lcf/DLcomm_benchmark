# Feature 17 — torchcomms as a `ccl_backend`

**Files:** `dl_comm/comm/torchcomms_backend.py` (new), `dl_comm/dl_comm_main.py`,
`dl_comm/config/config_spec.json`, `tests/test_torchcomms_backend.py` (new)

## What torchcomms is

`torchcomms` is Meta's experimental replacement for the `torch.distributed`
collectives API, open-sourced in October 2025 alongside NCCLX (a backend scaled
past 100,000 GPUs). It ships native NCCLX, RCCLX, NCCL, RCCL, **XCCL**, and
Gloo transports, so Intel GPUs are a first-class target rather than an
afterthought.

Verified present on Aurora, no installation required:

```
torch: 2.10.0a0+git449b176
torchcomms present: True
xccl available: True
```

## Why a backend and not a framework

`framework` and `ccl_backend` are separate axes in DLcomm's config. The initial
instinct was to add `torchcomms` next to `pytorch` and `jax`, but that is the
wrong axis on two counts.

Conceptually, torchcomms is a *transport*: it moves the same `torch.Tensor`
objects that `torch.distributed` moves. It belongs where `xccl` and `nccl`
already live.

Mechanically, the framework axis is expensive. The codebase branches on
`framework ==` in **56 places** across `dl_comm_main.py`, `collectives.py`, and
`utils/utility.py`. A third framework means auditing every one of them and
multiplying the framework x backend matrix. Adding a backend value touches two
sites.

## Design: a `dist`-shaped facade

The 13 registered collectives already receive their communication module as a
`dist=` parameter:

```python
run_collective(x, op_obj, group=my_within_group, dist=dist, framework=framework)
```

That parameter is the seam. `TorchCommsDist` implements the `torch.distributed`
surface those collectives call, backed by a torchcomms communicator, and
`dl_comm_main` binds `dist` to it when `ccl_backend: torchcomms`.

**No collective was modified.** All 13 work through the adapter unchanged.

Coverage is enforced by test, not by inspection: `test_torchcomms_backend.py`
greps every `dist.<name>` out of `collectives.py` *and* `dl_comm_main.py` and
asserts the adapter implements each one.

## Two API mismatches handled

**1. Object-oriented vs module-level.** torch.distributed is
`dist.all_reduce(t, group=g)`; torchcomms is `comm.all_reduce(t, op, async_op)`
on a communicator. Subcommunicators come from `comm.split(ranks, name)`, which
`new_group()` maps onto DLcomm's `group` concept, caching by rank tuple.

**2. Mandatory `async_op` and `TorchWork` returns.** Every torchcomms op takes a
positional `async_op` and returns a work handle. Blocking calls pass `False`;
`isend`/`irecv` pass `True` and wrap the handle in `_Work`, which exposes
`.wait()` so the calling code is identical either way.

## Op coverage

Every DLcomm collective maps to a native torchcomms op — no emulation:

| DLcomm | torchcomms |
|---|---|
| `allreduce` | `all_reduce` |
| `reduce` | `reduce` |
| `broadcast` | `broadcast` |
| `allgather` | `all_gather` |
| `gather` / `scatter` | `gather` / `scatter` |
| `reducescatter` | `reduce_scatter` |
| `alltoall` | `all_to_all` |
| `alltoallsingle` | `all_to_all_single` |
| `alltoallv` | `all_to_all_v_single` |
| `barrier` | `barrier` |
| `sendrecv` | `send` / `recv` |
| `sendrecv_async` | `send` / `recv` with `async_op=True` |

Reduce ops map through `ReduceOp`, including `AVG`, `PREMUL_SUM`, and the
bitwise variants. DLcomm's `mean` is translated to torchcomms' `AVG`, and
`prod` to `PRODUCT`.

## Deliberate refusals

`recv`/`irecv` **raise** when `src is None`. torch.distributed permits a
wildcard receive; torchcomms does not. Silently accepting a message from the
wrong peer would corrupt a correctness check while still reporting a pass, so
the adapter refuses rather than guesses.

Unimplemented `dist.*` attributes are left absent rather than stubbed, so a
missing entry point raises `AttributeError` instead of silently degrading to a
different transport.

## Shutdown

`destroy_process_group()` maps to torchcomms' `finalize()`, finalizing split
communicators before the parent — releasing a parent while a child is live is
undefined.

This gap was found by the LSP, not by the first version of the test: the
original conformance test only scanned `collectives.py`, and
`destroy_process_group` is called from `dl_comm_main.py`. Missing it would have
crashed at the very end of a run, after every measurement was taken. The test
now covers both files.

## Verification

Sabotage-proven, per the project's standard that an unfired threshold is
untested. Renaming `destroy_process_group` in the adapter:

```
E       assert not ['destroy_process_group']
FAILED tests/test_torchcomms_backend.py::test_adapter_covers_every_dist_call_the_main_loop_makes
1 failed, 9 passed
```

Restored: `10 passed`. Full suite: **159 passed**.

## Status

The adapter is complete and unit-tested on CPU. It has **not yet been executed
on Aurora hardware** — that requires a job with `ccl_backend: torchcomms`. Until
that run produces a correctness verdict, torchcomms support is implemented but
unproven.

## Usage

```yaml
framework: pytorch
ccl_backend: torchcomms
```

torchcomms derives rank and world size from `RANK`/`WORLD_SIZE` and bootstraps
via `MASTER_ADDR`/`MASTER_PORT`. DLcomm launches under mpiexec rather than
torchrun, so `dl_comm_main` exports those from the MPI rank before creating the
communicator.
