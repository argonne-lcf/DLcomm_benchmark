# Fix 24 — pals_env.sh must not set PMI_RANK/PMI_SIZE

**Status:** fixed; found by job 8824800

**Symptom:** every MPI-backed Python layer aborted at startup on Aurora.

```
Abort(16): Fatal error in internal_Init_thread: Internal MPI error!
```

torch.distributed exited 16 and torchcomms died on signal 11, at both 12 and
24 ranks, across jobs 8824800, 8824902, 8824944 and 8824968. The C++ oneCCL
and OSU layers were unaffected.

## Wrong diagnoses that cost four jobs

1. "oneCCL's ATL-MPI transport calls MPI_Init and aborts under PMIX, so use
   `CCL_ATL_TRANSPORT=ofi`." The ALCF documentation specifies `mpi`; `ofi` is
   not a supported transport here. Corrected by the user.
2. "MPI must be initialized before oneCCL touches it, so import mpi4py
   first." This is the pattern in the ALCF DDP example, but it did not fix
   the abort, because MPI initialization was not the problem.
3. "mpi4py's own MPI_Init is aborting." Probe 8824999 disproved this: bare
   mpi4py under the same launcher and the same environment initialized
   cleanly on 4/4 ranks.

Each of these blamed a component of the stack. The defect was local.

## Root cause

`pals_env.sh` exported:

```bash
export PMI_RANK="$RANK"
export PMI_SIZE="$WORLD_SIZE"
```

MPICH reads the `PMI_*` namespace for its own process-manager bootstrap.
Overwriting those variables corrupts `MPI_Init` for every stack that links
MPICH, which is why the failure was invisible to the pure-SYCL layers and
universal everywhere else.

The variables were added so torchcomms could discover a world size that PALS
does not publish. `PALS_SIZE` alone achieves that safely: it is the variable
PALS genuinely omits, and torchcomms reads it directly.

## Evidence

Probe 8825011, 12 ranks, one variable changed per case:

| case | mpi4py | CCL_ATL_TRANSPORT | via pals_env.sh | result |
|---|---|---|---|---|
| C | no | mpi | yes | Abort(16) |
| E | yes | mpi | yes | Abort(16) |
| F | yes | module default | yes | Abort(16) |

Probe 8824999, same launcher and environment, **not** via the wrapper:

| case | result |
|---|---|
| bare mpi4py | 4/4 ranks initialized |
| mpi4py + ALCF minimal set | 4/4 ranks initialized |

The wrapper is the only factor present in every failure and absent from every
success.

Case G of probe 8825011 (torchcomms) produced no usable output: the filter in
the probe script was `head -8`, and seven glog warnings consumed it before any
result line. That case is inconclusive, not a pass.

## Fix

Remove the `PMI_RANK` / `PMI_SIZE` exports, keep `PALS_SIZE`, and record why
at the site so they are not reinstated.

## Related corrections adopted at the same time

From a working Aurora reference harness
(the Aurora XCCL lifecycle reference, `aurora_xccl_lifecycle_rank.py`):

- Do not import mpi4py; read `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from the
  environment with `PALS_*` fallbacks.
- Select the device by **local rank**, not global rank. The previous
  `rank % device_count` happens to be correct on one node and selects the
  wrong devices on two or more.
- Pass `device_id=device` to `init_process_group`.
- Pin `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` rather than the module default
  `opencl:gpu;level_zero:gpu`, which exposes each tile through two backends.

## Verification

Pending job 8825030. The claim is verified only if torch.distributed and
torchcomms both report results at 12 and 24 ranks with `TILE_CHECK=PASS`.
