# Fix 23 — PALS provides no global size, so every rank saw world=1

**Status:** cause proved by a dedicated probe job; wrapper written and
unit-checked locally. Hardware validation is in job 8824800.
**Jobs:** 8824754, 8824771 (segfault), 8824781 (clean abort), 8824786
(environment dump).

## Symptom

Every torchcomms rank died during communicator construction:

```
KILLED BY SIGNAL: 11 (Segmentation fault)      x12
```

The crash was inside XCCL bootstrap, with no Python traceback.

## What was wrong with the first two diagnoses

The first theory was that passing an externally built `TCPStore` into
`new_comm()` corrupted bootstrap. Removing the store did not change the
outcome, so that theory was wrong.

The second theory was the launcher. That one was real but incomplete:
activating the conda env put `env2/bin/mpiexec` (a generic MPICH that ships
with the environment) ahead of `/opt/cray/pals/1.8/bin/mpiexec` on `PATH`.

```
before conda activate:  /opt/cray/pals/1.8/bin/mpiexec
after  conda activate:  .../env2/bin/mpiexec
```

Fixing the launcher removed the segfault but the ranks still failed, now with
a clear message instead of signal 11. The launcher was necessary, not
sufficient.

## Root cause

Job 8824786 dumped the real compute-node environment under the PALS launcher:

```
PALS_RANKID=0  PALS_LOCAL_RANKID=0  PALS_LOCAL_SIZE=4  PALS_NODEID=0
PALS_APID=...  PALS_DEPTH=1  PALS_PMI=pmix  PALS_TRANSFER=0
```

There is no global size variable. No `PALS_SIZE`, no `PMI_SIZE`, no
`WORLD_SIZE`. torchcomms' own source acknowledges this:

```c
// Note: PALS does not provide an env variable for size, like `PALS_SIZE`.
// TODO: replace with the correct PALS env var for size once it is available.
int size = env_to_value<int>("PALS_SIZE", -1);
size = env_to_value<int>("PMI_SIZE", -1);   // MPICH only
```

So `query_ranksize()` returned size 1 on all twelve ranks. Twelve independent
one-rank jobs then tried to form a communicator, and bootstrap segfaulted.
This is an upstream gap on Aurora's launcher, not a misconfiguration.

## Fix

`pals_env.sh` wraps each rank and derives the missing value:

```sh
export RANK="$PALS_RANKID"
export LOCAL_RANK="$PALS_LOCAL_RANKID"
export WORLD_SIZE="$(( PALS_LOCAL_SIZE * DLCOMM_NNODES ))"
export PMI_RANK="$RANK"  PMI_SIZE="$WORLD_SIZE"  PALS_SIZE="$WORLD_SIZE"
```

`DLCOMM_NNODES` comes from `PBS_NODEFILE` in the job script. `PMI_*` and
`PALS_*` are both set so the C++ and Python paths cannot discover different
topologies.

`MASTER_ADDR` is required from the caller rather than defaulted to localhost:
a per-node master gives each node its own rendezvous and hangs instead of
failing.

## Verification

Locally, before spending a queue slot:

| input | RANK | WORLD_SIZE |
|---|---|---|
| `PALS_RANKID=3  PALS_LOCAL_SIZE=12  DLCOMM_NNODES=1` | 3 | 12 |
| `PALS_RANKID=13 PALS_LOCAL_SIZE=12 DLCOMM_NNODES=2` | 13 | 24 |

Both guards were confirmed to fire:

```
DLCOMM_NNODES: caller must export DLCOMM_NNODES from PBS_NODEFILE
MASTER_ADDR: caller must export MASTER_ADDR
```

Hardware validation at 12 and 24 ranks is job 8824800. Until that job
reports, torchcomms remains unproven on XPU.

## Guards added

- The job script aborts unless `mpiexec` resolves under `/opt/cray/pals/`.
- `probe_tc03.py` exits with an explanatory message when `world < 2` rather
  than continuing into a segfault.
- `MASTER_ADDR` is mandatory.

## Lesson

Two plausible diagnoses were wrong before the right one. Neither was
disproved by reasoning: each was disproved by an experiment that changed one
variable. The environment dump cost one ten-minute job and ended the guessing.
