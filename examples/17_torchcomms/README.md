# Example 17 — torchcomms through the standard runner

Runs torchcomms as a DLcomm backend through `dl_comm.dl_comm_main`, driven by
YAML like every other example. One Aurora node, 12 XPU ranks.

## Contents

| File | Purpose |
|---|---|
| `17_torchcomms_xccl.yaml` | five operations with `ccl_backend: torchcomms` |
| `jobscript_torchcomms.sh` | PBS submission script with stack selection |

## Running

```
qsub jobscript_torchcomms.sh                              # torchcomms 0.1.0
qsub -v DLCOMM_TC_STACK=local jobscript_torchcomms.sh     # torchcomms 0.3.0
```

Results are written to `logs/run_<timestamp>/`.

## Relationship to example 16

Example 16 runs the standalone comparison harness, which measures five layers
side by side and produces the cross-layer table. This example runs torchcomms
alone through the normal DLcomm entry point, so the backend is exercised
through the same configuration, verification and reporting path as
`torch.distributed`.

Use this one to check that torchcomms behaves as a DLcomm backend; use example
16 to compare it against the other layers.

## Which build is on `PYTHONPATH` decides what runs

This is the single most important variable in this example.

| Stack | Version | Behaviour |
|---|---|---|
| `frameworks` (default) | 0.1.0 | implements `all_reduce`; other operations raise `XCCL <op> is not supported now and will be added later` |
| `local` | 0.3.0 | 12 of 12 probed operations work, including point-to-point |

With the default stack, expect one of the five sections to pass and four to
report the operation as unsupported. That is the shipped build's limitation,
not a DLcomm defect, and it is the reason the two operations that work
everywhere are ordered first in `order_of_run`.

The jobscript prints `TC_VERSION`, `TORCH_VERSION` and `TC_PATH` before
measuring anything, so the log records which build actually loaded rather than
which one was intended.

## Matched pairs

The 0.3.0 build must be used with the torch build it was compiled against.
Mixing a custom torch with an independently built torchcomms segfaults inside
`new_comm` — an ABI mismatch, not a configuration error. The jobscript sets
both paths together and fails with `VERDICT=STACK_MISSING` if either is
unreadable. See `docs/fixes/26`.

Note that the 0.3.0 stack carries torch 2.13 while the other layers run torch
2.10, so a torchcomms-versus-PyTorch comparison includes a version difference.
`docs/findings/04` bounds that effect where it has been measured.

## Watchdog

`DLCOMM_WATCHDOG=900` is set because the 0.3.0 communicator split path hangs
at 12 ranks. A stub backend refuses an operation and returns, but a hang would
otherwise consume the full walltime and produce no diagnostic output.
