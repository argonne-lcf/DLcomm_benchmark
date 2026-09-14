# Fix 26 — torchcomms: ABI pairing and the stock XCCL stub backend

**Status:** bootstrap verified by job 8825144 (`new_comm` returns on all 12 and all 24
ranks, `TORCHCOMMS_RUN_EXIT=0`). Operation coverage on the 0.3.0 stack is under test in
job 8825168.

Five consecutive jobs reported `TORCHCOMMS_RUN_EXIT=139` (SIGSEGV) at 12 ranks and `143`
at 24. The fault was inside `torchcomms.new_comm`, confirmed by `faulthandler`. Three
hypotheses were tested and rejected before the cause was found.

## Rejected hypotheses

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | PMIx bootstrap mismatch | added `--pmi=pmix --envall` | segfault unchanged |
| 2 | `LOCAL_WORLD_SIZE` unset | read from one reference file | **never consulted** by the extension |
| 3 | Device isolation via `ZE_AFFINITY_MASK` | job 8825078, mask applied per rank | segfault unchanged |

Hypothesis 2 deserves note. It was adopted after reading a single launcher and looked
plausible. `strings` on the extension shows the only environment variables it reads are:

```
MASTER_ADDR
MASTER_PORT
TORCHCOMM_XCCL_BOOTSTRAP_UNIQUEID_EXCHANGE_METHOD
```

`LOCAL_WORLD_SIZE` is absent. Had the job passed for an unrelated reason, a change with no
mechanism would have been recorded as the fix.

Hypothesis 3 was adopted because **81 of 81** torchcomms launchers under
`datascience_collab/pshukla` use the mask pattern. Unanimity across a corpus is still
correlation: job 8825078 applied the mask correctly on every rank and segfaulted
identically. The reference harnesses in `torch_and_comm_reference` do not use the mask at
all — they index the device by local rank, exactly as this benchmark already did.

## Cause

The `env2` environment was internally inconsistent. Importing torch there fails outright:

```
ImportError: env2/.../libsycl.so.9: undefined symbol: urDeviceWaitExp,
             version LIBUR_LOADER_0.12
```

and its torchcomms extension cannot resolve torch at all:

```
$ ldd env2/.../torchcomms/_comms_xccl.cpython-312-x86_64-linux-gnu.so
        libc10.so => not found
        libc10_xpu.so => not found
        libtorch_xpu.so => not found
        libtorch_python.so => not found
```

An earlier workaround prepended `$ENVPREFIX/lib` to `LD_LIBRARY_PATH`, which resolved the
symbol far enough for the import to succeed and for execution to reach `new_comm`, where
the mismatched ABI faulted. The workaround converted a clear import error into a
segfault four call layers away.

A custom torch build and a separately built extension must be used as a matched pair. The
reference job scripts do this explicitly:

```bash
export CUSTOM_TORCH=.../pytorch_c10d_torchcomms/pytorch
export CUSTOM_TORCHCOMMS=.../torchcomms_custom_torch
export PYTHONPATH=${CUSTOM_TORCHCOMMS}:${CUSTOM_TORCH}:${PYTHONPATH}
export LD_LIBRARY_PATH=${CUSTOM_TORCH}/torch/lib:${LD_LIBRARY_PATH}
```

## The stock backend is a stub

Running on the frameworks module removes the segfault, but the backend implements almost
nothing. Job 8825144, both scales:

```
TC_VERSION=0.1.0  TC_TORCH=2.10.0a0+git449b176
[yes] all_reduce
[NO ] barrier    XCCL barrier is not supported now and will be added later
... 12 of 13 operations identical
TC_SUPPORTED=1/13
```

Comparing the two binaries directly:

```
stock 0.1.0    16 "XCCL <op> is not supported" strings
0.3.0          0
```

The 0.3.0 build carries no unsupported-operation markers, including
`all_to_all_v_single`. Stack selection is therefore explicit:

- `DLCOMM_TC_STACK=frameworks` (default) — bootstraps, `all_reduce` only
- `DLCOMM_TC_STACK=pshukla` — torchcomms 0.3.0 with its matching torch

## Probe defects found by inspecting live signatures

The first run on the stock stack reported `TC_SUPPORTED=0/13` while still exiting 0. Every
call raised `TypeError`:

```
all_reduce(): incompatible function arguments
```

This build takes `async_op` as a **required positional** on every collective. Dumping the
live pybind11 signatures rather than assuming revealed two further errors in the probe:

- `reduce(tensor, root, op, async_op)` — root precedes op; the probe had them swapped
- `gather(output_tensor_list, input_tensor, root, async_op)` — list first

The reference `perf_test_helpers.py` handles the same divergence in `_resolve`, remapping
`gather` argument order between the c10d and torchcomms conventions.

## Reporting defect

A run that executed every operation and failed all 13 was indistinguishable in the job
summary from a clean pass: exit 0, no visible records, because the summary grep did not
match `TC_SUPPORTED`. The stage now emits an explicit verdict:

- `FAIL_ALL_OPS` when the matrix is `0/N`
- `PARTIAL` when some but not all probed operations are implemented
- `OK` only when all are

An exit status of 0 is not evidence that a layer measured anything.
