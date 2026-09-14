# torchcomms 0.3.0 + matching PyTorch — local copy

A working torchcomms 0.3.0 stack for Aurora, copied into
`/lus/flare/projects/datascience/kaushik/stacks/torchcomms_0.3.0/` so DLcomm
runs no longer depend on another user's home directory.

## Why this copy exists

DLcomm's torchcomms layer needs a build that actually implements the
collectives. The torchcomms shipped with `frameworks/2025.3.1` does not.

| Stack | torchcomms | torch | Stub markers in `_comms_xccl.so` | Usable ops |
|---|---|---|---|---|
| `frameworks/2025.3.1` | 0.1.0 | 2.10.0a0+git449b176 | 17 | `all_reduce` only |
| this copy | 0.3.0 | 2.13.0a0+git3300461 | 0 | 12 of 12 probed |

"Stub markers" is the count of `is not supported now` strings in the compiled
extension, from `strings <so> | grep -c "is not supported now"`. The shipped
build raises `XCCL <op> is not supported now and will be added later` for
everything except `all_reduce`; the 0.3.0 build contains no such string.

The original was built by user `pshukla` under `datascience_collab`. That
directory is readable today but is outside this project's control: if it is
moved, rebuilt or cleaned, every run pointing at it breaks. This copy removes
that dependency.

## Layout

```
stacks/torchcomms_0.3.0/
├── torchcomms/            torchcomms 0.3.0 package (add to PYTHONPATH)
│   ├── torchcomms/
│   └── torchcomms-0.3.0.dist-info/
├── pytorch/
│   └── torch/             torch 2.13.0a0 build the extension links against
└── notes/
    └── README.md          this file
```

## Usage

Both paths go on `PYTHONPATH` together, and the torch library directory goes
on `LD_LIBRARY_PATH`:

```bash
STACK=/lus/flare/projects/datascience/kaushik/stacks/torchcomms_0.3.0
export PYTHONPATH="$STACK/torchcomms:$STACK/pytorch:$PYTHONPATH"
export LD_LIBRARY_PATH="$STACK/pytorch/torch/lib:$LD_LIBRARY_PATH"
```

Verify before measuring anything:

```bash
python3 -c "import torchcomms, torch; print(torch.__version__, torchcomms.__file__)"
```

Expect `2.13.0a0+git3300461` and a path under this directory. A torch version
of `2.10.0a0+git449b176` means the module's torch won the import and the stack
is not in effect.

## Selecting this stack

DLcomm selects it with `DLCOMM_TC_STACK=local`:

```bash
qsub -v DLCOMM_TC_STACK=local examples/17_torchcomms/jobscript_torchcomms.sh
```

The value was previously `pshukla`, after the directory the build was copied
from. That name is still accepted as a deprecated alias and maps to `local`,
printing `TC_STACK_NOTE=pshukla is a deprecated alias for local`, so
submissions written against the old name keep working.

## The two halves are a matched pair

torchcomms is a C++ extension compiled against a specific torch ABI. Using the
0.3.0 extension with the module's torch 2.10 segfaults inside `new_comm`. That
failure looks like a configuration error — wrong device, bad bootstrap — but no
amount of configuration fixes it. Either both paths are set or neither is.

This is what broke the earlier `env2` attempt: a separately built torchcomms
against torch 2.13.0+xpu produced `undefined symbol: urDeviceWaitExp` and
unresolved `libc10.so`.

## Build provenance

Recovered from `build_pytorch_c10d_torchcomms.log` in the source tree:

- torch version `2.13.0a0+git3300461`, built from the `c10d_torchcomms` branch
- compiler `icpx` 2025.3.2 from `/opt/aurora/26.26.0/oneapi/compiler/latest`
- CMake 3.31.10, Ninja generator
- `-DBUILD_TEST=False -DBUILD_PYTHON=True -DUSE_NUMPY=True`
- Python `/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python`
  (3.12), so the extensions are `cpython-312` and require that interpreter
- `CMAKE_PREFIX_PATH` pointed at the `26.26.0` oneAPI tree, including
  `oneapi/ccl/latest/lib/cmake/oneCCL`

The interpreter matters: these are `cpython-312` binaries and will not import
under a different Python minor version.

## What was not copied

The source directories total 308 GB, almost entirely test fixtures:

| Path | Size | Copied |
|---|---|---|
| `torchcomms/tests/` | 231 GB | no |
| `c10d_torchcomms_perf/` | 76 GB | no |
| `torchcomms/experiments/` | 1.5 GB | no |
| `torchcomms/` (package) | ~51 MB | yes |
| `pytorch/torch/` | 2.9 GB | yes |

The PyTorch source and build trees were not copied either; only the installed
`torch/` package, which is what `PYTHONPATH` needs. This copy is for running,
not for rebuilding. Rebuilding requires the upstream sources and the log above.

## Known limitation

`split()` hangs at 12 ranks on this build: oneCCL `split_communicator` →
`onecclCommSplit` → `DefaultXcclApi::commSplit` does not return. DLcomm gates
the call behind `DLCOMM_TC_SPLIT` and leaves it off by default.

## Version confound

torchcomms here runs torch 2.13 while DLcomm's other layers run the module's
torch 2.10, so a torchcomms-versus-`torch.distributed` comparison carries a
torch version difference as well as a library difference. Measured on allgather
at 24 ranks, the two agree within 4.0 % at 1 MiB and 1.2 % at 4 MiB, so the
version cannot account for large gaps in that operation. No equivalent bound
has been established for alltoall or reduce_scatter.
