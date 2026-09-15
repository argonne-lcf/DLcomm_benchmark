# Fix 22 — PyTorch transfer layer timed out on the fill, not the transfer

**Status:** fixed; re-run pending.
**Job:** 8824725, `PYTORCH_EXIT=124`.

## Symptom

The PyTorch layer of the transfer job produced no output and was killed at the
600 s cap:

```
############ LAYER 2: PyTorch XPU ############
PYTORCH_EXIT=124
rank 0 died from signal 15
```

The C++ layer in the same job completed normally, so the device, the fabric
and the job script were all working.

The job's overall `Exit_status` was 0. A failed layer inside a successful job
is invisible unless the per-layer exit code is read.

## Cause

`dl_comm/transfer/h2d.py::fill_shuffled` filled each buffer with a shuffled
iota, mirroring the reference's `std::shuffle`:

```python
perm = torch_mod.randperm(n, device=tensor.device, dtype=tensor.dtype)
tensor.copy_(perm.view(tensor.shape))
```

At the measured size that is `randperm` over 2^28 elements, for four buffers,
on twelve ranks. On host tensors `randperm` is single-threaded, so the fill
dominated the run and never reached the transfer being measured.

The reference's choice was sound in C++; the direct torch translation was not.

## Fix

```python
iinfo = torch_mod.iinfo(tensor.dtype)
tensor.random_(iinfo.min, iinfo.max)
```

The fill exists to defeat compression and zero-page optimisation in the
transfer path. `random_` satisfies that requirement and is parallel on both
host and device.

Two supporting changes:

- the layer timeout was raised from 600 s to 900 s;
- the measurement script prints progress markers, so a future hang reports
  where it stopped instead of producing an empty file.

## Test

`test_fill_does_not_use_randperm` fails if `randperm` reappears in
`fill_shuffled`. It strips the docstring before checking, since the docstring
names `randperm` to explain the exclusion.
