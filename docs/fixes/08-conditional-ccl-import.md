# Fix 8 — Unconditional `oneccl_bindings_for_pytorch` import

**Severity:** critical — the benchmark could not start on Aurora's current module stack
**File:** `dl_comm/dl_comm_main.py`
**Found by:** Aurora validation job 8824229

## Defect

Selecting the `xccl` or `ccl` backend imported two Intel shims
unconditionally:

```python
if ccl_backend in ["xccl", "ccl"]:
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch
```

`oneccl_bindings_for_pytorch` is no longer shipped in
`frameworks/2025.3.1`. torch 2.10 provides XCCL natively through
`torch.distributed`. Measured on an Aurora login node:

```
backends: ['xccl', 'gloo']                                  <- native XCCL present
ModuleNotFoundError: No module named 'oneccl_bindings_for_pytorch'
ipex 2.10.10+gitd0f992f                                     <- IPEX still present
```

Every rank died at import, before any collective executed. The benchmark was
unrunnable on the current stack regardless of configuration.

## Fix

Import both shims defensively, and fail only when the requested backend is
genuinely unavailable — determined by asking `torch.distributed`, not by
assuming the shim's presence implies the backend:

```python
if ccl_backend in ["xccl", "ccl"]:
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except ImportError:
        pass
    try:
        import oneccl_bindings_for_pytorch  # noqa: F401
    except ImportError:
        native = getattr(dist, f"is_{ccl_backend}_available", lambda: False)()
        if not native:
            raise RuntimeError(
                f"backend '{ccl_backend}' is not available: "
                f"oneccl_bindings_for_pytorch is not installed and "
                f"torch.distributed has no native {ccl_backend} support "
                f"in this build")
```

This keeps the benchmark working on older stacks that still ship the shim,
works on current stacks that do not, and produces an actionable message when
the backend is truly missing.

## Notes

This class of defect is invisible to code review: the source is unchanged and
correct against the environment it was written for. It appears only when the
module stack moves underneath it, and only when the benchmark is actually
executed on the target machine.

Equivalent unguarded imports remain in `tools/examples_dl_scaling/` and the
legacy `tests/*.py` scripts. Those are standalone utilities outside the
benchmark entry point and were left as shipped.
