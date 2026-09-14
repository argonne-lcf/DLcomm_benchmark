# Example 16 — all five layers, both scales

Runs every measurement layer at 1 node / 12 ranks and 2 nodes / 24 ranks, and
produces the cross-layer comparison.

## Contents

| File | Purpose |
|---|---|
| `jobscript_all_layers.sh` | wrapper around `tools/run_all_scales.sh` |

## Running

```
qsub jobscript_all_layers.sh                          # shipped torchcomms (0.1.0)
qsub -v DLCOMM_TC_STACK=pshukla jobscript_all_layers.sh   # torchcomms 0.3.0
```

Results are written to
`/lus/flare/projects/datascience/kaushik/DLcomm/validation/<timestamp>/`, one
directory per scale.

## Layers

| Layer | Buffers | Path |
|---|---|---|
| OSU / MPI | host | MPI collectives |
| C++ oneCCL | device | CCL called directly from C++ |
| C++ SYCL transfer | device | H2D / D2H / D2D copies |
| torch.distributed | device | XCCL through PyTorch |
| torchcomms | device | XCCL through the torchcomms API |

OSU uses host buffers while the other layers use device buffers, so OSU
columns are not same-path with the rest. The comparison tool reports the
difference rather than computing a ratio across it; see the cross-layer
comparison section of the top-level README.

## torchcomms stack selection

The torchcomms that ships with `frameworks/2025.3.1` is version 0.1.0. It
implements `all_reduce` and stubs the remaining operations with
`XCCL <op> is not supported now and will be added later`, so a default run
produces one torchcomms column.

`DLCOMM_TC_STACK=pshukla` selects the 0.3.0 build, which implements 12 of 12
probed operations including point-to-point. That build pairs with torch
2.13, while the other layers run torch 2.10, so a torchcomms-versus-PyTorch
gap carries a version difference as well as a library difference. See
`docs/findings/04` for the size of that effect where it has been measured.

## Why this is a wrapper

The harness itself is `tools/run_all_scales.sh`, which produced the published
numbers. Copying it into the example directory would create a second copy to
keep in step, and the two would drift. The wrapper only locates and executes
it.

## Comparing results afterwards

```
python -m dl_comm.analysis.compare_layers <results-dir>/1node_12rank --ranks 12
```
