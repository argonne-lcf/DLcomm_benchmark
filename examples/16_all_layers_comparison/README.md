# Example 16 — all layers, both scales

Runs every measurement layer at 1 node / 12 ranks and 2 nodes / 24 ranks, and
produces the cross-layer comparison.

## Contents

| File | Purpose |
|---|---|
| `jobscript_all_layers.sh` | wrapper around `tools/run_all_scales.sh` |

## Running

```
qsub jobscript_all_layers.sh                          # shipped torchcomms (0.1.0)
qsub -v DLCOMM_TC_STACK=local jobscript_all_layers.sh     # torchcomms 0.3.0
```

Results are written to
`/lus/flare/projects/datascience/kaushik/DLcomm/validation/<timestamp>/`, one
directory per scale.

## Layers

Lowest to highest in the stack:

| Layer | Buffers | Path |
|---|---|---|
| libfabric (FI) | host/device | raw fabric API — the bottom |
| OSU / MPI | host | MPI collectives |
| C++ oneCCL | device | CCL called directly from C++ |
| C++ SYCL transfer | device | H2D / D2H / D2D copies |
| torch.distributed | device | XCCL through PyTorch |
| torchcomms | device | XCCL through the torchcomms API |
| NIXL | host/device | point-to-point RDMA — the top |

OSU uses host buffers while the other layers use device buffers, so OSU
columns are not same-path with the rest. The comparison tool reports the
difference rather than computing a ratio across it; see the cross-layer
comparison section of the top-level README.

## Where NIXL sits, and why it is at the top

NIXL is **above** torchcomms, not below libfabric. It is not a lower-level
transport that the collective stack is built on — it is a *separate consumer*
of the same fabric, used by inference serving (Dynamo/vLLM KV transfer) the
way a training job uses torchcomms.

Placing it below MPI would assert that the collectives run on top of it, which
is false, and would split the training path (`osu → ccl → torch → torchcomms`)
in half.

The practical consequence: a torchcomms → NIXL gap is labelled *"different
consumer of the fabric, not a subset"* instead of being attributed to an
overhead component, because the difference between them is not something
torchcomms lost.

## Fabric layers are opt-in

The FI and NIXL stages are off by default and enabled with `DLCOMM_FABRIC=1`:

```
qsub -v DLCOMM_FABRIC=1,DLCOMM_NIXL_REPRO=/path/to/repro_nixl_2rank_transfer.py \
     jobscript_all_layers.sh
```

They are not in the default path because Aurora has no CXI provider — running
them there would print "not measured" on every line and add nothing. A layer
that cannot apply is skipped **with a stated reason**, never emitted as a zero.

`DLCOMM_FI_PROV` (default `cxi`), `DLCOMM_NIXL_GIB` (0.25) and
`DLCOMM_NIXL_ITERS` (5) tune the stages.

## NIXL needs two nodes

At the 1-node scale the NIXL stage does not produce a number. The repro script
refuses a same-node transfer outright:

```
ERROR: need exactly 2 distinct nodes, saw 1: [...].
       a same-node transfer proves nothing about the fabric.
```

That refusal is correct — an intra-node copy never reaches the NIC, so a
"bandwidth" from it would measure memcpy and invite a false comparison against
the 2-node result. The 1-node scale therefore covers the collective layers and
the FI inventory, and records NIXL as not-applicable with the reason attached.

Measured on Tara North (job 6931), 2 nodes, 0.25 GiB READ, 5 iterations:

| Memory | best | mean | rails | byte-exact |
|---|---|---|---|---|
| DRAM | 13.18 GB/s | 7.97 GB/s | 4 (cxi0–3) | PASS |
| VRAM | 11.94 GB/s | 8.48 GB/s | 1 (cxi0) | PASS |

VRAM lights one rail because GPU→NIC affinity pins a device to its local NIC;
four GPUs driving traffic are needed to light all four. This is why the rail
count is printed next to the bandwidth rather than assumed.

## torchcomms stack selection

The torchcomms that ships with `frameworks/2025.3.1` is version 0.1.0. It
implements `all_reduce` and stubs the remaining operations with
`XCCL <op> is not supported now and will be added later`, so a default run
produces one torchcomms column.

`DLCOMM_TC_STACK=local` selects the 0.3.0 build, which implements 12 of 12
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
