# Fix 25 — torch.distributed green at 12 and 24 ranks

**Status:** verified by job 8825050 (`TORCH_RUN_EXIT=0` at both scales, `TILE_CHECK=PASS`).

The torch.distributed layer had failed in every prior job. Two independent defects were
responsible, plus one latent defect found while fixing them.

## Defect 1 — launcher did not request PMIx

`mpiexec` was invoked without `--pmi=pmix`, so PALS used its own PMI while
`CCL_PROCESS_LAUNCHER=pmix` told oneCCL to expect a PMIx namespace. The two disagreed
during bootstrap.

```diff
-"$MPIEXEC" -n "$nranks" -ppn "$ppn" $hostarg
+"$MPIEXEC" --pmi=pmix --envall -n "$nranks" -ppn "$ppn" $hostarg
```

Applied to all five launch sites. This matches the working Aurora reference launchers
under `datascience_collab/pshukla`.

## Defect 2 — alltoall tensor not divisible by the group size

```
RuntimeError: xpu_alltoall_base: tensor's dim 0 does not divide equally across group size
```

The benchmark derives `count = nbytes / 4`, a power of two. That divides by 12 but not
by 24, so the 1-node scale passed and the 2-node scale failed. The C++ layer already
trimmed; the torch stage did not.

```python
count_div = (count // world) * world
```

The trimmed count is reported in the output record, so the measured bytes match what was
actually moved rather than what was requested.

## Defect 3 (latent) — device selected by global rank

`probe_tc03.py` selected its device with `rank % torch.xpu.device_count()`. This is the
same global-versus-local defect fixed earlier in `ccl_bench.cpp` and `pci_fixed.cpp`.
It is correct on one node and wrong on two: ranks 12–23 must map back onto devices 0–11
of the second node.

`TILE_CHECK` cannot catch this. It asserts that the `(host, device)` pairs are distinct,
and they remain distinct under the wrong mapping — only the assignment is wrong. The fix
indexes `LOCAL_RANK` (falling back to `PALS_LOCAL_RANKID`) and asserts the value lies
within the visible device count.

## Measured result

torch.distributed busbw, GB/s, 4 MiB per rank:

| collective | 12 ranks | 24 ranks |
|---|---:|---:|
| allgather | 92.53 | 42.58 |
| allreduce | 32.33 | 20.81 |
| alltoall | 16.10 | 2.41 |
| reduce | 11.51 | 7.57 |
| broadcast | 10.38 | 6.14 |

All five collectives increase monotonically with message size at both scales.

## Note on a change that was reverted

An earlier revision set `LOCAL_WORLD_SIZE`, on the basis that one reference file exported
it. Inspecting the extension shows it is never read:

```
$ strings _comms_xccl.cpython-312-x86_64-linux-gnu.so | grep -E '^(LOCAL_|PALS_|PMI_|MASTER_)'
MASTER_ADDR
MASTER_PORT
```

The variable was removed. A change with no mechanism should not be carried, because if
the job then passes for an unrelated reason the false attribution is locked in.
