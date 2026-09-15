# Fix 2 — Across-node bandwidth group size

**Status:** fixed; confirmed on hardware (job 8824234)

**Severity:** high — reported across-node bandwidth was wrong by a factor of `nodes / devices_per_node`
**Files:** `dl_comm/analysis/bandwidth.py`
**Tests:** `tests/test_bandwidth.py`

## The defect

`bandwidth.py` contained two implementations of the same calculation. The live
one derived the communicator size from `num_devices_per_node` for every mode,
including `across_node`. A second, unreachable implementation below it used
`num_compute_nodes`.

An `across_node` group spans one device index across *all* nodes, so its size
is the node count. The live path was wrong; the dead path was right.

## Magnitude

At the `examples/13_for_paper/B_nodes_1024` geometry — 1024 nodes, 4 devices
per node, 10 MB buffer, 10 ms elapsed:

| path | group size used | reported bandwidth |
|---|---|---|
| live (shipped) | 4 | 4.194304e+09 B/s |
| dead (unreachable) | 1024 | 1.073742e+12 B/s |

Ratio 256×, exactly `1024 / 4`. The error scales with the job: every published
across-node number is off by `num_compute_nodes / num_devices_per_node`, so it
grows as the scaling study grows.

## Secondary issue: neither formula was bus bandwidth

Both paths reported a quantity that is neither algorithmic nor bus bandwidth.
For the same geometry, algorithmic bandwidth is 1.048576e+09 B/s and allreduce
bus bandwidth is 2.095104e+09 B/s; the live value was 2.0016× busbw — close
enough to busbw to be mistaken for it in a plot, but not equal to it.

## The fix

`group_size_for(comm_mode, mode_cfg)` returns the communicator size from the
topology:

| mode | group size |
|---|---|
| `within_node` | `num_devices_per_node` |
| `across_node` | `num_compute_nodes` |
| `flatview` | `num_compute_nodes * num_devices_per_node` |

The dead code path is removed rather than left in place.

Both bandwidth conventions are now reported explicitly:

- `algbw = buffer_bytes / seconds`
- `busbw = algbw * factor(collective, n)`

using the standard NCCL factors, so numbers are comparable with
`nccl-tests` and `oneCCL` benchmarks:

| collective | factor |
|---|---|
| allreduce | `2(n-1)/n` |
| reduce, broadcast | `1` |
| allgather, reducescatter, alltoall | `(n-1)/n` |

## Verification

```
$ python -m pytest tests/test_bandwidth.py -q
```

`test_across_node_group_size_is_not_devices_per_node` asserts the corrected
group size is 1024 and that the ratio to the old value is exactly 256.0.
`test_busbw_factors_match_nccl_convention` pins each factor.

## Action required

This changes reported numbers. Any across-node result produced before this fix
is scaled by `num_compute_nodes / num_devices_per_node` and needs regenerating
before publication. Within-node and flatview numbers are unaffected in group
size, but the algbw/busbw split is new.
