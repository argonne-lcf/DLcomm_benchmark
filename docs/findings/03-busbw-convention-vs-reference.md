# Finding 03 — busbw convention differs from the Aurora reference harness for reduce and broadcast

**Status:** resolved in favour of the nccl-tests convention. No code change made. Recorded
because the two sources disagree and any comparison against reference numbers must account
for it.

## The disagreement

The reference harness under `torch_and_comm_reference` applies `(n-1)/n` to every
collective except allreduce:

```python
# reduce_perf.py
# Reduce: (n-1) / n * size / time
return algo_bw * (num_ranks - 1) / num_ranks / 1000.0

# broadcast_perf.py
# Broadcast: (n-1) / n * size / time
return algo_bw * (num_ranks - 1) / num_ranks / 1000.0
```

`dl_comm/analysis/bandwidth.py` applies `1.0` to both:

```python
if name in ("reduce", "broadcast", "bcast"):
    return 1.0
```

## Which is correct

nccl-tests `doc/PERFORMANCE.md` states the correction factors explicitly:

| collective | factor |
|---|---|
| AllReduce | 2(n−1)/n |
| ReduceScatter | (n−1)/n |
| AllGather | (n−1)/n |
| AllToAll | (n−1)/n |
| **Broadcast** | **1** |
| **Reduce** | **1** |

The reasoning is that broadcast and reduce are bottlenecked at the root rank, which has
only its own link bandwidth `B` available: all data must leave (or enter) that one rank,
so `t = S/B` and `busbw = algbw`. There is no `(n-1)/n` discount because the transfer
cannot be spread across all `n` links.

The benchmark's existing implementation matches nccl-tests and is left unchanged.

## Consequence for comparisons

Reference reduce and broadcast numbers are lower than this benchmark's by a factor of
`(n-1)/n` for the same underlying performance:

- 12 ranks: reference reads 0.917× of the value reported here
- 24 ranks: reference reads 0.958× of the value reported here

This is small enough to be mistaken for run-to-run variation, which is precisely why it is
recorded. When comparing a reduce or broadcast figure against the reference harness, divide
the value reported here by `(n-1)/n` first, or the two will appear to disagree by a few
percent for no visible reason.

Collectives unaffected: allreduce, allgather, reduce_scatter, alltoall, gather, scatter —
the factors agree.

## Verification

A unit test pins the factors against the nccl-tests table so that a future edit cannot
silently adopt the other convention; see `tests/test_bandwidth.py`.
