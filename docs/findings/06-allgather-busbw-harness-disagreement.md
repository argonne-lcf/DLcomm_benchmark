# Finding 06 — allgather and reduce_scatter bandwidth differs 12× between two DLcomm harnesses

**Status:** open. Both harnesses are self-consistent; they disagree on what the
`bytes` term means. No numbers from the two should be placed in the same table
until this is settled.

## Observation

Example 17 and `tools/probe_tc03.py` measure the same operation, on the same
stack, at the same scale, and report bandwidths that differ by roughly the
group size:

| source | op | scale | busbw |
|---|---|---|---:|
| `tools/probe_tc03.py` (finding 04) | all_gather_single, 4 MiB | 12 ranks | 94.46 GB/s |
| example 17 (job 8826362) | allgather, 4 MiB | 12 ranks | 3.96 GB/s |

The ratio is 24×, of which exactly 12× — the group size — is a convention
difference and the remainder is a harness difference.

## Cause

Both compute `algbw = bytes / t` then multiply by the same nccl-tests factor
`(n−1)/n`, verified identical in `dl_comm.analysis.bandwidth.busbw_factor` and
`probe_tc03._busbw_factor`. They disagree on `bytes`:

- `probe_tc03._bench` passes `moved_multiplier=world` for
  `all_gather_single` and `reduce_scatter_single`, so `bytes` is the **output**
  buffer, `4 MiB × 12 = 48 MiB`.
- DLcomm's `algorithmic_bandwidth(buffer_size, t)` uses the configured
  per-rank buffer, `4 MiB`.

Recomputing from the measured median confirms the arithmetic, with no residual:

```
t_med = 0.000971 s   (example 17, allgather, 4 MiB, 12 ranks)
DLcomm : 4 MiB      / t × 11/12 =  3.96 GB/s   (reported 3.958)
probe  : 4 MiB × 12 / t × 11/12 = 47.52 GB/s
```

`47.52` against the probe's own `94.46` leaves a further 2.0× that is *not* a
convention difference. The two harnesses do different work per iteration:
`probe_tc03._bench` calls one collective in a tight loop on a preallocated
buffer, whereas example 17 rebuilds a rank-dependent payload each iteration
(`build_payload`, required by fix 01) and brackets the collective with two
`time_barrier` calls. The barriers are outside the timed region (fix 05) and
the correctness check is outside it as well, so this is not timer
contamination — but the device state differs between the two loops.

## Which convention is right

nccl-tests defines allgather `algbw` on the **per-rank** `sendcount`, not on
the gathered output, which is DLcomm's convention. The reference factor table
in `docs/findings/03` was checked against DLcomm and agrees for every op; it
does not cover the `bytes` term, which is where this disagreement lives.

On that reading `probe_tc03` overstates allgather and reduce_scatter by the
group size — 12× at one node, 24× at two. That would make the finding-04
figures of 94.46 GB/s for allgather and 111.76 GB/s for reduce_scatter
inflated, and it explains why finding 04's reduce_scatter exceeded allgather
at the same scale, which doc 21 already flagged as a duality violation when
the C++ layer did the same thing.

This has not yet been confirmed against the upstream nccl-tests source, so the
conclusion is stated as the likely reading rather than as established.

## Consequence for published numbers

Until this is resolved:

- The finding-04 table is **suspect for `all_gather_single` and
  `reduce_scatter_single` only**. Its other four operations pass `bytes`
  without a multiplier and are unaffected.
- Example 17's numbers (jobs 8826362, 8826378) use DLcomm's own convention
  consistently and are internally comparable across scales.
- Cross-layer comparisons in the other findings all come from the DLcomm
  harness, so they are not affected.

## Next step

Compare both harnesses against the vendor reference at one point — allgather,
4 MiB, 12 ranks — and adopt whichever convention the reference uses, then
correct or annotate finding 04.
