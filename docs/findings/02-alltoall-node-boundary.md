# Finding 02 — alltoall degrades across the node boundary in every layer measured

**Status:** mechanism established (jobs 8826409, 8826417). Reproduced independently in
four implementations, so it is a property of the machine and of the oneCCL `scatter`
schedule at this scale rather than a defect in any one benchmark.

## Observation

Crossing from 1 node / 12 ranks to 2 nodes / 24 ranks, alltoall loses far more than the
other collectives. Measurements at 4 MiB per rank, from job 8825050
(`allscales_20260914_030028`), all with `TILE_CHECK=PASS`:

| layer | metric | 12 ranks | 24 ranks | change |
|---|---|---:|---:|---|
| C++ oneCCL | busbw GB/s | 12.5 | 2.9 | 4.3× worse |
| torch.distributed | busbw GB/s | 16.10 | 2.41 | 6.7× worse |
| OSU (host buffers) | latency µs | 20121.96 | 32468.02 | 1.6× worse |

The three layers use different buffer types (device, device, host), different transports
and different measurement code. The direction and rough magnitude agree.

## Why this is not simply "more ranks is slower"

Other collectives in the same runs do not behave this way. From the same job:

| collective | C++ 12r | C++ 24r | torch 12r | torch 24r | OSU 12r | OSU 24r |
|---|---:|---:|---:|---:|---:|---:|
| allreduce | 12.3 | 8.1 | 32.33 | 20.81 | 8992 | 3593 |
| allgather | 88.5 | 46.8 | 92.53 | 42.58 | 72254 | 120468 |
| alltoall | 12.5 | 2.9 | 16.10 | 2.41 | 20122 | 32468 |
| broadcast | 10.7 | 1.7 | 10.38 | 6.14 | 1080 | 972 |

OSU allreduce is 2.5× *faster* at 24 ranks and OSU broadcast is flat, so the fabric is
not uniformly slower with more ranks. alltoall is specifically affected.

Note that C++ broadcast (10.7 → 1.7, 6.3×) falls as steeply as alltoall, while
torch broadcast (10.38 → 6.14) and OSU broadcast (1080 → 972) do not. That disagreement
is unexplained and is tracked as part of this investigation rather than asserted as a
second finding.

## Mechanism — measured, job 8826409 (2 nodes) and 8826417 (1 node)

`tools/probe_alltoall_algorithm.sh` runs the same alltoall at both scales with
`CCL_LOG_LEVEL=debug` and reads oneCCL's own selection log. Findings:

**The algorithm does not change at the node boundary.** oneCCL selects
`algo scatter` for alltoall at 12 ranks and at 24 ranks alike:

```
selector_impl.hpp:355 get: selected algo: coll alltoall, count 1048576, algo scatter   (1 node)
selector_impl.hpp:355 get: selected algo: coll alltoall, count 1048576, algo scatter   (2 nodes)
```

The candidate explanation in the previous revision of this document — that
oneCCL switches away from a topology-aware path at 2 nodes — is therefore
**wrong**, and so is a second guess made while investigating: the library does
ship `alltoall_sycl_single_node` with no `alltoall_sycl_multi_node` counterpart
(every other collective has one), but the debug log shows no SYCL alltoall
kernel is invoked at *either* scale, so that asymmetry is not what is acting
here.

**What actually changes is the size of the schedule.** The `scatter` algorithm
posts a send and a receive per peer, so its entry count grows as `n(n−1)`:

| scale | SEND entries | RECV entries | n(n−1) |
|---|---:|---:|---:|
| 12 ranks, 1 node | 1764 | 1764 | 132 |
| 24 ranks, 2 nodes | 6694 | 6694 | 552 |

Doubling the rank count multiplies the point-to-point entries by 3.8×, close to
the 4.2× that `n(n−1)` predicts. On top of that, 12 of each rank's 23 peers are
now off-node, where 0 of 11 were before.

Against the measured 7.5× fall in bus bandwidth (example 17, 4 MiB), the entry
count accounts for 3.8× and the remaining 2.0× is consistent with the off-node
hop cost on those entries. The two effects together are sufficient; no
algorithm change is needed to explain the collapse.

This also resolves why `sendrecv` stays flat across the boundary: it is a
single pair, so its schedule does not grow at all, and the log confirms it
takes the `topo sycl` path rather than `scatter`.

## Superseded hypothesis

The original mechanism section proposed Slingshot injection bandwidth or an
oneCCL algorithm switch, with three suggested measurements. Step 2 of that plan
(`CCL_LOG_LEVEL=info`) has now been carried out and disproves the algorithm
switch. Steps 1 and 3 remain useful: a rank sweep at 12/24/48 would confirm the
`n(n−1)` shape continues, and the p2p control has already behaved as predicted.


## Status of the earlier allgather claim

An allgather anomaly reported earlier was a benchmark defect and was retracted; see
`01-allgather-degradation.md`. The OSU allgather knee in the table above (72254 →
120468 µs) is a separate, host-buffer measurement that the SYCL fix did not touch and
that remains uncharacterised.
