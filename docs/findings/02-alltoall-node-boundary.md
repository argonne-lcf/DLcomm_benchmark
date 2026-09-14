# Finding 02 — alltoall degrades across the node boundary in all three layers

**Status:** open. Reproduced independently in three implementations, so it is a property
of the machine at this scale rather than a defect in any one benchmark.

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

## Mechanism, not yet established

alltoall moves `(world − 1)/world` of each rank's buffer off-node once a second node is
involved, whereas allreduce and broadcast are tree- or ring-structured and can exploit
intra-node links for most of their traffic. The candidate explanation is therefore
Slingshot injection bandwidth per node, or the oneCCL algorithm switching away from a
topology-aware path at 2 nodes.

This has not been measured. Confirming it requires:

1. A rank-count sweep at fixed per-rank bytes (12, 24, 48) to see whether the loss is a
   one-time step at the node boundary or continues with scale.
2. `CCL_LOG_LEVEL=info` to record which algorithm oneCCL selects at each scale.
3. The p2p layer as a control: `sendrecv` was flat across the boundary (25.9 → 26.2 GB/s
   in C++), which already suggests the raw off-node path is not itself degraded.

## Status of the earlier allgather claim

An allgather anomaly reported earlier was a benchmark defect and was retracted; see
`01-allgather-degradation.md`. The OSU allgather knee in the table above (72254 →
120468 µs) is a separate, host-buffer measurement that the SYCL fix did not touch and
that remains uncharacterised.
