# Finding 04 — torchcomms XCCL measured, and two caveats on the numbers

**Status:** measured (job 8825248), reproduced (job 8825296); one cell unstable

**Source:** job 8825248, `validation/allscales_20260914_044215`, torchcomms 0.3.0 on
torch 2.13.0a0+git3300461, XCCL backend, Aurora.

`TC_SUPPORTED=12/12` and `TORCHCOMMS_VERDICT=OK` at both 12 and 24 ranks. This is the
first run in which the torchcomms layer produced bandwidth records rather than only a
capability matrix.

## Measured bus bandwidth (GB/s)

| op | 1 MiB @12 | 2 MiB @12 | 4 MiB @12 | 1 MiB @24 | 2 MiB @24 | 4 MiB @24 |
|---|---|---|---|---|---|---|
| all_gather_single | 50.45 | 73.69 | 94.46 | 32.96 | 39.34 | 41.79 |
| reduce_scatter_single | 53.91 | 83.93 | 111.76 | 27.39 | 40.51 | unstable (see caveat 1) |
| all_reduce | 11.29 | 21.06 | 34.04 | 9.74 | 15.40 | 22.35 |
| reduce | 3.55 | 6.75 | 12.13 | 2.75 | 4.87 | 7.87 |
| broadcast | 5.58 | 8.17 | 10.59 | 3.80 | 4.87 | 5.81 |
| all_to_all_single | 1.89 | 3.54 | 6.47 | 0.63 | 1.11 | 1.82 |

Figures are from job 8825248 and were reproduced by job 8825296 to within 0.3 % at 12
ranks and 3.1 % at 24 ranks, with the single exception noted in caveat 1.

Bus-bandwidth factors follow the nccl-tests convention and were cross-checked against
`dl_comm.analysis.bandwidth.busbw_factor` for every op at both scales: 0 mismatches.
The reported figures were recomputed independently from the raw `T_MED` values and
reproduce exactly (for example all_to_all_single at 12 ranks: 1048560 B / 508.706 µs =
2.06 GB/s algbw, × 11/12 = 1.89 GB/s busbw).

## Caveat 1 — reduce_scatter_single at 24 ranks is non-monotonic (reproduced)

Bus bandwidth does not rise with message size at 24 ranks. The shape was confirmed by
an independent repeat (job 8825296, `allscales_20260914_050129`):

| cell | run 8825248 | run 8825296 | spread |
|---|---|---|---|
| reduce_scatter 24r 1 MiB | 27.39 | 27.98 | 2.1 % |
| reduce_scatter 24r 2 MiB | 40.51 | 40.07 | 1.1 % |
| reduce_scatter 24r 4 MiB | 28.36 | 34.94 | **20.8 %** |

Both runs fall at 4 MiB after peaking at 2 MiB (40.5 → 28.4 and 40.1 → 34.9), so the
non-monotonic shape is reproducible and not a single-run artefact. The magnitude is
not: the 4 MiB cell differs by 20.8 % between runs, while every other cell in the table
agrees to within 3.1 % and all six 12-rank cells agree to within 0.3 %.

The instability is specific to this one cell. allgather at the same scale and sizes is
monotonic in both runs (33.0 → 39.3 → 41.8 and 32.0 → 38.6 → 41.3), and
reduce_scatter at 12 ranks is monotonic in both (53.9 → 83.9 → 111.8 and
54.1 → 83.8 → 112.1).

**Reportable conclusion:** reduce_scatter_single at 24 ranks degrades at 4 MiB instead
of scaling. **Not reportable:** any specific bandwidth figure for that cell — a median
of 20 iterations does not stabilise it, so a point value would be misleading. The other
17 cells in the measured table are reproducible and may be quoted.

The cause is open. The cell sits exactly where the message crosses 4 MiB at two nodes,
which is also where `docs/findings/01` records an OSU allgather knee, but no common
mechanism has been demonstrated.

## Caveat 2 — the torchcomms layer runs a different torch build

The torchcomms layer runs torch 2.13.0a0+git3300461, because torchcomms 0.3.0 must be
paired with the torch it was built against (see `docs/fixes/26`). Every other layer
runs the frameworks module's torch 2.10.0a0+git449b176. A torchcomms-versus-
torch.distributed difference therefore carries a torch-version difference inside it.

An empirical bound on that confound is available from allgather at 24 ranks, where both
layers measured the same operation at the same scale:

| message | torchcomms (torch 2.13) | torch.distributed (torch 2.10) | difference |
|---|---|---|---|
| 1 MiB | 0.731 ms | 0.762 ms | 4.0 % |
| 4 MiB | 2.309 ms | 2.337 ms | 1.2 % |

For allgather the two stacks agree to within 4 %, so the torch version is not
responsible for large gaps in that operation. This bound has only been established for
allgather; it should not be assumed for alltoall or reduce_scatter without the same
check.

## Cross-layer note

all_to_all_single shows the node-boundary collapse already documented in
`docs/findings/02` for C++ oneCCL, torch.distributed and OSU. At 4 MiB it falls from
6.47 GB/s at 12 ranks to 1.82 GB/s at 24, a 3.6× drop, while all_reduce over the same
transition falls only 1.5×. torchcomms is the fourth independent layer to show the
pattern.
