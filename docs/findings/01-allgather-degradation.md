# Finding 01 — allgather degradation was a benchmark defect, not a bottleneck

**Status:** resolved. The reported effect does not exist on the hardware.

## Original observation

Across five jobs and two independent stacks, allgather appeared to be the
slowest collective on Aurora by two orders of magnitude, and it appeared to
get *worse* as the buffer grew while every other collective improved.

| per-rank buffer | C++ oneCCL busbw (as reported) |
|---|---|
| 1 MiB | 0.172 GB/s |
| 2 MiB | 0.170 GB/s |
| 4 MiB | 0.157 GB/s |

OSU/MPI showed a matching knee: allgather latency rose 9 496 us -> 72 108 us
between 2 MiB and 4 MiB, a 7.6x jump for a 2x size increase.

Because two unrelated stacks agreed, the effect was initially treated as a
real property of the machine.

## Root cause

Two independent defects in this benchmark, both in the measuring code.

### 1. All ranks used the same GPU tile

`dl_comm/ccl/ccl_bench.cpp` selected its device with:

```cpp
sycl::queue q{sycl::gpu_selector_v};
```

and `dl_comm/transfer/pci_fixed.cpp` used a default-constructed `sycl::queue`.

Aurora runs with `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, so each of the 12 tiles
enumerates as a separate root device. Both forms return the *same* device on
every rank, so all 12 ranks on a node drove tile 0 while 11 tiles sat idle.

allgather is the collective most sensitive to this: its output grows with
rank count, so forcing all ranks through one tile's memory and one PCIe path
degrades it further as the buffer grows. That produced the negative slope.

The OSU knee is a separate, genuine MPI-level effect and is not explained by
this defect; it is recorded below as still open.

### 2. The busbw numerator used the per-rank slice

allgather produces `buffer * world` bytes of output, but the algbw numerator
used the per-rank buffer, understating allgather by exactly the rank count
(12x at 12 ranks). This was raised in review by A-Bot-CELS.

Correcting the numerator alone did not remove the negative slope — it only
changed the magnitude. Both fixes were required.

## Evidence after the fix

Job 8824968, both scales, with `TILE_CHECK=PASS` asserting that the number of
distinct `(host, device)` pairs equals the rank count:

| per-rank buffer | 12 ranks | 24 ranks |
|---|---|---|
| 1 MiB | 43.5 GB/s | 32.6 GB/s |
| 2 MiB | 65.9 GB/s | 41.7 GB/s |
| 4 MiB | 88.5 GB/s | 46.8 GB/s |

allgather now increases with buffer size at both scales, which is the
expected shape. It is no longer an outlier.

## Guards added

- `TILE_CHECK` in `run_all_scales.sh` fails the run unless distinct
  `(host, device)` pairs equal the rank count.
- A build-time marker assertion refuses to run a stale source tree.
- `test_no_collective_exceeds_hardware_ceiling` fails any conversion
  reporting a bandwidth above an aggregate hardware ceiling.

## Corrections issued

Five ratios derived from the pre-fix measurements were retracted publicly.
Every C++ measurement taken before job 8824968 is void.

## Still open

- The OSU 2 MiB -> 4 MiB allgather knee was measured through MPI host
  buffers and is unaffected by the SYCL device-selection defect. It has not
  been re-measured since and remains uncharacterised.
- `alltoall` and `broadcast` lose 4.4x and 6.1x respectively crossing the
  1-node to 2-node boundary while `sendrecv` stays flat (25.9 -> 26.2 GB/s).
  This is a scale-out effect in corrected data and is the next candidate for
  investigation.
