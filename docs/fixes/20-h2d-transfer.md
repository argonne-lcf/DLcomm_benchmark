# Feature 20 — host-to-device transfer measurement

**Status:** implemented, unit-tested (8 tests); hardware run pending.
**Reference:** user-supplied `pci.cpp` (SYCL H2D/D2H/bidirectional benchmark).

## Why

DLcomm measures collectives, which move data device-to-device. The host-device
link is not exercised, so a degraded PCIe path is invisible to every existing
DLcomm metric while still slowing real training. Causes seen in practice:

* a rank bound to a CPU core on a NUMA node remote from its GPU,
* a link trained below full width or at a lower generation,
* pageable host memory forcing an extra staging copy.

Measuring the transfer directly turns an invisible ceiling into a number that
can be attributed to hardware.

## Method

Taken from the reference benchmark:

| Property | Reason |
| --- | --- |
| Shuffled iota fill | A constant or zero buffer can be compressed or served from a zero page, inflating apparent bandwidth. |
| Best of N iterations | The minimum time is the cleanest estimate of link capability; means are dragged by scheduler noise. |
| Barrier, then `min(start)`/`max(end)` across ranks | Reports the aggregate wall time of the slowest rank, not a lucky one. |
| Bidirectional issues both copies before waiting | H2D and D2H overlap, so the figure reflects the full-duplex link. |

Two deliberate additions:

* **Pinned and pageable host memory are both measured.** The gap between them
  is normally large; reporting only pageable misattributes a staging-buffer
  cost to the link itself.
* **Byte counts are computed in Python integers.** See the defect below.

## Defect found in the reference

`pci.cpp` computes the transferred byte count in C `int`:

```c
const int N = 1 << 28;                 // 268435456 elements
const int N_byte = N * sizeof(int);    // 1073741824 -- fits, just
const double H2D_bw = (N_byte * world_size) / H2D_time;
```

`N_byte * world_size` is evaluated as `int * int` **before** the conversion to
`double`. `N_byte` is 2^30, so the product overflows for any `world_size > 1`:

| world_size | true product | as int32 |
| ---: | ---: | ---: |
| 1 | 1073741824 | 1073741824 |
| 2 | 2147483648 | **-2147483648** |
| 4 | 4294967296 | **0** |
| 12 | 12884901888 | **0** |
| 24 | 25769803776 | **0** |

At every rank count that is a multiple of 4 the product is exactly zero, so the
benchmark reports **0 GB/s**. On a full Aurora node (12 ranks) or two nodes
(24), the reported bandwidth is always zero. At 2 ranks it is negative.

A secondary issue: `(N_byte * world_size) / H2D_time` divides an integer by an
unsigned long, so the division is integral and the fractional part is
discarded before the result reaches the `double`.

Fix in this implementation: all byte arithmetic uses Python integers, which are
arbitrary precision. `tests/test_h2d_transfer.py::test_byte_count_does_not_
overflow_at_scale` pins the 24-rank case and asserts the value that wraps to
zero in C stays exact here.

## Layer coverage

The point of the feature is comparing the same transfer across every layer, so
a limitation can be localised.

| Layer | Mechanism | Status |
| --- | --- | --- |
| SYCL (C++) | the reference `pci.cpp`, compiled with `icpx -fsycl` | available |
| PyTorch | `dl_comm.transfer.measure` via `tensor.copy_` | implemented |
| torchcomms | same path, torchcomms build | pending the 0.3.0 build |
| OSU | — | **not available**, see below |

**OSU cannot participate.** OSU 7.1's `configure --help` offers
`--enable-cuda`, `--enable-rocm`, and `--enable-openacc`; there is no SYCL or
Level Zero option, and no `sycl` string anywhere in its sources. Its `-d`
device-buffer mode therefore cannot allocate Intel GPU memory, so OSU measures
host-to-host only on Aurora. The comparison table renders `-` for OSU rather
than a host-to-host number relabelled as H2D.

## Usage

```python
from dl_comm.transfer import measure, format_table

results = measure(torch, device, nbytes=1 << 30, iterations=10,
                  pinned=True, dist=dist, world_size=world_size)
print(format_table(results))
```

## Tests

`tests/test_h2d_transfer.py` — 8 tests covering the overflow property, the
bidirectional byte count, best-vs-median ordering, bandwidth arithmetic, and
the error paths for missing and non-positive timings.
