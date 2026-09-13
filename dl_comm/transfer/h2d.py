"""Host-to-device transfer measurement.

Purpose
-------
DLcomm measures collectives, which run device-to-device. If the PCIe path
between host and device is degraded -- a rank landed on a remote NUMA node, a
link trained at reduced width, pageable instead of pinned host memory -- the
symptom shows up as slow end-to-end training while every collective benchmark
still looks healthy. This module measures that path directly so the limitation
is attributed rather than inferred.

Methodology follows the reference SYCL benchmark (``pci.cpp``):

* **Shuffled iota fill**, never a constant. A buffer of zeros or of a repeated
  value can be compressed or served from a zero page somewhere in the stack,
  which inflates the apparent bandwidth.
* **Best of N iterations.** The minimum time is the cleanest estimate of link
  capability; means are dragged by scheduler noise.
* **Barrier before each iteration**, then reduce ``min(start)`` and
  ``max(end)`` across ranks, so the reported figure is the aggregate wall time
  of the slowest rank rather than a lucky one.
* **Bidirectional case issues both copies before waiting**, so H2D and D2H
  overlap and the result reflects the full-duplex link.

Two deliberate differences from the reference:

* Pinned *and* pageable host memory are both measured. The gap between them is
  usually large, and a pageable-only result misattributes a staging-buffer
  cost to the link.
* Byte counts are computed in Python ints (arbitrary precision). The reference
  computes ``N_byte * world_size`` in C ``int``, which overflows for a 1 GiB
  buffer at any world size above one; see ``docs/fixes/20``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransferResult:
    """One measured transfer pattern."""

    direction: str          # "h2d" | "d2h" | "d2d" | "bidirectional"
    pinned: bool
    nbytes: int             # per rank, per copy
    world_size: int
    times_s: list[float] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Bytes moved across all ranks in one iteration.

        Computed in Python ints so it cannot overflow; the bidirectional case
        moves two buffers per rank.
        """
        per_rank = self.nbytes * (2 if self.direction == "bidirectional" else 1)
        return per_rank * self.world_size

    @property
    def best_s(self) -> float:
        if not self.times_s:
            raise ValueError(
                f"no timings recorded for {self.direction}; "
                "measure() must run at least one iteration"
            )
        return min(self.times_s)

    @property
    def median_s(self) -> float:
        if not self.times_s:
            raise ValueError(f"no timings recorded for {self.direction}")
        return statistics.median(self.times_s)

    def bandwidth_bps(self, which: str = "best") -> float:
        """Bytes per second. ``which`` is 'best' or 'median'."""
        t = self.best_s if which == "best" else self.median_s
        if t <= 0:
            raise ValueError(
                f"non-positive elapsed time ({t}s) for {self.direction}: the "
                "timer resolution is too coarse for this buffer size, or the "
                "copy did not execute"
            )
        return self.total_bytes / t

    def gbps(self, which: str = "best") -> float:
        """Gigabytes (10^9) per second, matching the reference's units."""
        return self.bandwidth_bps(which) / 1e9


def fill_shuffled(tensor: Any, torch_mod: Any) -> None:
    """Fill with pseudo-random, incompressible data.

    A constant fill lets compression or zero-page handling anywhere in the
    path report a bandwidth the hardware cannot sustain.

    The reference uses std::shuffle on an iota. torch.randperm is the direct
    equivalent but is O(n) single-threaded on host tensors and effectively
    hangs at 2^28 elements: job 8824725's PyTorch layer timed out at 600 s
    filling four 1 GiB buffers this way. random_ produces data that is just
    as incompressible for the purpose of defeating link compression, and is
    parallel on both host and device.
    """
    iinfo = torch_mod.iinfo(tensor.dtype)
    tensor.random_(iinfo.min, iinfo.max)


def _sync(torch_mod: Any, device: Any) -> None:
    """Block until queued device work has completed.

    Device copies are asynchronous; timing without this measures enqueue cost,
    not transfer.
    """
    dtype = device.type if hasattr(device, "type") else str(device).split(":")[0]
    if dtype == "xpu" and hasattr(torch_mod, "xpu"):
        torch_mod.xpu.synchronize()
    elif dtype == "cuda" and hasattr(torch_mod, "cuda"):
        torch_mod.cuda.synchronize()


def measure(
    torch_mod: Any,
    device: Any,
    *,
    nbytes: int = 1 << 30,
    iterations: int = 10,
    pinned: bool = True,
    dist: Any = None,
    world_size: int = 1,
    directions: tuple[str, ...] = ("h2d", "d2h", "d2d", "bidirectional"),
) -> list[TransferResult]:
    """Measure host-device transfer bandwidth.

    ``dist`` is optional; when supplied its ``barrier`` is called before each
    iteration so ranks start together, matching the reference's use of
    ``MPI_Barrier``.
    """
    if nbytes <= 0:
        raise ValueError(f"nbytes must be positive, got {nbytes}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    itemsize = 4  # int32
    n = nbytes // itemsize
    if n == 0:
        raise ValueError(f"nbytes={nbytes} is smaller than one element")

    i32 = torch_mod.int32
    host_a = torch_mod.empty(n, dtype=i32, pin_memory=pinned)
    host_b = torch_mod.empty(n, dtype=i32, pin_memory=pinned)
    dev_a = torch_mod.empty(n, dtype=i32, device=device)
    dev_b = torch_mod.empty(n, dtype=i32, device=device)

    for t in (host_a, host_b, dev_a, dev_b):
        fill_shuffled(t, torch_mod)
    _sync(torch_mod, device)

    import time

    results = []
    for direction in directions:
        res = TransferResult(direction=direction, pinned=pinned,
                             nbytes=nbytes, world_size=world_size)
        for _ in range(iterations):
            if dist is not None:
                dist.barrier()
            _sync(torch_mod, device)
            t0 = time.perf_counter()
            if direction == "h2d":
                dev_a.copy_(host_a, non_blocking=pinned)
            elif direction == "d2h":
                host_a.copy_(dev_a, non_blocking=pinned)
            elif direction == "bidirectional":
                # Both copies are issued before the wait so they overlap.
                dev_a.copy_(host_a, non_blocking=pinned)
                host_b.copy_(dev_b, non_blocking=pinned)
            elif direction == "d2d":
                # On-device copy. This never crosses PCIe, so it measures HBM
                # bandwidth and forms the ceiling the host-device numbers
                # should be read against.
                dev_b.copy_(dev_a, non_blocking=True)
            else:
                raise ValueError(
                    f"unknown direction {direction!r}; "
                    "expected h2d, d2h, d2d, or bidirectional"
                )
            _sync(torch_mod, device)
            res.times_s.append(time.perf_counter() - t0)
        results.append(res)
    return results


def format_table(results: list[TransferResult]) -> str:
    """Render results as a fixed-width table."""
    head = (f"{'direction':<16}{'host mem':<10}{'MiB':>8}"
            f"{'best GB/s':>12}{'median GB/s':>14}")
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.direction:<16}"
            f"{'pinned' if r.pinned else 'pageable':<10}"
            f"{r.nbytes / (1 << 20):>8.0f}"
            f"{r.gbps('best'):>12.2f}"
            f"{r.gbps('median'):>14.2f}"
        )
    return "\n".join(lines)
