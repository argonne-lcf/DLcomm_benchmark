"""Cross-layer bottleneck analysis.

The benchmark measures the same collective at several layers of the stack:

    OSU (MPI)  ->  C++ CCL  ->  torch.distributed  ->  torchcomms

Each layer adds something on top of the one below it. Listing four bandwidth
numbers side by side does not say where time goes; this module derives the
gaps between adjacent layers and names the component responsible for each.

Two rules are enforced throughout:

1. Layers are only compared when they measured the same thing. A device-buffer
   result and a host-buffer result are different measurements, and dividing
   one by the other produces a number that looks like a speedup and is not.
   This is not hypothetical -- an earlier revision of the OSU comparison
   reported DLcomm as 78x faster than MPI for allgather purely because the
   two sides used different memory.

2. A missing layer is reported as missing. It is never skipped silently and
   never interpolated from its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Stack order, lowest to highest. Each layer is expected to be no faster than
# the one below it; a violation is reported rather than hidden.
#
# libfabric (fi) is the bottom: the raw fabric API every transport above it
# eventually calls.
#
# NIXL sits at the TOP, above torchcomms. It is not a lower-level transport
# that the collective stack is built on -- it is a separate consumer of the
# fabric, used by inference serving (Dynamo/vLLM KV transfer) the way a
# training job uses torchcomms. Placing it above torchcomms reflects how it is
# used, and keeps the training path (osu -> ccl -> torch -> torchcomms) as one
# unbroken chain instead of splitting it with an unrelated layer.
#
# NIXL is also point-to-point rather than collective, so in practice it shares
# no collective name with the layers below and no ratio is computed against
# them; the ordering matters for report layout and for the fi -> NIXL gap.
LAYER_ORDER = ("fi", "osu", "cpp_ccl", "torch_dist", "torchcomms", "nixl")

LAYER_LABEL = {
    "fi": "libfabric (FI)",
    "osu": "OSU / MPI",
    "cpp_ccl": "C++ CCL",
    "torch_dist": "torch.distributed",
    "torchcomms": "torchcomms",
    "nixl": "NIXL",
}

# What sits between two adjacent layers. Used to attribute a gap to a named
# component instead of reporting a bare percentage.
BETWEEN = {
    ("fi", "osu"): "MPI transport over the fabric",
    ("fi", "nixl"): "NIXL agent + memory registration",
    ("osu", "cpp_ccl"): "transport choice (MPI vs CCL)",
    ("cpp_ccl", "torch_dist"): "PyTorch dispatch + process group",
    ("torch_dist", "torchcomms"): "torchcomms API layer",
    ("torchcomms", "nixl"): "different consumer of the fabric, not a subset",
    ("osu", "torch_dist"): "MPI->PyTorch, CCL layer not measured",
    ("cpp_ccl", "torchcomms"): "PyTorch stack, torch.distributed not measured",
    ("osu", "torchcomms"): "MPI->torchcomms, intermediate layers not measured",
}


@dataclass(frozen=True)
class LayerMeasurement:
    """One layer's result for one collective at one size.

    ``buffer`` records where the payload lived ("device" or "host"). It is a
    required part of the identity of a measurement, not an annotation.
    """

    layer: str
    collective: str
    size_bytes: int
    busbw_bps: float | None
    buffer: str = "device"
    ranks: int = 0
    note: str = ""

    @property
    def available(self) -> bool:
        return self.busbw_bps is not None and self.busbw_bps > 0


@dataclass
class Gap:
    """The difference between two adjacent layers, with attribution."""

    lower: str
    upper: str
    collective: str
    size_bytes: int
    lower_bps: float
    upper_bps: float
    component: str

    @property
    def efficiency(self) -> float:
        """Fraction of the lower layer's bandwidth the upper layer retains."""
        return self.upper_bps / self.lower_bps

    @property
    def overhead_pct(self) -> float:
        """Percentage of the lower layer's bandwidth lost at the upper layer."""
        return (1.0 - self.efficiency) * 100.0

    @property
    def inverted(self) -> bool:
        """True when the upper layer beat the layer below it.

        Usually means the two measurements are not comparable, or that the
        lower layer is misconfigured. Reported, never silently dropped.
        """
        return self.upper_bps > self.lower_bps


@dataclass
class BottleneckReport:
    collective: str
    size_bytes: int
    measurements: dict[str, LayerMeasurement] = field(default_factory=dict)
    gaps: list[Gap] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    incomparable: list[str] = field(default_factory=list)

    @property
    def dominant(self) -> Gap | None:
        """The adjacent-layer gap that loses the most bandwidth."""
        real = [g for g in self.gaps if not g.inverted]
        return max(real, key=lambda g: g.overhead_pct) if real else None


def analyse(measurements: list[LayerMeasurement]) -> BottleneckReport:
    """Derive per-layer gaps for one collective at one size."""
    if not measurements:
        raise ValueError("no measurements supplied")

    collectives = {m.collective for m in measurements}
    sizes = {m.size_bytes for m in measurements}
    if len(collectives) != 1 or len(sizes) != 1:
        raise ValueError(
            "analyse() compares one collective at one size; got "
            f"collectives={sorted(collectives)} sizes={sorted(sizes)}"
        )

    report = BottleneckReport(collective=measurements[0].collective,
                              size_bytes=measurements[0].size_bytes)
    for m in measurements:
        report.measurements[m.layer] = m

    for layer in LAYER_ORDER:
        m = report.measurements.get(layer)
        if m is None:
            report.missing.append(f"{LAYER_LABEL[layer]}: not measured")
        elif not m.available:
            reason = m.note or "no result"
            report.missing.append(f"{LAYER_LABEL[layer]}: {reason}")

    usable = [layer for layer in LAYER_ORDER
              if (m := report.measurements.get(layer)) is not None and m.available]

    for i in range(len(usable) - 1):
        lo, hi = usable[i], usable[i + 1]
        m_lo, m_hi = report.measurements[lo], report.measurements[hi]

        if m_lo.buffer != m_hi.buffer:
            report.incomparable.append(
                f"{LAYER_LABEL[lo]} ({m_lo.buffer}) vs {LAYER_LABEL[hi]} "
                f"({m_hi.buffer}): different memory, no ratio computed"
            )
            continue

        lo_bps, hi_bps = m_lo.busbw_bps, m_hi.busbw_bps
        # Both are non-None here: `usable` filtered on .available. Asserting
        # keeps that guarantee checkable rather than implied.
        assert lo_bps is not None and hi_bps is not None

        report.gaps.append(Gap(
            lower=lo, upper=hi,
            collective=report.collective, size_bytes=report.size_bytes,
            lower_bps=lo_bps, upper_bps=hi_bps,
            component=BETWEEN.get((lo, hi), "unattributed"),
        ))

    return report


def format_report(report: BottleneckReport) -> str:
    """Render one collective's cross-layer analysis."""
    mib = report.size_bytes / (1 << 20)
    lines = [
        f"{report.collective}  @ {mib:.0f} MiB",
        "-" * 66,
    ]

    lines.append(f"{'layer':<20}{'buffer':<10}{'busbw GB/s':>12}")
    for layer in LAYER_ORDER:
        m = report.measurements.get(layer)
        if m is None:
            lines.append(f"{LAYER_LABEL[layer]:<20}{'-':<10}{'not measured':>12}")
        elif not m.available:
            lines.append(
                f"{LAYER_LABEL[layer]:<20}{m.buffer:<10}"
                f"{(m.note or 'unavailable'):>12}")
        else:
            bps = m.busbw_bps
            assert bps is not None  # .available guarantees this
            lines.append(
                f"{LAYER_LABEL[layer]:<20}{m.buffer:<10}"
                f"{bps / 1e9:>12.2f}")

    if report.gaps:
        lines.append("")
        lines.append("gaps between adjacent layers:")
        for g in report.gaps:
            if g.inverted:
                lines.append(
                    f"  {LAYER_LABEL[g.lower]} -> {LAYER_LABEL[g.upper]}: "
                    f"upper layer faster ({g.efficiency:.2f}x) -- check "
                    f"comparability"
                )
            else:
                lines.append(
                    f"  {LAYER_LABEL[g.lower]} -> {LAYER_LABEL[g.upper]}: "
                    f"{g.efficiency * 100:.1f}% retained, "
                    f"{g.overhead_pct:.1f}% lost to {g.component}"
                )

    d = report.dominant
    if d is not None:
        lines.append("")
        lines.append(
            f"dominant cost: {d.component} "
            f"({d.overhead_pct:.1f}% of {LAYER_LABEL[d.lower]} bandwidth)"
        )

    if report.incomparable:
        lines.append("")
        lines.append("not compared:")
        lines.extend(f"  {s}" for s in report.incomparable)

    if report.missing:
        lines.append("")
        lines.append("missing layers:")
        lines.extend(f"  {s}" for s in report.missing)

    return "\n".join(lines)
