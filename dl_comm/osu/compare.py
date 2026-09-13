"""Compare DLcomm measurements against an OSU reference run.

DLcomm reports algbw/busbw in bytes/second; OSU reports latency in
microseconds or bandwidth in MB/s. This module normalises both to B/s and
reports the ratio, so a DLcomm number can be read against the MPI transport
floor for the same operation and payload.

The comparison is deliberately conservative:

* A collective with no OSU analogue is reported as ``no_equivalent``, never
  compared against a near-neighbour benchmark.
* A missing OSU data point at the requested size is ``no_reference``, not zero.
* Ratios are reported without a pass/fail threshold. A framework collective is
  legitimately slower than raw MPI (device buffers, Python dispatch, group
  setup), so an arbitrary cutoff would manufacture false failures. The number
  is the result; interpretation is the reader's.
"""

from __future__ import annotations

from dataclasses import dataclass

from dl_comm.osu.runner import OsuResult, equivalent_for


@dataclass
class Comparison:
    collective: str
    size_bytes: int
    dlcomm_bps: float | None
    osu_benchmark: str | None
    osu_bps: float | None
    status: str

    @property
    def ratio(self) -> float | None:
        """DLcomm / OSU. >1 means DLcomm moved more bytes per second."""
        if self.dlcomm_bps and self.osu_bps:
            return self.dlcomm_bps / self.osu_bps
        return None

    def format_row(self) -> str:
        def fmt(v):
            return "-" if v is None else f"{v:.3e}"
        r = self.ratio
        return (
            f"{self.collective:<16} {self.size_bytes:>10} "
            f"{fmt(self.dlcomm_bps):>12} {fmt(self.osu_bps):>12} "
            f"{('-' if r is None else f'{r:.2f}x'):>8}  {self.status}"
        )


def compare(collective: str, size_bytes: int, dlcomm_bps: float | None,
            osu: OsuResult | None) -> Comparison:
    """Build one comparison row, reporting absence honestly."""
    bench = equivalent_for(collective)
    if bench is None:
        return Comparison(collective, size_bytes, dlcomm_bps, None, None,
                          "no_equivalent")
    if osu is None:
        return Comparison(collective, size_bytes, dlcomm_bps, bench, None,
                          "no_reference")
    osu_bps = osu.bytes_per_second_at(size_bytes)
    if osu_bps is None:
        return Comparison(collective, size_bytes, dlcomm_bps, bench, None,
                          f"no_reference_at_{size_bytes}B")
    if dlcomm_bps is None:
        return Comparison(collective, size_bytes, None, bench, osu_bps,
                          "no_dlcomm_measurement")
    return Comparison(collective, size_bytes, dlcomm_bps, bench, osu_bps, "ok")


HEADER = (
    f"{'collective':<16} {'bytes':>10} {'dlcomm B/s':>12} "
    f"{'osu B/s':>12} {'ratio':>8}  status"
)


def format_table(rows: list[Comparison]) -> str:
    out = [HEADER, "-" * len(HEADER)]
    out += [r.format_row() for r in rows]
    return "\n".join(out)
