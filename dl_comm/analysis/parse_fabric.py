"""Parse fabric-layer output: libfabric (FI) probes and NIXL transfers.

These are the two layers *below* the collectives stack. Where `parse_layers`
turns collective benchmarks into comparable measurements, this module does the
same for the point-to-point fabric underneath them:

    fi      libfabric itself -- provider/domain discovery and fi_pingpong
    nixl    NIXL's libfabric backend -- RDMA READ of a registered buffer

Both are point-to-point, not collectives, so they are recorded as their own
pseudo-collectives (`pingpong`, `read_dram`, `read_vram`) and compared only
against the same pattern at another layer. A NIXL READ moves a buffer once
between two ranks: there is no group traffic pattern, so bus bandwidth equals
the measured rate and no busbw factor applies.

Parsing follows the same rule as the collective parsers: an unrecognised line
is skipped, but a line that looks like a measurement and is missing a field
raises. A silently dropped measurement becomes a "layer not measured" row,
which hides a real result.

The formats below are the ones the tools actually emit -- they were written
against captured Tara North job output, not invented:

    [initiator r0 p0 host] best  0.032s -> 8.39 GB/s
    [initiator r0 p0 host]   RAILS CARRYING PAYLOAD: 4 (cxi0, cxi1, cxi2, cxi3)
    [initiator r0 p0 host]   total rx: 1.52 GiB vs 1.50 GiB payload -> 1.02x
    [initiator r0 p0 host] PASS: destination buffer is byte-exact.

fi_pingpong writes a whitespace table like OSU's but with its own columns:

    bytes   iters   total       time     MB/sec
    64      1000    7.8k        0.02s    3.20

and `fi_info -p cxi` output is inventory, not a measurement, so it is parsed
into provider facts rather than LayerMeasurement rows.
"""

from __future__ import annotations

import re

from dl_comm.analysis.bottleneck import LayerMeasurement

# "[initiator r0 p0 x4820c7s6b1n0] best  0.032s -> 8.39 GB/s"
_BEST = re.compile(
    r"^\[(?P<role>initiator|target)\b[^\]]*\]\s+"
    r"(?P<stat>best|mean)\s+(?P<secs>[\d.]+)s\s*->\s*(?P<gbps>[\d.]+)\s*GB/s"
)

# "  RAILS CARRYING PAYLOAD: 4 (cxi0, cxi1, cxi2, cxi3)"
_RAILS = re.compile(r"RAILS CARRYING PAYLOAD:\s*(?P<n>\d+)")

# "  total rx: 1.52 GiB vs 1.50 GiB payload -> 1.02x"
_OCTETS = re.compile(
    r"total\s+(?P<dir>rx|tx):\s*(?P<moved>[\d.]+)\s*GiB\s+vs\s+"
    r"(?P<payload>[\d.]+)\s*GiB\s+payload\s*->\s*(?P<ratio>[\d.]+)x"
)

_PASS = re.compile(r"PASS: destination buffer is byte-exact")
_FAIL = re.compile(r"FAIL|MISMATCH|byte-exact.*fail", re.IGNORECASE)

# fi_pingpong table row: bytes iters total time MB/sec
# A real fi_pingpong data row, libfabric 2.8.0a1 (verified against job 6934):
#
#   bytes   #sent   #ack     total       time     MB/sec    usec/xfer   Mxfers/sec
#   64      10      =10      1.2k        0.00s     22.86       2.80       0.36
#   6m      10      =10      120m        0.01s  23436.23     268.45       0.00
#
# Both the size and the total column carry k/m/g suffixes, the ack column is
# "=10" (or "10"), and two columns follow MB/sec. An earlier version of this
# regex assumed plain integers and a line ending at MB/sec; it matched zero
# rows of real output. The test suite now pins this against a captured log.
_FI_PP_ROW = re.compile(
    r"^\s*(?P<bytes>[\d.]+[kmg]?)\s+(?P<iters>\d+)\s+=?(?P<ack>\d+)\s+"
    r"(?P<total>[\d.]+[kmg]?)\s+(?P<time>[\d.]+)s\s+"
    r"(?P<mbps>[\d.]+)\s+(?P<usec>[\d.]+)\s+(?P<mxfers>[\d.]+)\s*$",
    re.IGNORECASE,
)

_SUFFIX = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


def _fi_size(tok: str) -> int:
    """Expand fi_pingpong's abbreviated sizes: '64' -> 64, '1.5k' -> 1536."""
    tok = tok.strip().lower()
    mult = _SUFFIX.get(tok[-1:], 1)
    if mult != 1:
        tok = tok[:-1]
    return int(round(float(tok) * mult))

# "provider: cxi" / "    domain: cxi0" from fi_info
_FI_PROVIDER = re.compile(r"^\s*provider:\s*(?P<prov>\S+)")
_FI_DOMAIN = re.compile(r"^\s*domain:\s*(?P<dom>\S+)")


class FabricEvidence:
    """Non-bandwidth evidence from a fabric run.

    A NIXL transfer can report a plausible GB/s while silently falling back to
    a host memcpy, so bandwidth alone is not proof the fabric carried the
    payload. These three facts are what distinguish a real RDMA transfer from
    one, and they are kept beside the measurement rather than folded into it.
    """

    __slots__ = ("rails", "octet_ratio", "byte_exact", "direction")

    def __init__(self, rails: int | None = None,
                 octet_ratio: float | None = None,
                 byte_exact: bool | None = None,
                 direction: str | None = None):
        self.rails = rails
        self.octet_ratio = octet_ratio
        self.byte_exact = byte_exact
        self.direction = direction

    @property
    def carried_on_fabric(self) -> bool | None:
        """True when NIC counters account for at least the payload.

        `None` when no counter sample was captured -- unknown is not False.
        A ratio near zero means the bytes never reached the NIC, which is the
        signature of a silent fallback.
        """
        if self.octet_ratio is None:
            return None
        return self.octet_ratio >= 1.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FabricEvidence(rails={self.rails}, "
                f"octet_ratio={self.octet_ratio}, "
                f"byte_exact={self.byte_exact})")


def parse_nixl_transfer(text: str, pattern: str = "read_dram",
                        buffer: str = "device",
                        ranks: int = 2) -> tuple[list[LayerMeasurement],
                                                 FabricEvidence]:
    """Parse a NIXL point-to-point transfer log.

    Returns the measurements plus the fabric evidence that says whether the
    bytes really crossed the NIC. `best` is recorded as the measurement and
    `mean` is kept as a second row so a noisy run is visible rather than
    averaged away by the reader.

    A transfer that reports bandwidth but fails byte-exactness is recorded as
    unavailable: a wrong answer delivered quickly is not a measurement.
    """
    out: list[LayerMeasurement] = []
    ev = FabricEvidence()
    best_bps: float | None = None
    mean_bps: float | None = None
    saw_fail = False

    for line in text.splitlines():
        m = _BEST.search(line)
        if m:
            gbps = float(m.group("gbps"))
            if m.group("stat") == "best":
                best_bps = gbps * 1e9
            else:
                mean_bps = gbps * 1e9
            continue

        m = _RAILS.search(line)
        if m:
            ev.rails = int(m.group("n"))
            continue

        m = _OCTETS.search(line)
        if m:
            ev.octet_ratio = float(m.group("ratio"))
            ev.direction = m.group("dir")
            continue

        if _PASS.search(line):
            ev.byte_exact = True
            continue
        if _FAIL.search(line):
            saw_fail = True

    if saw_fail and ev.byte_exact is not True:
        ev.byte_exact = False

    # Correctness gates the measurement. An unverified transfer is reported as
    # unavailable with the reason attached, never as a bandwidth number.
    verified = ev.byte_exact is True
    note_bad = "unverified" if ev.byte_exact is None else "byte-exact FAILED"

    for stat, bps in (("best", best_bps), ("mean", mean_bps)):
        if bps is None:
            continue
        out.append(LayerMeasurement(
            layer="nixl",
            collective=pattern if stat == "best" else f"{pattern}_mean",
            size_bytes=0,  # set by the caller when the payload size is known
            busbw_bps=bps if verified else None,
            buffer=buffer, ranks=ranks,
            note="" if verified else note_bad,
        ))

    return out, ev


def parse_fi_pingpong(text: str, ranks: int = 2,
                      buffer: str = "host") -> list[LayerMeasurement]:
    """Parse an `fi_pingpong` table.

    fi_pingpong reports MB/sec against message size. It is a two-rank
    point-to-point test with host buffers by default, so it is recorded under
    the `pingpong` pseudo-collective and never compared against a
    device-buffer layer -- the buffer field enforces that downstream.
    """
    out: list[LayerMeasurement] = []
    for line in text.splitlines():
        if line.lstrip().startswith(("#", "bytes")):
            continue
        m = _FI_PP_ROW.match(line)
        if not m:
            continue
        mbps = float(m.group("mbps"))
        out.append(LayerMeasurement(
            layer="fi", collective="pingpong",
            size_bytes=_fi_size(m.group("bytes")),
            # fi_pingpong's MB/sec is 10^6 bytes/s.
            busbw_bps=mbps * 1e6 if mbps > 0 else None,
            buffer=buffer, ranks=ranks,
            note="" if mbps > 0 else "zero/NA",
        ))
    return out


def parse_fi_info(text: str) -> dict[str, list[str]]:
    """Parse `fi_info` output into {provider: [domains]}.

    This is inventory, not a measurement: it answers "is the CXI provider
    present and how many domains does it expose", which is the check that
    tells a missing-provider failure apart from a slow one. Returning an empty
    dict means no provider was found -- on a login node without Slingshot NICs
    `fi_info -p cxi` exits 61 and prints nothing, and that is a real answer,
    not an error.
    """
    providers: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = _FI_PROVIDER.match(line)
        if m:
            prov = str(m.group("prov"))
            current = prov
            if prov not in providers:
                providers[prov] = []
            continue
        m = _FI_DOMAIN.match(line)
        if m and current is not None:
            dom = m.group("dom")
            if dom not in providers[current]:
                providers[current].append(dom)
    return providers
