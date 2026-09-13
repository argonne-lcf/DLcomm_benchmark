"""Locate, build, run, and parse OSU Micro-Benchmarks.

Design note -- why not a git submodule
--------------------------------------
The initial request was for OSU as a dependent package or submodule. Probing
Aurora first showed that is the wrong shape:

* ``/soft/tools/osu/`` already exists with prebuilt p2p binaries
  (``osu_latency``, ``osu_mbw_mr``, ``osu_multi_lat``, ``osu_latency_mp``)
  built against the site MPI.
* ``/soft/tools/osu/osu-micro-benchmarks-7.1-1.tar.gz`` carries the complete
  source: 14 blocking collectives, 14 non-blocking, 7 pt2pt, one-sided, NCCL.

OSU is an MPI-ABI-sensitive C program. A vendored submodule would have to be
rebuilt against each site's MPI anyway, and on Aurora it would *shadow* a
working site build with one compiled against whatever mpicc happened to be
loaded -- a real risk of comparing DLcomm's XCCL numbers against a
mis-built MPI reference. So this module *discovers* OSU with a documented
precedence, and only builds from the site tarball when no binary is found.

Precedence:
    1. ``$DLCOMM_OSU_DIR``          -- explicit user override, wins always
    2. ``/soft/tools/osu``          -- Aurora site install
    3. ``$PATH``                    -- module-provided install
    4. built tarball under          -- last resort, opt-in via build_from_tarball()
       ``$DLCOMM_OSU_BUILD``
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

#: Site locations searched for OSU binaries, in precedence order.
SITE_DIRS = ("/soft/tools/osu",)

#: Site source tarballs searched when a build is requested.
SITE_TARBALLS = ("/soft/tools/osu/osu-micro-benchmarks-7.1-1.tar.gz",)

#: DLcomm collective -> OSU benchmark measuring the same operation.
#:
#: Only genuine equivalents are listed. A DLcomm collective absent from this
#: map has no OSU counterpart and must not be silently compared against a
#: near-neighbour -- that would produce a plausible but meaningless ratio.
OSU_EQUIVALENTS = {
    "allreduce":      "osu_allreduce",
    "allgather":      "osu_allgather",
    "alltoall":       "osu_alltoall",
    "alltoallsingle": "osu_alltoall",
    "alltoallv":      "osu_alltoallv",
    "broadcast":      "osu_bcast",
    "gather":         "osu_gather",
    "scatter":        "osu_scatter",
    "reduce":         "osu_reduce",
    "reducescatter":  "osu_reduce_scatter",
    "barrier":        "osu_barrier",
    "sendrecv":       "osu_bw",
    "sendrecv_async": "osu_bibw",
}


class OsuNotFound(RuntimeError):
    """Raised when no OSU installation can be located."""


@dataclass
class OsuResult:
    """One OSU run: size -> metric, plus provenance."""

    benchmark: str
    binary: str
    #: size in bytes -> measured value (us for latency, MB/s for bandwidth)
    points: dict[int, float] = field(default_factory=dict)
    #: "latency" (microseconds) or "bandwidth" (MB/s)
    metric: str = "unknown"
    raw: str = ""

    def value_at(self, size_bytes: int) -> float | None:
        return self.points.get(size_bytes)

    def bytes_per_second_at(self, size_bytes: int) -> float | None:
        """Normalise to B/s so OSU can be compared with DLcomm's algbw."""
        v = self.points.get(size_bytes)
        if v is None:
            return None
        if self.metric == "bandwidth":
            return v * 1e6          # MB/s -> B/s (OSU uses 1e6, not 2^20)
        if self.metric == "latency" and v > 0:
            return size_bytes / (v * 1e-6)
        return None


def _candidate_dirs() -> Iterable[Path]:
    override = os.environ.get("DLCOMM_OSU_DIR")
    if override:
        yield Path(override)
    for d in SITE_DIRS:
        yield Path(d)
    build = os.environ.get("DLCOMM_OSU_BUILD")
    if build:
        # autotools installs land in libexec/osu-micro-benchmarks/mpi/*
        root = Path(build)
        yield root
        for sub in ("mpi/collective", "mpi/pt2pt",
                    "libexec/osu-micro-benchmarks/mpi/collective",
                    "libexec/osu-micro-benchmarks/mpi/pt2pt"):
            yield root / sub


def discover(benchmark: str) -> str:
    """Return the path to an OSU benchmark binary.

    Raises OsuNotFound with the searched locations rather than returning None,
    so a missing install can never be mistaken for a zero measurement.
    """
    searched: list[str] = []
    for d in _candidate_dirs():
        searched.append(str(d))
        p = d / benchmark
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        # recurse one level: site layouts vary
        if d.is_dir():
            for child in sorted(d.glob(f"*/{benchmark}")):
                if child.is_file() and os.access(child, os.X_OK):
                    return str(child)
    on_path = shutil.which(benchmark)
    if on_path:
        return on_path
    searched.append("$PATH")
    raise OsuNotFound(
        f"OSU benchmark '{benchmark}' not found. Searched: {searched}. "
        f"Set DLCOMM_OSU_DIR to an OSU bin directory, or call "
        f"build_from_tarball() to compile the site tarball."
    )


def equivalent_for(collective: str) -> str | None:
    """OSU benchmark measuring the same op, or None when there is no analogue."""
    return OSU_EQUIVALENTS.get(collective.lower())


#: OSU emits "# Size Avg Latency(us)" or "# Size Bandwidth (MB/s)" headers.
_HDR = re.compile(r"^#.*\b(Latency|Bandwidth)\b", re.IGNORECASE)
_ROW = re.compile(r"^\s*(\d+)\s+([0-9.eE+-]+)")


def parse_osu_output(text: str) -> OsuResult:
    """Parse OSU stdout into size -> value.

    OSU prints comment lines beginning with '#', then whitespace-separated
    "size value" rows. Barrier is the exception: it emits a single value with
    no size column, which is recorded under size 0.
    """
    metric = "unknown"
    points: dict[int, float] = {}
    for line in text.splitlines():
        m = _HDR.search(line)
        if m:
            metric = "latency" if m.group(1).lower() == "latency" else "bandwidth"
            continue
        if line.lstrip().startswith("#") or not line.strip():
            continue
        row = _ROW.match(line)
        if row:
            points[int(row.group(1))] = float(row.group(2))
            continue
        # barrier: a lone float on its own line
        bare = line.strip().split()
        if len(bare) == 1:
            try:
                points[0] = float(bare[0])
                if metric == "unknown":
                    metric = "latency"
            except ValueError:
                pass
    return OsuResult(benchmark="", binary="", points=points, metric=metric, raw=text)


def run_osu(benchmark: str, ranks: int, ppn: int | None = None,
            min_size: int | None = None, max_size: int | None = None,
            iterations: int | None = None, launcher: str = "mpiexec",
            extra_args: list[str] | None = None,
            timeout: int = 900) -> OsuResult:
    """Run an OSU benchmark under mpiexec and parse its output.

    The binary is discovered, not assumed. A non-zero exit or unparseable
    output raises rather than returning empty points, so a failed reference run
    cannot be reported as a comparison.
    """
    binary = discover(benchmark)
    cmd = [launcher, "-n", str(ranks)]
    if ppn:
        cmd += ["-ppn", str(ppn)]
    cmd.append(binary)
    if min_size is not None and max_size is not None:
        cmd += ["-m", f"{min_size}:{max_size}"]
    if iterations is not None:
        cmd += ["-i", str(iterations)]
    if extra_args:
        cmd += extra_args

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"OSU run failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"stderr: {proc.stderr[:800]}"
        )
    res = parse_osu_output(proc.stdout)
    if not res.points:
        raise RuntimeError(
            f"OSU produced no parseable data points: {' '.join(cmd)}\n"
            f"stdout head: {proc.stdout[:500]}"
        )
    res.benchmark = benchmark
    res.binary = binary
    return res


def build_from_tarball(dest: str, tarball: str | None = None,
                       cc: str = "mpicc", cxx: str = "mpicxx",
                       timeout: int = 1800) -> str:
    """Compile OSU from the site tarball into ``dest``; return the install dir.

    Only used when no site binary exists. On Aurora prefer the prebuilt
    binaries: a local build picks up whichever mpicc is loaded, which may not
    match the MPI the rest of the job uses.
    """
    src_tar = tarball
    if src_tar is None:
        for cand in SITE_TARBALLS:
            if Path(cand).is_file():
                src_tar = cand
                break
    if src_tar is None or not Path(src_tar).is_file():
        raise OsuNotFound(
            f"No OSU source tarball found. Looked for: {list(SITE_TARBALLS)}. "
            f"Pass tarball= explicitly."
        )

    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "xzf", src_tar, "-C", str(dest_p)],
                   check=True, timeout=timeout)
    roots = [p for p in dest_p.iterdir() if p.is_dir() and p.name.startswith("osu-")]
    if not roots:
        raise RuntimeError(f"tarball {src_tar} did not unpack an osu-* directory")
    root = roots[0]
    prefix = dest_p / "install"
    subprocess.run(["./configure", f"CC={cc}", f"CXX={cxx}",
                    f"--prefix={prefix}"],
                   cwd=root, check=True, timeout=timeout,
                   capture_output=True, text=True)
    subprocess.run(["make", "-j", "8"], cwd=root, check=True,
                   timeout=timeout, capture_output=True, text=True)
    subprocess.run(["make", "install"], cwd=root, check=True,
                   timeout=timeout, capture_output=True, text=True)
    return str(prefix)
