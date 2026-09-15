"""OSU Micro-Benchmarks integration for DLcomm.

Provides an external reference point: DLcomm measures framework-level
collectives (torch.distributed / torchcomms over XCCL), while OSU measures the
same operations at the MPI transport layer. Running both on identical payloads
turns "our number is 2.9 GB/s" into "our number is 2.9 GB/s against an MPI
floor of X GB/s", which is the difference between a measurement and a result.

See ``docs/fixes/18-osu-integration.md``.
"""

from dl_comm.osu.runner import (  # noqa: F401
    OSU_EQUIVALENTS,
    OsuNotFound,
    OsuResult,
    build_from_tarball,
    discover,
    equivalent_for,
    parse_osu_output,
    run_osu,
)

__all__ = [
    "OSU_EQUIVALENTS",
    "OsuNotFound",
    "OsuResult",
    "build_from_tarball",
    "discover",
    "equivalent_for",
    "parse_osu_output",
    "run_osu",
]
