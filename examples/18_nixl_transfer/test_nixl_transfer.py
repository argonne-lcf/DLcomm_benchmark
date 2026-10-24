#!/usr/bin/env python3
"""Unit tests for the NIXL transfer layer that need no NIXL and no GPU.

Run:
    python -m pytest examples/18_nixl_transfer/test_nixl_transfer.py -q
    python examples/18_nixl_transfer/test_nixl_transfer.py        # no pytest

These cover the reporting arithmetic and argument handling -- the parts that
were actually wrong during bring-up on Tara and that a cluster run does NOT
catch, because a bandwidth number that is 4x too high still looks plausible
in a log.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dl_comm.transfer.nixl_transfer import (  # noqa: E402
    NixlTransferResult,
    format_nixl_table,
)


def test_bandwidth_is_per_rank_not_scaled_by_world_size() -> None:
    """Regression: world_size must not inflate a rank's own bandwidth.

    The original code returned ``nbytes * world_size``, so the SAME measured
    time reported 4x the bandwidth at 16 ranks that it did at 4 -- per-node
    throughput appeared to rise from 6.8 to 27.7 GB/s on identical hardware.
    """
    nbytes = 64 * 1024 * 1024
    secs = 0.137

    one = NixlTransferResult(direction="h2d", backend="UCX",
                             nbytes=nbytes, world_size=1, times_s=[secs])
    many = NixlTransferResult(direction="h2d", backend="UCX",
                              nbytes=nbytes, world_size=16, times_s=[secs])

    assert one.gbytes_per_s == many.gbytes_per_s, (
        "identical timings must yield identical per-rank bandwidth "
        f"({one.gbytes_per_s} vs {many.gbytes_per_s})"
    )
    assert abs(one.gbytes_per_s - nbytes / secs / 1e9) < 1e-9


def test_best_s_uses_minimum_not_last() -> None:
    r = NixlTransferResult(direction="d2d", backend="UCX", nbytes=1024,
                           world_size=1, times_s=[0.5, 0.2, 0.9])
    assert r.best_s == 0.2


def test_empty_timings_raise_instead_of_reporting_zero() -> None:
    """A run that measured nothing must fail loudly, not report 0.00 GB/s."""
    r = NixlTransferResult(direction="d2d", backend="UCX", nbytes=1024,
                           world_size=1, times_s=[])
    for attr in ("best_s", "gbytes_per_s"):
        try:
            getattr(r, attr)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{attr} must raise when no timings exist")


def test_table_renders_every_direction() -> None:
    rows = [
        NixlTransferResult(direction=d, backend="UCX", nbytes=4096,
                           world_size=1, times_s=[0.001])
        for d in ("h2d", "d2h", "d2d")
    ]
    out = format_nixl_table(rows)
    for d in ("h2d", "d2h", "d2d"):
        assert d in out
    assert "(no NIXL results)" == format_nixl_table([])


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
