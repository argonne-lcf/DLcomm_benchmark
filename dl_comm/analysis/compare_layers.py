"""Cross-layer comparison entry point.

Reads the output files produced by a multi-layer run and prints one table per
collective showing every layer side by side, followed by the derived gaps.

Usage:

    python -m dl_comm.analysis.compare_layers RESULTS_DIR

RESULTS_DIR is a directory written by one of the PBS run scripts. Files are
matched by name:

    *ccl*.txt, size_sweep.txt   C++ CCL layer (LAYER=cpp_ccl records)
    osu_<binary>.txt            OSU tables, one per binary
    torch_dist.txt              torch.distributed (LAYER=torch_dist records)
    torchcomms.txt              torchcomms (LAYER=torchcomms records)

A layer with no file is reported as not measured. It is never inferred from
another layer.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from dl_comm.analysis.bottleneck import analyse, format_report
from dl_comm.analysis.parse_layers import (
    group_by_op_size,
    parse_kv_lines,
    parse_osu,
    parse_transfer,
)


def collect(results_dir: pathlib.Path, ranks: int):
    """Read every recognised layer file in a results directory."""
    measurements = []
    seen_files = []

    for path in sorted(results_dir.glob("*.txt")):
        text = path.read_text(errors="replace")
        name = path.name

        if name.startswith("osu_"):
            binary = name[:-4]  # drop .txt
            found = parse_osu(text, binary, ranks=ranks)
        elif "LAYER=cpp_ccl" in text:
            found = parse_kv_lines(text, "cpp_ccl")
        elif "PATTERN=" in text:
            # Transfer output (h2d/d2h/d2d/bidirectional), not a collective.
            found = parse_transfer(text, ranks=ranks)
        elif "LAYER=torch_dist" in text:
            found = parse_kv_lines(text, "torch_dist")
        elif "LAYER=torchcomms" in text:
            found = parse_kv_lines(text, "torchcomms")
        else:
            continue

        if found:
            measurements.extend(found)
            seen_files.append(f"{name} ({len(found)} records)")

    return measurements, seen_files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=pathlib.Path)
    ap.add_argument("--ranks", type=int, default=12,
                    help="rank count used for OSU busbw conversion")
    args = ap.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"not a directory: {args.results_dir}", file=sys.stderr)
        return 2

    measurements, seen = collect(args.results_dir, args.ranks)

    print(f"results dir : {args.results_dir}")
    print(f"ranks       : {args.ranks}")
    if seen:
        print("files read  :")
        for s in seen:
            print(f"  {s}")
    else:
        print("files read  : none recognised")
        print()
        print("No layer output found. Nothing is inferred from an empty run.")
        return 1
    print()

    buckets = group_by_op_size(measurements)
    for (collective, size) in sorted(buckets):
        print(format_report(analyse(buckets[(collective, size)])))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
