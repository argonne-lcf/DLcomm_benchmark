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
from dl_comm.analysis.parse_fabric import (
    parse_fi_pingpong,
    parse_nixl_transfer,
)
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
    evidence = {}

    for path in sorted(results_dir.glob("*.txt")):
        text = path.read_text(errors="replace")
        name = path.name

        if name.startswith("osu_"):
            binary = name[:-4]  # drop .txt
            found = parse_osu(text, binary, ranks=ranks)
        elif "RAILS CARRYING PAYLOAD" in text or "NIXL" in text:
            # NIXL point-to-point transfer. The pattern is taken from the
            # filename so a DRAM and a VRAM run in the same directory stay
            # distinct instead of overwriting each other.
            stem = name[:-4]
            if "vram" in stem.lower():
                pattern, buf = "read_vram", "device"
            else:
                pattern, buf = "read_dram", "host"
            found, ev = parse_nixl_transfer(text, pattern=pattern,
                                            buffer=buf, ranks=ranks)
            if found:
                evidence[name] = ev
        elif "fi_pingpong" in name or "MB/sec" in text:
            found = parse_fi_pingpong(text, ranks=ranks)
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

    return measurements, seen_files, evidence


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=pathlib.Path)
    ap.add_argument("--ranks", type=int, default=12,
                    help="rank count used for OSU busbw conversion")
    args = ap.parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"not a directory: {args.results_dir}", file=sys.stderr)
        return 2

    measurements, seen, evidence = collect(args.results_dir, args.ranks)

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

    # Fabric evidence is reported separately from bandwidth: a NIXL transfer
    # can print a plausible GB/s while silently falling back to a host memcpy,
    # so the NIC counters and the byte-exact check are what make the number
    # trustworthy. Unknown is printed as unknown, never as a pass.
    if evidence:
        print("fabric evidence (NIXL transfers)")
        print("-" * 66)
        print(f"{'file':<26}{'rails':>6}{'NIC/payload':>14}{'byte-exact':>14}")
        for fname, ev in sorted(evidence.items()):
            rails = str(ev.rails) if ev.rails is not None else "?"
            ratio = (f"{ev.octet_ratio:.2f}x"
                     if ev.octet_ratio is not None else "not sampled")
            if ev.byte_exact is True:
                exact = "PASS"
            elif ev.byte_exact is False:
                exact = "FAIL"
            else:
                exact = "unverified"
            print(f"{fname:<26}{rails:>6}{ratio:>14}{exact:>14}")
        print()
        print("  NIC/payload >= 1.00x means the bytes reached the wire; near")
        print("  zero is the signature of a silent host-memcpy fallback.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
