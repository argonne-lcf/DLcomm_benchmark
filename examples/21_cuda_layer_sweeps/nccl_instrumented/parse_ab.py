#!/usr/bin/env python3
"""Pull the nccl-tests result row + transport out of an A/B log.

nccl-tests writes its data rows to stdout while NCCL_DEBUG=INFO writes to
stderr, and mpiexec interleaves them into the same file -- so a naive
'grep ^<size>' matches INFO lines that merely start with the size, and
column indexing lands on 'NCCL'/'INFO' instead of numbers. Identify a data
row structurally instead: >= 11 whitespace fields, first two integers,
third field a known dtype.
"""
import re
import sys

DTYPES = {"int8", "uint8", "int32", "uint32", "int64", "uint64",
          "half", "float", "double", "bfloat16", "fp8_e4m3", "fp8_e5m2"}


def data_rows(path):
    """Return (time_us, algbw, busbw) for each result row.

    mpiexec interleaves NCCL's stderr into nccl-tests' stdout mid-line, so
    the row's leading fields (size, count, dtype, redop, root) are often
    overwritten and only the trailing numeric block survives, e.g.

      3990.12   67.28  117.73       0  3605.50   74.45  130.29       0

    That is [time algbw busbw wrong] x2 (in-place, out-of-place). Match on
    that structure -- 8 numeric fields with integer 4th and 8th -- rather
    than on a leading size that may no longer be present.
    """
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "NCCL INFO" in line or "NET/OFI" in line or line.startswith("#"):
                continue
            f = line.split()
            # full, uncorrupted row
            if len(f) >= 11 and f[0].isdigit() and f[1].isdigit() and f[2] in DTYPES:
                out.append((f[5], f[6], f[7]))
                continue
            # truncated row: the 8-column numeric tail
            if len(f) == 8:
                try:
                    vals = [float(x) for x in f]
                except ValueError:
                    continue
                if f[3].isdigit() and f[7].isdigit():
                    out.append((f[0], f[1], f[2]))
    return out


def scalar(path, pattern):
    rx = re.compile(pattern)
    with open(path, errors="replace") as fh:
        for line in fh:
            m = rx.search(line)
            if m:
                return m.group(1)
    return None


def transport(path):
    seen = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.search(r"Using network (\S+)", line)
            if m and m.group(1) not in seen:
                seen.append(m.group(1))
    return ",".join(seen) if seen else "UNKNOWN"


def main():
    print(f"{'arm':<8} {'transport':<12} {'time_us':>12} "
          f"{'algbw_GBs':>11} {'busbw_GBs':>11} {'avg_busbw':>10}")
    for path in sys.argv[1:]:
        arm = path.rstrip("/").split("/")[-1].replace(".txt", "")
        rows = data_rows(path)
        avg = scalar(path, r"Avg bus bandwidth\s*:\s*([\d.]+)")
        tr = transport(path)
        if not rows:
            print(f"{arm:<8} {tr:<12} {'NO_DATA_ROW':>12}"
                  f"{'':>11}{'':>11} {avg or 'n/a':>10}")
            continue
        for t, alg, bus in rows:
            print(f"{arm:<8} {tr:<12} {t:>12} "
                  f"{alg:>11} {bus:>11} {avg or 'n/a':>10}")


if __name__ == "__main__":
    main()
