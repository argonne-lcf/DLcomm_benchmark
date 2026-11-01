#!/usr/bin/env python3
"""Decompose NCCL run-to-run variance into within-job and across-job parts.

The ~57% figure was never attributed. Within-run jitter was a candidate, not
a diagnosis. Each input is one INDEPENDENT PBS job that repeated the same
cell N times, so:

  within-job  spread = repeats inside one allocation   -> jitter
  across-job  spread = spread of per-job medians       -> placement/topology

If across-job dominates, tuning inside a run cannot fix it. Node lists are
printed so a slow allocation can be tied to specific hardware.

Usage: variance_stats.py <job_dir> [<job_dir> ...]
       each dir holds rep*.txt and nodes.txt
"""
import re
import statistics as st
import sys
from pathlib import Path

ROW = re.compile(r"^\s*(?:\S+\s+)*?(\d+\.\d+)\s+")


def parse_rep(path):
    """Return (busbw, transport) or (None, reason).

    Rows are matched structurally: stderr interleaves into stdout under
    mpiexec and destroys the leading fields, so a size-anchored grep is
    unreliable. A run that failed validation is never given a number --
    nccl-tests prints a bandwidth even when its own check failed.
    """
    txt = path.read_text(errors="replace")
    if "error while loading shared libraries" in txt:
        return None, "libfail"
    m = re.search(r"Out of bounds values : (\d+)", txt)
    if m and m.group(1) != "0":
        return None, f"validation_oob_{m.group(1)}"
    tm = re.search(r"Using network (\w+)", txt)
    transport = tm.group(1) if tm else "unknown"
    bm = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", txt)
    if not bm:
        return None, "no_result"
    return (float(bm.group(1)), transport)


def cov(xs):
    """Coefficient of variation, as a percentage."""
    if len(xs) < 2:
        return 0.0
    return 100.0 * st.stdev(xs) / st.mean(xs)


def spread(xs):
    """max/min - 1, as a percentage: the 'X% variance' people usually mean."""
    if len(xs) < 2:
        return 0.0
    return 100.0 * (max(xs) / min(xs) - 1.0)


def main(dirs):
    jobs = []
    for d in dirs:
        d = Path(d)
        reps, bad = [], []
        for f in sorted(d.glob("rep*.txt")):
            val, info = parse_rep(f)
            (reps.append(val) if val is not None else bad.append((f.name, info)))
        nodes = []
        nf = d / "nodes.txt"
        if nf.exists():
            nodes = [n.strip() for n in nf.read_text().split() if n.strip()]
        jobs.append({"dir": d.name, "reps": reps, "bad": bad, "nodes": nodes})

    print("== per-job (each is a separate allocation) ==")
    print(f"{'job':<28} {'n':>2} {'median':>9} {'min':>9} {'max':>9} "
          f"{'CoV%':>7} {'spread%':>8}  nodes")
    medians = []
    for j in jobs:
        r = j["reps"]
        if not r:
            print(f"{j['dir']:<28} {'0':>2}  NO VALID REPS  {j['bad']}")
            continue
        medians.append(st.median(r))
        print(f"{j['dir']:<28} {len(r):>2} {st.median(r):>9.2f} {min(r):>9.2f} "
              f"{max(r):>9.2f} {cov(r):>7.2f} {spread(r):>8.1f}  "
              f"{','.join(n.split('.')[0] for n in j['nodes'])}")
        for name, why in j["bad"]:
            print(f"{'':<28}    EXCLUDED {name}: {why}")

    if not medians:
        print("\nno valid data")
        return 1

    within = [cov(j["reps"]) for j in jobs if len(j["reps"]) > 1]
    print("\n== decomposition ==")
    print(f"  jobs with valid data     : {len(medians)}")
    print(f"  total valid measurements : {sum(len(j['reps']) for j in jobs)}")
    if within:
        print(f"  within-job CoV  (jitter)      : "
              f"mean {st.mean(within):.2f}%  max {max(within):.2f}%")
    print(f"  across-job CoV  (placement)   : {cov(medians):.2f}%")
    allv = [v for j in jobs for v in j["reps"]]
    print(f"  overall CoV  (what a user sees): {cov(allv):.2f}%")
    print(f"  overall spread (max/min-1)     : {spread(allv):.1f}%")
    print(f"  overall min/max                : {min(allv):.2f} / {max(allv):.2f} GB/s")

    print("\n== verdict ==")
    if within:
        w, a = st.mean(within), cov(medians)
        if a > 2 * w:
            print(f"  ACROSS-JOB dominates ({a:.2f}% vs {w:.2f}%): variance is")
            print("  placement/topology, not in-run jitter. Tuning inside a run")
            print("  will not fix it.")
        elif w > 2 * a:
            print(f"  WITHIN-JOB dominates ({w:.2f}% vs {a:.2f}%): variance is")
            print("  in-run jitter -- proxy scheduling is the place to look.")
        else:
            print(f"  COMPARABLE (within {w:.2f}% vs across {a:.2f}%): both")
            print("  contribute; neither alone explains the spread.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
