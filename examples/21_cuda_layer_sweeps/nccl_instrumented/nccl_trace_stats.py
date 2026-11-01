#!/usr/bin/env python3
"""Aggregate NCCL profiler-plugin traces into a phase breakdown.

Input: the per-rank JSON written by ext-profiler/example when
NCCL_PROFILE_DUMP_FILE is set (Chrome trace format: paired 'b'/'e' events
keyed by (name, id, pid, tid)).

The point of this over NCCL_DEBUG text: it separates GPU-side channel time
(cat=GPU) from host proxy progress (cat=PROXY) and from network wait
(cat=NET), so "the collective is slow" becomes an attributable split.

Usage: nccl_trace_stats.py <trace_*.json> ...
"""
import json
import statistics
import sys
from collections import defaultdict


def durations(path):
    """Pair begin/end events and return {(cat,name): [durations_us]}."""
    with open(path) as fh:
        events = json.load(fh)

    open_ev = defaultdict(list)
    out = defaultdict(list)
    for e in events:
        ph = e.get("ph")
        if ph not in ("b", "e"):
            continue
        key = (e.get("name"), e.get("id"), e.get("pid"), e.get("tid"))
        if ph == "b":
            open_ev[key].append(e)
        else:
            if not open_ev.get(key):
                continue  # unmatched end; ignore rather than guess
            start = open_ev[key].pop()
            dur = e.get("ts", 0.0) - start.get("ts", 0.0)
            if dur >= 0:
                out[(start.get("cat", "?"), start.get("name", "?"))].append(dur)
    unmatched = sum(len(v) for v in open_ev.values())
    return out, unmatched


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: nccl_trace_stats.py <trace_*.json> ...")
        return 1

    agg = defaultdict(list)
    total_unmatched = 0
    for p in paths:
        d, un = durations(p)
        total_unmatched += un
        for k, v in d.items():
            agg[k].extend(v)

    print(f"files={len(paths)}  unmatched_begin_events={total_unmatched}")
    print()
    print(f"{'cat':<7} {'event':<18} {'n':>6} {'mean_us':>11} "
          f"{'median_us':>11} {'max_us':>11} {'total_us':>13}")
    print("-" * 82)

    # Sort by total time: the biggest contributor first is the whole point.
    for (cat, name), vals in sorted(
            agg.items(), key=lambda kv: -sum(kv[1])):
        print(f"{cat:<7} {name:<18} {len(vals):>6} "
              f"{statistics.mean(vals):>11.2f} "
              f"{statistics.median(vals):>11.2f} "
              f"{max(vals):>11.2f} {sum(vals):>13.2f}")

    # Category rollup answers "GPU or wire?" directly.
    print()
    by_cat = defaultdict(float)
    for (cat, _), vals in agg.items():
        by_cat[cat] += sum(vals)
    grand = sum(by_cat.values())
    print(f"{'category':<10} {'total_us':>14} {'share':>8}")
    print("-" * 34)
    for cat, tot in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        share = (100.0 * tot / grand) if grand else 0.0
        print(f"{cat:<10} {tot:>14.2f} {share:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
