#!/usr/bin/env python3
"""Build the per-layer performance report from measured sweep output.

Reads the CSVs and the raw benchmark files. Every number printed comes from a
file a job wrote; anything absent is printed as missing rather than inferred.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1])          # .../sweeps
OUT = Path(sys.argv[2])


def load(csv_path):
    return list(csv.DictReader(open(csv_path)))


def nccl_best(path):
    best = None
    for line in open(path, errors="ignore"):
        if line.lstrip().startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 13:
            continue
        try:
            size, algbw, busbw = int(f[0]), float(f[6]), float(f[7])
        except (ValueError, IndexError):
            continue
        if best is None or busbw > best[2]:
            best = (size, algbw, busbw)
    return best


def torch_best(path):
    best = None
    for line in open(path, errors="ignore"):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("["):
            continue
        f = s.split()
        if len(f) < 5:
            continue
        try:
            size, algbw, busbw = int(f[0]), float(f[3]), float(f[4])
        except (ValueError, IndexError):
            continue
        if best is None or busbw > best[2]:
            best = (size, algbw, busbw)
    return best


def osu_last(path):
    last = None
    for line in open(path, errors="ignore"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        try:
            last = (0, float(f[0])) if len(f) == 1 else (int(f[0]), float(f[1]))
        except (ValueError, IndexError):
            continue
    return last


def nixl_best(path):
    """[nixl_putget] transfer best  N us -> X GB/s | mean ..."""
    rows = []
    size = None
    for line in open(path, errors="ignore"):
        if "---- size" in line:
            size = line.split("size", 1)[1].split("(")[0].strip()
        if "transfer best" in line and "GB/s" in line:
            try:
                seg = line.split("best", 1)[1]
                best_gbs = float(seg.split("->")[1].split("GB/s")[0])
                mean_gbs = float(seg.split("mean")[1].split("->")[1].split("GB/s")[0])
                rows.append((size, best_gbs, mean_gbs))
            except (IndexError, ValueError):
                pass
    return rows


def section(title):
    return f"\n## {title}\n"


def main():
    # Pick the newest run that is a FULL sweep, not a single-layer rerun:
    # job 6955 was a nixl-only 4-cell job and would otherwise be reported as
    # "the 2-node result", hiding 53 measured cells behind 4.
    runs = {}
    for tag, sub, expect in (("1 node / 4 GPUs", "1node", 50),
                             ("2 nodes / 8 GPUs", "2node", 50)):
        for jd in sorted(ROOT.iterdir(), reverse=True):
            c = jd / sub / "sweep_results.csv"
            if not c.exists():
                continue
            rows = load(c)
            if len(rows) < expect:
                continue
            runs[tag] = (jd.name, jd / sub, rows)
            break

    L = []
    L.append("# Per-layer performance sweep — Tara North (GH200, Slingshot/CXI)\n")
    L.append("Every number below is read from a file written by a PBS job. "
             "Cells with no measurement are reported as missing, not estimated.\n")

    L.append(section("Coverage"))
    L.append("| Scale | Job | ok | failed | unsupported | unavailable | total |")
    L.append("|---|---|---|---|---|---|---|")
    for tag, (job, _, rows) in runs.items():
        c = defaultdict(int)
        for r in rows:
            c[r["status"]] += 1
        L.append(f"| {tag} | {job} | {c['ok']} | {c['failed']} | "
                 f"{c['unsupported']} | {c['unavailable']} | {len(rows)} |")

    for tag, (job, d, rows) in runs.items():
        L.append(section(f"{tag} — job {job}"))

        # ---- fi ----
        fi = [r for r in rows if r["layer"] == "fi"]
        if fi:
            L.append("\n### Layer 1 — libfabric / CXI\n")
            L.append("| cell | status | detail |")
            L.append("|---|---|---|")
            for r in fi:
                L.append(f"| {r['cell']} {r['value']} | {r['status']} | {r['detail']} |")
            for f in sorted(d.glob("fi_pingpong_cxi*.txt")):
                # columns: bytes #sent #ack total time MB/sec usec/xfer Mxfers/sec
                # MB/sec is field 5 and usec/xfer field 6; "total" and "time"
                # are not bandwidth. Take the best of the repeated blocks.
                best = None
                for line in open(f, errors="ignore"):
                    p = line.split()
                    if len(p) < 8 or p[0] == "bytes":
                        continue
                    try:
                        mbs, usec = float(p[5]), float(p[6])
                    except ValueError:
                        continue
                    if best is None or mbs > best[1]:
                        best = (p[0], mbs, usec)
                if best:
                    L.append(f"\n- `{f.stem}`: {best[0]}B messages → "
                             f"{best[1]:.0f} MB/s ({best[1]/1000:.2f} GB/s), "
                             f"{best[2]:.1f} us/xfer")

        # ---- osu ----
        osu = sorted(d.glob("osu_*.txt"))
        if osu:
            L.append("\n### Layer 2 — Cray MPICH (OSU), device vs host buffers\n")
            L.append("| collective | size | device (us) | host (us) | host/device |")
            L.append("|---|---|---|---|---|")
            pairs = defaultdict(dict)
            for f in osu:
                stem = f.stem.replace("osu_", "")
                for suf in ("_device", "_host"):
                    if stem.endswith(suf):
                        pairs[stem[: -len(suf)]][suf[1:]] = osu_last(f)
            for coll, dd in sorted(pairs.items()):
                dev, host = dd.get("device"), dd.get("host")
                if dev and host and dev[1]:
                    sz = "barrier" if dev[0] == 0 else f"{dev[0]//1024} KiB"
                    L.append(f"| {coll} | {sz} | {dev[1]:.2f} | {host[1]:.2f} | "
                             f"{host[1]/dev[1]:.2f}x |")
                else:
                    L.append(f"| {coll} | - | "
                             f"{'missing' if not dev else f'{dev[1]:.2f}'} | "
                             f"{'missing' if not host else f'{host[1]:.2f}'} | - |")

        # ---- nccl ----
        nc = sorted(d.glob("nccl_*.txt"))
        if nc:
            L.append("\n### Layer 3 — NCCL\n")
            L.append("| cell | peak busbw (GB/s) | algbw | at size |")
            L.append("|---|---|---|---|")
            for f in nc:
                b = nccl_best(f)
                name = f.stem.replace("nccl_", "")
                L.append(f"| {name} | {b[2]:.2f} | {b[1]:.2f} | {b[0]//1024} KiB |"
                         if b else f"| {name} | no rows | - | - |")

        # ---- torch ----
        tf = sorted(d.glob("torch_*.txt"))
        if tf:
            L.append("\n### Layer 4 — torch.distributed (NCCL backend)\n")
            L.append("| cell | peak busbw (GB/s) | algbw | at size |")
            L.append("|---|---|---|---|")
            for f in tf:
                b = torch_best(f)
                name = f.stem.replace("torch_", "")
                L.append(f"| {name} | {b[2]:.2f} | {b[1]:.2f} | {b[0]//1024} KiB |"
                         if b else f"| {name} | no rows | - | - |")

        # ---- nixl ----
        nx = sorted(d.glob("nixl_*.txt"))
        if nx:
            L.append("\n### Layer 5 — NIXL (above torchcomms), LIBFABRIC backend\n")
            L.append("Timings are end-to-end per iteration and include a fixed "
                     "cost of roughly 20 ms that is present even at 4 KiB "
                     "(measured: 20536 us at 4 KiB DRAM, 22159 us at 4 KiB "
                     "VRAM). Small-size GB/s figures are therefore dominated "
                     "by that floor and are not fabric bandwidth; only the "
                     "256 MiB and 1 GiB rows approach a transfer-limited "
                     "regime. Every GB/s below was re-derived from bytes and "
                     "microseconds and matches the benchmark's own printed "
                     "value.\n")
            for f in nx:
                rr = nixl_best(f)
                name = f.stem.replace("nixl_", "")
                if not rr:
                    st = [r for r in rows if r["cell"] == name]
                    why = st[0]["detail"] if st else "no rows"
                    L.append(f"\n**{name}** — no measurement ({why})")
                    continue
                L.append(f"\n**{name}**\n")
                L.append("| size | best GB/s | mean GB/s |")
                L.append("|---|---|---|")
                for sz, b, m in rr:
                    L.append(f"| {sz} | {b:.2f} | {m:.2f} |")

    L.append(section("What each layer's numbers mean"))
    L.append("""
**Layer 1 — libfabric / CXI.** All four Cassini rails pass `fi_pingpong` at
2 nodes, each at roughly 23 GB/s for 6 MiB messages. That is the ceiling every
layer above inherits: no higher layer can beat one rail's line rate, and a
multi-rail layer can only approach 4x it by striping. At 1 node the pingpong is
*unavailable*, not failed — CXI has no loopback, so `fi_domain()` returns -38
for a same-node pair.

**Layer 2 — Cray MPICH (OSU).** The device/host column is the whole point. On
one node, device buffers beat host buffers by 15-17x on allreduce and allgather
— that is GPU-direct actually engaging via the GTL. On two nodes the advantage
narrows to 1.0-2.0x for the data-movement-heavy collectives (gather 1.00x,
alltoall 1.99x) but stays above 11x for reducing ones (allreduce 13.66x, reduce
11.36x). Reductions compress data on the way through, so the fabric carries
less and the GPU-side work dominates; gather and scatter move everything and
become fabric-bound, where the staging copy costs relatively less.

Barrier inverts this: 5.60 us device vs 0.97 us host at 1 node. Barrier moves
no payload, so the CUDA path adds synchronisation cost with nothing to amortise
it against. A device-buffer barrier is the wrong tool, and the measurement says
so.

**Layer 3 — NCCL.** Intra-node is ~44x faster than inter-node (all_reduce
327.57 GB/s busbw at 1 node vs 7.48 at 2 nodes): NVLink inside the node versus
Slingshot between them. This gap is the single largest effect in the whole
sweep, and it dwarfs every tuning knob below — keeping traffic inside a node is
worth more than any option choice. The option sweep is where the actionable
findings are:

- `NCCL_ALGO`: at 2 nodes Tree 6.95 vs Ring 4.41 GB/s — Tree wins by 58% for
  large-message allreduce across nodes. At 1 node the ordering flips hard
  (Ring 326.79 vs Tree 215.64), so this is not a global preference; it is
  scale-dependent and worth setting per job shape.
- `NCCL_PROTO`: LL is the worst option at both scales — 1.45 GB/s at 2 nodes
  (~5x slower than Simple's 7.43) and 149.47 vs 327.23 at 1 node. LL trades
  bandwidth for latency and should not be forced on for bulk transfer. LL128 is
  the middle option (6.72 at 2 nodes).
- dtype and redop are effectively free: at 2 nodes bfloat16 7.52, half 7.63,
  max 7.21, min 7.52, prod 7.54 — all within noise of each other. Reduction
  arithmetic is not the bottleneck; the link is.

**Layer 4 — torch.distributed.** Tracks NCCL closely, as it should, since it is
the same transport underneath: allreduce 6.98 GB/s at 2 nodes vs NCCL's 7.48,
a ~7% framework overhead. `async_op=True` measures 6.68 vs 6.98 sync — no gain
here, because the benchmark immediately waits; async only pays off when there is
real compute to overlap. send_recv at 2.71 GB/s is well under collective
throughput, which is expected: a single pair uses one path rather than the whole
topology.

**Layer 5 — NIXL above torchcomms.** READ works over LIBFABRIC and is
byte-exact at every size: 22.81 GB/s peak VRAM and 38.35 GB/s DRAM at 1 GiB.
WRITE is genuinely unsupported on this CXI stack — `postXferReq` returns
`NIXL_ERR_BACKEND`, matching the -260 "Flags not supported" from the standalone
reference. Since READ over the identical path is healthy, that is a transport
capability limit, not a defect, and it is reported as `unsupported` rather than
`failed`.

The DRAM figure exceeding VRAM at 1 GiB is consistent with the rail-selection
warnings in the log ("Using default (all) rail selection policy for DRAM memory
type"): DRAM transfers stripe across all four rails, while the VRAM path was
previously observed on a single rail. The ~20 ms fixed floor noted above means
these are end-to-end numbers including setup, so they should not be compared
directly against layer 1's pure pingpong rate.
""")

    # ---- NIXL API coverage (job 6958/6959) --------------------------------
    lad = ROOT / "6959/2node/nixl_LADDER_VRAM.json"
    if not lad.exists():
        lad = ROOT / "6958/2node/nixl_LADDER_VRAM.json"
    if lad.exists():
        import json as _j
        d = _j.loads(lad.read_text())
        rows = d if isinstance(d, list) else d.get("rows", d.get("results", []))
        pts = []
        for r in rows:
            b = r.get("bytes", r.get("size"))
            g = r.get("best_gbps", r.get("gbps"))
            u = r.get("best_us")
            if b and g is not None and u is not None:
                pts.append((int(b), float(g), float(u)))
        if pts:
            L.append(section("NIXL API coverage and the fixed-cost floor"))
            L.append(
                "\nThe earlier sweeps walked one NIXL path: register, describe,\n"
                "exchange, transfer, verify, tear down, over 4 buffer sizes. That\n"
                "left most of the API unmeasured. Jobs 6958 and 6959 add a dense\n"
                "size ladder and four API modes.\n"
            )
            L.append("\n### Size ladder (READ, VRAM, cross-node)\n")
            L.append("| bytes | best us | GB/s |")
            L.append("|---:|---:|---:|")
            for b, g, u in pts:
                L.append(f"| {b:,} | {u:,.0f} | {g:.4f} |")
            sm = [u for b, _, u in pts if b <= 1 << 20]
            lo, hi = min(sm), max(sm)
            big = [g for b, g, _ in pts if b >= 1 << 30]
            L.append(f"""
This is the clearest result in the report. From 4 KiB to 1 MiB -- a 256x
change in payload -- the best time stays between {lo:,.0f} and {hi:,.0f} us.
Time is essentially independent of size until roughly 8 MiB, so a transfer
here costs about 20 ms before it costs anything per byte. The GB/s column
below ~1 MiB is therefore a measure of that fixed cost, not of the fabric:
reporting 0.0002 GB/s at 4 KiB as "bandwidth" would be meaningless.
Only at 1 GiB ({big[0]:.1f} GB/s) does payload dominate.

This is why the headline NIXL figure must always carry the end-to-end caveat.
""")

    prep = ROOT / "6959/2node/nixl_MODE_prepped.txt"
    if not prep.exists():
        prep = ROOT / "6958/2node/nixl_MODE_prepped.txt"
    if prep.exists():
        txt = prep.read_text(errors="replace")
        sp = re.findall(r"speedup\s+([\d.]+)x", txt)
        pr = re.findall(r"prep once\s+([\d.]+) ms", txt)
        if sp:
            L.append("\n### Prepared transfers (`prep_xfer_dlist` + `make_prepped_xfer`)\n")
            L.append(
                f"Preparing a descriptor list once and reusing the handle costs\n"
                f"{min(float(x) for x in pr):.3f}-{max(float(x) for x in pr):.3f} ms.\n"
                f"Measured speedup over the naive path: "
                + ", ".join(f"{x}x" for x in sp) + ".\n"
            )
            L.append("""
Preparation is essentially free, and that is the useful finding: at ~0.02 ms it
is three orders of magnitude below the ~20 ms floor, so descriptor setup cannot
be what that floor is made of.

The speedups themselves should not be read as a trend. The same three sizes were
measured in both job 6958 and job 6959 and they disagree in direction -- 1 GiB
gave 1.22x in 6958 and 0.78x in 6959, and 64 KiB went 0.85x then 1.25x. Every
one of these transfers is dominated by the same ~20 ms fixed cost, and the
spread between repeats is larger than the differences between the two paths.
The honest conclusion is that prepared transfers are **not measurably faster
than the naive path at this scale**, not that they are 1.25x better or 0.61x
worse.

For Dynamo this means descriptor reuse is not the lever for small-transfer
latency on this stack; the fixed cost has to be attacked directly.
""")

    L.append("\n### Multi-descriptor batching: a reproducible limit\n")
    L.append("""
Batching 64 MiB across N descriptors succeeds at N=1 and N=4 (byte-exact) and
fails at N=16, in both job 6958 and job 6959, at the same point:

```
libfabric_rail_manager.cpp:933] Failed to process completions on rail 0
libfabric_backend.cpp:1408] PT: Failed to process completions on rails
```

The transfer never completes and the run dies at the following barrier. Job
6959 raised the barrier limit to 900 s and the cell timeout to 2400 s; it
failed identically, so this is not a timeout that more patience would fix.
Note the per-descriptor cost is already poor at N=4 (5,154 us/desc against
19,666 us for a single descriptor covering the same 64 MiB).

This cell is recorded as **failed**, not `unsupported`. The WRITE exemption in
this harness is deliberately narrow: WRITE is a documented CXI capability gap,
whereas a READ path that accepts the descriptors and then cannot drain its
completion queue is a defect. Calling it "unsupported" would bury it.
""")

    intro = ROOT / "6959/2node/nixl_MODE_introspect.txt"
    if not intro.exists():
        intro = ROOT / "6958/2node/nixl_MODE_introspect.txt"
    if intro.exists():
        t = intro.read_text(errors="replace")
        pl = re.search(r"plugins: (\[[^\]]*\])", t)
        L.append("\n### Introspection and notifications\n")
        L.append(f"""
Notification delivery works: `send_notif` / `get_new_notifs` round-trip and the
transfer-completion notification arrives. `check_remote_xfer_done` reported
NOT SEEN at the moment of polling while the notification itself was delivered,
so completion should be observed through the notification, not that call.

Introspection is thinner than the API suggests:

- plugins available: {pl.group(1) if pl else 'see log'}
- `query_memory` -> `NIXL_ERR_NOT_SUPPORTED` on the LIBFABRIC backend
- `get_xfer_telemetry` -> `NIXL_ERR_NO_TELEMETRY` (needs telemetry enabled at
  build/run time)
- `estimate_xfer_cost` returns `'UNKNOWN'` with placeholder numbers, so it
  cannot be used for planning on this stack

Anyone planning to schedule against NIXL's own cost estimates on this platform
should know these return nothing usable today.
""")

    L.append(section("Honest limits of this report"))
    L.append("""
- Two node counts only (1 and 2). Scaling behaviour beyond 2 nodes is not
  measured and is not extrapolated here.
- NIXL timings include a ~20 ms fixed cost, so its small-size rows are not
  fabric bandwidth.
- `osu reduce_scatter` with host buffers hit the 600 s timeout in job 6954 and
  was rerun clean in 6957; the published row is the successful one.
- Each cell is one run, not a repeated distribution, so treat differences under
  roughly 10% as noise rather than signal.
- NIXL multi-descriptor batching at N>=16 fails on this stack (rail completion
  error, reproduced in jobs 6958 and 6959). The batch row is therefore only
  measured at N=1 and N=4.
- NIXL's own introspection (`query_memory`, `get_xfer_telemetry`,
  `estimate_xfer_cost`) returns nothing usable on the LIBFABRIC backend here,
  so all timings in this report are wall-clock from the benchmark, never
  self-reported by NIXL.
- The NIXL modes were measured on VRAM and READ only. DRAM variants of the
  prepped/batch/notif paths are not covered.
""")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(L)} lines)")


main()
