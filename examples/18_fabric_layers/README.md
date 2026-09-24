# Example 18 — fabric layers (libfabric / NIXL)

Measures the two layers *below* the collectives stack on a Slingshot/CXI
machine, and feeds them to the same cross-layer comparison as everything else.

    libfabric (FI)   provider + domain discovery, fi_pingpong
    NIXL             RDMA READ of a registered buffer, DRAM and VRAM

These are point-to-point, not collectives, so they are recorded as their own
pseudo-collectives (`pingpong`, `read_dram`, `read_vram`) and are never
divided against an allreduce number.

## Contents

| File | Purpose |
|---|---|
| `jobscript_fabric_layers.sh` | 2-node PBS job: FI probes, then NIXL DRAM + VRAM |

## Running

```
qsub jobscript_fabric_layers.sh
python -m dl_comm.analysis.compare_layers <results-dir> --ranks 2
```

## Why bandwidth alone is not the result

A NIXL transfer that silently falls back to a host memcpy still prints a
plausible GB/s, and still passes a byte-exact check — the bytes do arrive,
they just never crossed the NIC. Bandwidth and correctness together do not
distinguish that case from a real RDMA transfer.

The CXI hardware octet counters do. The comparison prints them beside every
transfer:

```
fabric evidence (NIXL transfers)
file                       rails   NIC/payload    byte-exact
nixl_dram.txt                  4         1.02x          PASS
nixl_vram.txt                  1         1.02x          PASS
```

`NIC/payload` is the ratio of octets the NIC actually moved to the payload
size. At or above 1.00x the bytes reached the wire and the excess is protocol
overhead; near zero means a fallback. A run with no counter sample prints
`not sampled` and the byte-exact column prints `unverified` — unknown is never
reported as a pass.

## Measured on Tara North

2 × GH200 nodes, libfabric 2.8.0a1 (CXI), NIXL 1.0.0, 0.25 GiB, 5 iterations:

| Path | best | mean | rails | NIC/payload | byte-exact |
|---|---|---|---|---|---|
| DRAM READ | 8.39 GB/s | 7.63 | 4 (cxi0–3) | 1.02x | PASS |
| VRAM READ | 16.20 GB/s | 10.18 | 1 (cxi0) | 1.02x | PASS |

VRAM uses one rail because GPU→NIC affinity pins a device to its local NIC;
lighting all four requires four GPUs driving traffic concurrently. DRAM has no
such affinity and spreads across all four. This is why the rails column is
printed rather than assumed — a 1-rail result is correct for VRAM and a
symptom for DRAM.

## Where these layers sit in the stack

    libfabric (FI)      <- bottom: the raw fabric API
    OSU / MPI
    C++ CCL
    torch.distributed
    torchcomms
    NIXL                <- top

NIXL is at the top, above torchcomms, not down next to libfabric. It is not a
lower-level transport the collective stack is built on — it is a *separate
consumer* of the same fabric, used by inference serving (Dynamo/vLLM KV
transfer) the way a training job uses torchcomms. Putting it below MPI would
imply the collectives run on top of it, which is false, and would split the
training path in half.

The consequence in a report: a torchcomms → NIXL gap is labelled "different
consumer of the fabric, not a subset" rather than being attributed to an
overhead component, because the difference between them is not something
torchcomms lost.

## Ordering caveat

DRAM and VRAM results use different memory, so the comparison refuses to
compute a ratio between them. That is the same guard the collective layers
use, and it is why VRAM's higher number is not reported as "VRAM is 1.9x
faster than DRAM": they are not the same measurement.

## The nodefile order is load-bearing

PALS assigns **rank 0 to the first line of `$PBS_NODEFILE` as PBS wrote it**,
and PBS does not write that file sorted. Job 6935 was allocated:

```
x4820c7s6b1n0     <- rank 0 lands here
x4820c7s0b0n0
```

`sort -u | head -1` returns `s0b0n0` — the wrong host. Any server address
predicted that way points at a node where nothing is listening, and the
failure surfaces as `Connection refused`, which looks like a fabric fault.
That cost jobs 6931 and 6932.

Proven in job 6935, same allocation, same job:

| Server address predicted from | Result |
|---|---|
| `sort -u` -> `s0b0n0` | `Connection refused` |
| natural order -> `s6b1n0` | **18.77 GB/s** |
| natural order + explicit `--hostfile` | **19.75 GB/s** |

Use `head -1 "$PBS_NODEFILE"` or `awk '!seen[$0]++'`, never `sort -u`. Better
still, pass an explicit `--hostfile` so placement and prediction come from the
same list — job 6935 confirms `--hostfile` does control which host receives
rank 0.

`tests/test_nodefile_ordering.py` enforces this against both job scripts.

## Why NIXL needs no such care, and fi_pingpong does

NIXL finds its peer through `--sync-dir`, a file rendezvous, so it never
predicts a hostname and is immune to the trap above. `fi_pingpong` takes the
server address on the command line and has no rendezvous mechanism, which is
why it was the tool that exposed the bug.
