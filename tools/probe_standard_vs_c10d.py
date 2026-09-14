"""Compare torchcomms' native TorchComm API against c10d for reductions.

Every rank contributes 1.0 except rank 3, which contributes 0.0, so each
reduction has a distinct expected value and a wrong operator cannot
coincidentally produce the right answer:

    SUM = world - 1,  MIN = 0,  MAX = 1,  PRODUCT = 0

Each op runs twice on identical input -- once on the native ``TorchComm``
obtained from ``new_comm``, once through ``torch.distributed`` with the XCCL
backend -- so a discrepancy localises the fault to one of the two paths.

The script also checks that the ``ReduceOp`` members are mutually distinct,
holding all references simultaneously. Fetching them one at a time and
comparing addresses is meaningless: each temporary is freed before the next is
created and pybind11 reuses the address, which previously led to a false report
that SUM, MIN, MAX and AVG were the same object.

Run under mpiexec with RANK and WORLD_SIZE set (see the companion .sh).
Exit status is unreliable -- this stack has a teardown race documented in
docs/torchcomms-stack.md. Read the printed table.
"""

import os

import torch
import torchcomms as tc

OPS = ["SUM", "MIN", "MAX", "PRODUCT"]
ZERO_RANK = 3


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(0)
    dev = torch.device("xpu:0")

    expected = {
        "SUM": float(world - 1),
        "MIN": 0.0,
        "MAX": 1.0,
        "PRODUCT": 0.0,
    }

    def payload():
        t = torch.ones(1, dtype=torch.float32, device=dev)
        if rank == ZERO_RANK:
            t.zero_()
        return t

    # 1. ReduceOp members must be mutually distinct.
    if rank == 0:
        held = {n: getattr(tc.ReduceOp, n) for n in OPS}   # all live at once
        print("=== ReduceOp member identity (references held simultaneously) ===")
        for a in OPS:
            collisions = [b for b in OPS if b != a and held[a] == held[b]]
            print(f"  {a:8s} collides with {collisions if collisions else '[]'}")

    # 2. Standard path: native TorchComm.
    comm = tc.new_comm("xccl", dev, "standard_vs_c10d")
    standard = {}
    for name in OPS:
        t = payload()
        comm.all_reduce(t, getattr(tc.ReduceOp, name), False)
        torch.xpu.synchronize()
        standard[name] = t.item()

    # 3. c10d path: torch.distributed over XCCL.
    import torch.distributed as dist

    c10d = {}
    c10d_error = None
    try:
        dist.init_process_group(backend="xccl", rank=rank, world_size=world)
        for name in OPS:
            t = payload()
            dist.all_reduce(t, op=getattr(dist.ReduceOp, name))
            torch.xpu.synchronize()
            c10d[name] = t.item()
    except Exception as exc:  # noqa: BLE001 - report, do not mask
        c10d_error = f"{type(exc).__name__}: {exc}"

    if rank != 0:
        return 0

    print(f"\n=== reductions, world={world}, rank {ZERO_RANK} contributes 0.0 ===")
    print(f"  {'op':8s} {'expect':>9s} {'STANDARD':>12s} {'C10D':>12s}   verdict")
    bad = 0
    for name in OPS:
        want = expected[name]
        got_s = standard.get(name)
        got_c = c10d.get(name)
        ok_s = got_s is not None and abs(got_s - want) < 1e-6
        ok_c = got_c is not None and abs(got_c - want) < 1e-6
        if not ok_s or (c10d_error is None and not ok_c):
            bad += 1
        shown_c = "n/a" if got_c is None else f"{got_c:12.2f}"
        verdict = "ok" if (ok_s and (ok_c or c10d_error)) else "MISMATCH"
        print(f"  {name:8s} {want:9.2f} {got_s:12.2f} {shown_c:>12s}   {verdict}")

    if c10d_error:
        print(f"\n  c10d path unavailable: {c10d_error}")
    print(f"\n  {'PASS' if bad == 0 else 'FAIL'}: {len(OPS) - bad}/{len(OPS)} ops correct")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
