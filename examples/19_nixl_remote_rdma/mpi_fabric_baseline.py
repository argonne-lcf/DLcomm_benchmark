#!/usr/bin/env python3
"""MPI cross-node bandwidth baseline for the NIXL remote example.

Run before the NIXL measurement, in the same allocation, between the same
nodes. Without it a slow NIXL number is unattributable: fabric, placement,
and NIXL's own transport selection all produce the same symptom. With it,
the comparison is direct.

A separate file rather than a heredoc piped to mpiexec on purpose -- mpiexec
forwards stdin to rank 0 only, so `python3 -` leaves higher ranks with an
empty script and the job hangs to walltime.
"""
from __future__ import annotations

import socket
import time

import numpy as np
from mpi4py import MPI

SIZES = (1 << 20, 1 << 24, 1 << 26)


def main() -> int:
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    if world < 2:
        if rank == 0:
            print("MPI_BASELINE_SKIP needs >= 2 ranks", flush=True)
        return 1

    hosts = comm.allgather(socket.gethostname())
    if rank == 0 and hosts[0] == hosts[1]:
        # Same-node MPI is a shared-memory copy, not a fabric number. Saying
        # so is the difference between a baseline and a misleading one.
        print(f"MPI_BASELINE_WARN ranks 0 and 1 share host {hosts[0]}; "
              "this measures shared memory, not the fabric", flush=True)

    for nbytes in SIZES:
        buf = np.ones(nbytes // 4, dtype=np.int32)
        rbuf = np.empty_like(buf)
        times = []
        for i in range(12):
            comm.Barrier()
            t0 = time.monotonic()
            if rank == 0:
                comm.Send([buf, MPI.INT], dest=1, tag=7)
                comm.Recv([rbuf, MPI.INT], source=1, tag=8)
            elif rank == 1:
                comm.Recv([rbuf, MPI.INT], source=0, tag=7)
                comm.Send([buf, MPI.INT], dest=0, tag=8)
            elapsed = time.monotonic() - t0
            if i >= 4:
                times.append(elapsed)
        if rank == 0:
            best = min(times)
            # Round trip moves nbytes each way; one-way time is best/2.
            print(f"MPI_BASELINE bytes={nbytes} rt_best_s={best:.6f} "
                  f"oneway_GB/s={nbytes / (best / 2) / 1e9:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
