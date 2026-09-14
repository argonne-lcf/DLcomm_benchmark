# Example 15 — host/device transfer bandwidth

Measures H2D, D2H, D2D and bidirectional memory-copy bandwidth through SYCL.
One Aurora node, 12 XPU ranks, one rank per tile.

## Contents

| File | Purpose |
|---|---|
| `jobscript_transfer.sh` | builds and runs `dl_comm/transfer/pci_fixed.cpp` |

## Running

```
qsub jobscript_transfer.sh
```

Results are written to `logs/run_<timestamp>/`.

## What this layer is for

Transfer bandwidth is the floor under every other measurement in the
benchmark. A collective operating on device buffers cannot move data faster
than the device can copy it, so these figures bound the collective results and
make an implausible collective number visible.

There is no YAML configuration. The binary is built and launched directly
rather than through `dl_comm.dl_comm_main`, and its sizes and iteration counts
are compiled in.

## Build note

`pci_fixed.cpp` calls `MPI_Barrier` and `MPI_Reduce`, so it is compiled with
the MPI wrapper (`mpicxx -cxx=icpx -fsycl`). Building with bare `icpx` fails at
link time with undefined references to `MPI_Barrier`, which is what job
8824800 hit.

The binary is compiled inside the job rather than committed, so it is always
built against the module stack it will run on.

## Verification gates

The jobscript does not treat a zero exit status as success. Two checks run
after the job:

- **Record count.** A run that exits clean having measured nothing fails with
  `VERDICT=NO_RECORDS`. Jobs have exited zero after running no work at all;
  see `docs/fixes/09`.
- **Tile mapping.** Every rank must occupy its own tile. Ranks sharing a tile
  halve the effective device count and inflate per-device bandwidth. The check
  compares host and device index together, since a device index alone repeats
  across nodes. A collapse fails with `VERDICT=TILE_COLLAPSE`; see
  `docs/fixes/23`.
