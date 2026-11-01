#!/bin/bash
# Rank-to-GPU wrapper for OSU under PALS.
#
# OSU probes these four variables to learn its local rank, and PALS sets none
# of them. Without this every rank selects device 0: four ranks contend on one
# GPU and the numbers are meaningless, or it segfaults in a way that mimics a
# missing GTL. The warning to watch for is:
#     "OMB could not identify the local rank of the process"
# Treat that as fatal for device runs, not cosmetic.
export MPI_LOCALRANKID="${PALS_LOCAL_RANKID:-0}"
export MV2_COMM_WORLD_LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"
export MVP_COMM_WORLD_LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"
export OMPI_COMM_WORLD_LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"
export LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"

# Deliberately NOT exporting CUDA_VISIBLE_DEVICES here: OSU selects its device
# from the local-rank variables above, and masking the device list would make
# every rank see a single device numbered 0, re-creating the bug this wrapper
# exists to fix.
exec "$@"
