#!/bin/bash
# Translate PALS rank variables into what torch.distributed expects.
#
# PALS exports no global world size -- PALS_LOCAL_SIZE is per-node -- so
# WORLD_SIZE is derived as PALS_LOCAL_SIZE x DLCOMM_NNODES, supplied by the
# caller.
#
# MASTER_ADDR must come from the nodefile in its NATURAL order: PALS puts rank
# 0 on the first line as PBS wrote it, and PBS does not write it sorted (job
# 6935). A sorted MASTER_ADDR points at a node where no rendezvous is
# listening and init_process_group hangs until timeout.
export RANK="${PALS_RANKID:?}"
export LOCAL_RANK="${PALS_LOCAL_RANKID:?}"
export WORLD_SIZE=$(( ${PALS_LOCAL_SIZE:?} * ${DLCOMM_NNODES:?} ))
export MASTER_ADDR="${MASTER_ADDR:?}"
export MASTER_PORT="${MASTER_PORT:-29531}"

# One GPU per rank: expose only this rank's device so a stray .cuda() cannot
# land on a device another rank owns.
export CUDA_VISIBLE_DEVICES="$LOCAL_RANK"
# LOCAL_RANK indexes into the visible set, which is now a single device.
export LOCAL_RANK=0

exec "$@"
