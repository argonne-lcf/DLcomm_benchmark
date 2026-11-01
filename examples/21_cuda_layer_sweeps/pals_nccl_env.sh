#!/bin/bash
# Rank-to-GPU wrapper for nccl-tests under PALS.
#
# nccl-tests with `-g 1` uses one GPU per rank and picks it from the MPI local
# rank. PALS does not export the variables it looks for, so they are published
# here.
export OMPI_COMM_WORLD_LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"
export MPI_LOCALRANKID="${PALS_LOCAL_RANKID:-0}"
export LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"

# Do NOT mask CUDA_VISIBLE_DEVICES here. nccl-tests with `-g 1` selects its
# device from the MPI local rank against the FULL device list; masking to a
# single device made it request rank+1 GPUs from a 1-device view and abort
# with "Invalid number of GPUs: 4 requested but only 1 were found".
# Leaving all devices visible lets each rank bind to its own GPU.

# NCCL over Slingshot: let NCCL use the CXI interfaces rather than falling back
# to a management NIC, which silently produces a fraction of the bandwidth.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

exec "$@"
