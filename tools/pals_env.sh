#!/bin/bash
# Translate Aurora's PALS launcher variables into the rank/size variables that
# torch.distributed and torchcomms expect, then exec the real command.
#
# Job 8824786 dumped the actual compute-node environment under
# /opt/cray/pals/1.8/bin/mpiexec. PALS sets:
#
#     PALS_RANKID        global rank          (0..N-1)
#     PALS_LOCAL_RANKID  rank within the node (0..ppn-1)
#     PALS_LOCAL_SIZE    ranks per node
#     PALS_NODEID        node index
#
# and it sets NO global size variable. torchcomms' query_ranksize() looks for
# PALS_SIZE first and its own source comments that PALS does not provide one,
# falling back to PMI_SIZE (MPICH only) and WORLD_SIZE. Under PALS all three
# are unset, so every rank saw world=1 and XCCL bootstrap segfaulted
# (8824754, 8824771) or aborted on the guard (8824781).
#
# The global size is derivable: ranks-per-node times node count. DLCOMM_NNODES
# must be exported by the caller, which knows it from PBS_NODEFILE.
set -u

: "${PALS_RANKID:?not running under the PALS launcher}"
: "${PALS_LOCAL_SIZE:?PALS_LOCAL_SIZE unset}"
: "${DLCOMM_NNODES:?caller must export DLCOMM_NNODES from PBS_NODEFILE}"

export RANK="$PALS_RANKID"
export LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"
export WORLD_SIZE="$(( PALS_LOCAL_SIZE * DLCOMM_NNODES ))"

# Do NOT set PMI_RANK/PMI_SIZE here. MPICH reads the PMI_* namespace for its
# own bootstrap, so overwriting it corrupts MPI_Init and every MPI-backed
# stack aborts with:
#
#     Abort(16): Fatal error in internal_Init_thread: Internal MPI error!
#
# Probe 8824999 showed bare mpi4py initializing cleanly under this launcher
# (4/4 ranks), while probe 8825011 aborted in all three cases that went
# through this wrapper -- with mpi4py, without it, and with or without
# CCL_ATL_TRANSPORT=mpi. The wrapper was the common factor, not the stack.
#
# PALS_SIZE is safe: it is the variable PALS itself omits, and torchcomms
# reads it directly.
export PALS_SIZE="$WORLD_SIZE"

# Rendezvous. MASTER_ADDR must be the same host on every rank, so the caller
# supplies it; falling back to the local hostname would give each node a
# different master and hang.
: "${MASTER_ADDR:?caller must export MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT:-29522}"

if [ "$RANK" = "0" ]; then
    echo "pals_env: world=$WORLD_SIZE ppn=$PALS_LOCAL_SIZE nodes=$DLCOMM_NNODES" \
         "master=$MASTER_ADDR:$MASTER_PORT" >&2
fi

exec "$@"
