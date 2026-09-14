#!/bin/bash -x
#PBS -A datascience
#PBS -k doe
#PBS -l select=2:ncpus=208
#PBS -q debug-scaling
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare
#PBS -j oe
#PBS -o /dev/null

# ============================================================================
# All five layers at 1 node / 12 ranks and 2 nodes / 24 ranks.
#
#   OSU / MPI            host buffers, MPI collectives
#   C++ oneCCL           device buffers, CCL called directly from C++
#   C++ SYCL transfer    H2D / D2H / D2D memory copies
#   torch.distributed    device buffers, XCCL through PyTorch
#   torchcomms           device buffers, XCCL through the torchcomms API
#
# This is a thin wrapper around tools/run_all_scales.sh, which is the script
# that produced the published numbers. The wrapper exists so the example is
# reproducible with a single qsub; the harness itself is not duplicated here,
# because a copy would drift from the original.
#
# torchcomms stack selection:
#   default          the torchcomms shipped with frameworks/2025.3.1 (0.1.0),
#                    which implements only all_reduce and stubs the rest
#   pshukla          the 0.3.0 build, which implements 12 of 12 probed ops
#
# Select with: qsub -v DLCOMM_TC_STACK=pshukla jobscript_all_layers.sh
# ============================================================================

if [[ -n "${PBS_O_WORKDIR:-}" && "${PBS_ENVIRONMENT:-}" == "PBS_BATCH" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SRC="${BASH_SOURCE[0]:-$0}"
    SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd -P)"
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
HARNESS="$REPO_ROOT/tools/run_all_scales.sh"

if [[ ! -f "$HARNESS" ]]; then
    echo "harness not found: $HARNESS"
    exit 1
fi

# The harness discovers its own scales, builds each layer inside the job, and
# writes per-scale directories under validation/<timestamp>/.
exec bash "$HARNESS"
