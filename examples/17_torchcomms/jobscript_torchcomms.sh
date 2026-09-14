#!/bin/bash -x
#PBS -A datascience
#PBS -k doe
#PBS -l select=1:ncpus=208
#PBS -q debug-scaling
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare
#PBS -j oe
#PBS -o /dev/null

# ============================================================================
# torchcomms through the standard DLcomm runner, 1 node / 12 XPU ranks.
#
# Unlike example 16, which drives the standalone comparison harness, this runs
# torchcomms through dl_comm.dl_comm_main with a YAML configuration -- the same
# entry point as every other example. The backend is chosen in the YAML with
# ccl_backend: torchcomms.
#
# Stack selection (the single most important variable here):
#
#   DLCOMM_TC_STACK=frameworks   torchcomms 0.1.0 from frameworks/2025.3.1.
#                                Implements all_reduce; the rest raise
#                                "XCCL <op> is not supported now and will be
#                                added later". Expect 1 of 5 sections to pass.
#
#   DLCOMM_TC_STACK=pshukla      torchcomms 0.3.0 paired with the torch build
#                                it was compiled against. 12 of 12 probed ops
#                                work. Expect all 5 sections to pass.
#
# Submit as:
#   qsub jobscript_torchcomms.sh                          # 0.1.0, default
#   qsub -v DLCOMM_TC_STACK=pshukla jobscript_torchcomms.sh   # 0.3.0
#
# The two components of a stack must be used as a matched pair. Mixing a
# custom torch with an independently built torchcomms produces a segfault
# inside new_comm, which is an ABI mismatch and not a configuration error;
# see docs/fixes/26.
# ============================================================================

module load frameworks

if [[ -n "${PBS_O_WORKDIR:-}" && "${PBS_ENVIRONMENT:-}" == "PBS_BATCH" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SRC="${BASH_SOURCE[0]:-$0}"
    SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd -P)"
fi

EXAMPLES_DIR="$SCRIPT_DIR/.."
WORKDIR="$EXAMPLES_DIR"
cd "$WORKDIR"

# ----------------------------------------------------------------------------
# Stack selection
# ----------------------------------------------------------------------------
TC_STACK="${DLCOMM_TC_STACK:-frameworks}"
PSHUKLA_TORCH=/lus/flare/projects/datascience_collab/pshukla/pytorch_c10d_torchcomms/pytorch
PSHUKLA_TC=/lus/flare/projects/datascience_collab/pshukla/torchcomms_custom_torch

TC_PYTHONPATH=""
TC_LDPATH=""
if [[ "$TC_STACK" == "pshukla" ]]; then
    if [[ ! -d "$PSHUKLA_TC" || ! -d "$PSHUKLA_TORCH" ]]; then
        echo "VERDICT=STACK_MISSING ($TC_STACK paths not readable)"
        exit 1
    fi
    TC_PYTHONPATH="$PSHUKLA_TC:$PSHUKLA_TORCH"
    TC_LDPATH="$PSHUKLA_TORCH/torch/lib:"
fi
echo "TORCHCOMMS_STACK=$TC_STACK"

# ----------------------------------------------------------------------------
# Environment. CCL_ATL_TRANSPORT=mpi is required on Aurora; ofi does not work
# with this stack. FLAT hierarchy exposes each tile as its own device.
# ----------------------------------------------------------------------------
export PYTHONPATH="${TC_PYTHONPATH:+$TC_PYTHONPATH:}$WORKDIR/..:$PYTHONPATH"
export LD_LIBRARY_PATH="${TC_LDPATH}${LD_LIBRARY_PATH:-}"
export CCL_ATL_TRANSPORT=mpi
export CCL_ATL_SHM=0
export CCL_PROCESS_LAUNCHER=pmix
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export CCL_ENABLE_AUTO_CACHE=0
export TORCH_CPP_LOG_LEVEL=error
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=30
export PALS_PMI=pmix
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# A stub backend refuses an operation rather than hanging, but the 0.3.0
# communicator split path does hang at 12 ranks, so the watchdog stays on.
export DLCOMM_WATCHDOG=900
export DLCOMM_VERDICT_TIMEOUT=120

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
export RUN_LOG_DIR
export TERMINAL_LOG_FILE="$RUN_LOG_DIR/terminal_output.log"
export DL_COMM_LOG_DIR="$RUN_LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

CONFIG_NAME="17_torchcomms_xccl"

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# Report which torchcomms actually loaded, before measuring anything. A run
# that silently picks up a different build than intended is the failure this
# guards against.
python3 -c "
import torchcomms, torch
print('TC_VERSION=' + getattr(torchcomms, '__version__', 'unknown'))
print('TORCH_VERSION=' + torch.__version__)
print('TC_PATH=' + torchcomms.__file__)
" 2>&1 | tee -a "$TERMINAL_LOG_FILE"

mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --depth 16 \
        --cpu-bind depth \
        --pmi=pmix \
        --envall \
        python3 -m dl_comm.dl_comm_main --config-path="$SCRIPT_DIR" --config-name="$CONFIG_NAME" 2>&1 | tee -a "$TERMINAL_LOG_FILE"

EXIT_STATUS=${PIPESTATUS[0]}
exit $EXIT_STATUS
