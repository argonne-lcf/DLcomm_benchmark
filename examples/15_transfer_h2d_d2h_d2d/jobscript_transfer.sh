#!/bin/bash -x
#PBS -A datascience
#PBS -k doe
#PBS -l select=1:ncpus=208
#PBS -q debug-scaling
#PBS -l walltime=00:20:00
#PBS -l filesystems=flare
#PBS -j oe
#PBS -o /dev/null

# ============================================================================
# Host/device transfer bandwidth: H2D, D2H, D2D and bidirectional.
#
# This measures the SYCL memory-copy layer rather than a collective. It is the
# floor under every other measurement in the benchmark: a collective moving
# device buffers cannot exceed the device's own copy bandwidth, so these
# figures bound the rest.
#
# Source: dl_comm/transfer/pci_fixed.cpp. Records are emitted as LAYER=cpp
# lines and parsed by dl_comm.analysis.parse_layers.
# ============================================================================

module load frameworks

if [[ -n "${PBS_O_WORKDIR:-}" && "${PBS_ENVIRONMENT:-}" == "PBS_BATCH" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SRC="${BASH_SOURCE[0]:-$0}"
    SCRIPT_DIR="$(cd "$(dirname "$SRC")" && pwd -P)"
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
cd "$REPO_ROOT"

# FLAT exposes each tile as its own device, giving 12 per Aurora node.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

RUN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="$SCRIPT_DIR/logs/run_${RUN_TIMESTAMP}"
mkdir -p "$RUN_LOG_DIR"
LOG="$RUN_LOG_DIR/terminal_output.log"

# ----------------------------------------------------------------------------
# Build inside the job, against the module stack it will run on.
#
# pci_fixed.cpp calls MPI_Barrier and MPI_Reduce, so it needs the MPI compiler
# wrapper. Building with bare icpx fails at link time with undefined
# references to MPI_Barrier (job 8824800).
# ----------------------------------------------------------------------------
BIN="$RUN_LOG_DIR/pci_fixed"
mpicxx -cxx=icpx -fsycl -std=c++17 -O2 dl_comm/transfer/pci_fixed.cpp \
    -o "$BIN" 2>&1 | grep -v "^/usr/bin/ld: warning" | tee -a "$LOG"
echo "BUILD_EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"

if [[ ! -x "$BIN" ]]; then
    echo "VERDICT=BUILD_FAILED" | tee -a "$LOG"
    exit 1
fi

# ----------------------------------------------------------------------------
# Run: one rank per tile. The binary takes no arguments; sizes and iteration
# counts are compiled in.
# ----------------------------------------------------------------------------
NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))

mpiexec --np ${NRANKS} \
        -ppn ${RANKS_PER_NODE} \
        --depth 16 \
        --cpu-bind depth \
        --pmi=pmix \
        --envall \
        "$BIN" 2>&1 | grep -E "^(LAYER=|MAP )" | tee -a "$LOG"

EXIT_STATUS=${PIPESTATUS[0]}

# ----------------------------------------------------------------------------
# An exit status of 0 is not evidence that anything was measured. Jobs have
# exited clean after running nothing at all (docs/fixes/09), so require
# records before calling the run good.
# ----------------------------------------------------------------------------
RECORDS=$(grep -c "^LAYER=cpp" "$LOG" 2>/dev/null || true)
echo "TRANSFER_RECORDS=${RECORDS:-0}" | tee -a "$LOG"
if [[ "${RECORDS:-0}" -eq 0 ]]; then
    echo "VERDICT=NO_RECORDS (exit $EXIT_STATUS, nothing measured)" | tee -a "$LOG"
    exit 1
fi

# Each rank must land on its own tile. Ranks sharing a tile silently halve the
# apparent device count and inflate per-device bandwidth (docs/fixes/23).
# Host and device are compared together: dev_idx alone repeats across nodes.
TILES=$(awk '/^MAP /{print $4, $6}' "$LOG" | sort -u | wc -l)
echo "DISTINCT_HOST_DEV=$TILES EXPECTED=$NRANKS" | tee -a "$LOG"
if [[ "$TILES" -ne "$NRANKS" ]]; then
    echo "VERDICT=TILE_COLLAPSE (ranks sharing tiles; numbers not trustworthy)" | tee -a "$LOG"
    exit 1
fi

echo "VERDICT=OK" | tee -a "$LOG"
exit $EXIT_STATUS
