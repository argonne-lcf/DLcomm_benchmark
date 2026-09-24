#!/bin/bash
#PBS -N fabric_layers
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:40:00
#PBS -j oe

# ============================================================================
# Fabric layers: libfabric (FI) and NIXL, on 2 nodes.
#
#   FI    fi_info provider/domain inventory, then fi_pingpong between 2 nodes
#   NIXL  RDMA READ of a registered buffer, host DRAM and device VRAM
#
# Both are point-to-point. Output lands in one directory per run and is read
# by:  python -m dl_comm.analysis.compare_layers <dir> --ranks 2
#
# Platform notes (Tara North, and why these are not optional):
#   * PBS does not provision a Slingshot VNI the way SLURM's plugin does, so
#     every rank runs under mpiexec (PALS), which supplies one. Without it
#     fi_domain fails with "Function not implemented".
#   * FI_CXI_DISABLE_HMEM_DEV_REGISTER=1 is required for GPU memory
#     registration; without it cxil_map fails on the VRAM path.
#   * CXI supports RDMA READ but not WRITE here (fi_writedata returns -260,
#     "Flags not supported"), so the transfer direction is READ.
# ============================================================================

set -u

RESULTS="${RESULTS:-$PBS_O_WORKDIR/results_$PBS_JOBID}"
mkdir -p "$RESULTS"

NODES=$(sort -u "$PBS_NODEFILE")
NNODES=$(echo "$NODES" | wc -l)
echo "nodes ($NNODES):"; echo "$NODES"

if [[ "$NNODES" -lt 2 ]]; then
    echo "need 2 nodes for a cross-node fabric test; got $NNODES" >&2
    exit 1
fi

# --- FI layer: inventory ----------------------------------------------------
# fi_info is not a measurement, it answers "is the CXI provider present and
# how many domains does it expose". An empty result here explains a later
# failure that would otherwise look like a performance problem.
echo "[1/3] fi_info"
fi_info -p cxi > "$RESULTS/fi_info_cxi.txt" 2>&1
FI_RC=$?
echo "  fi_info -p cxi rc=$FI_RC ($(grep -c '^provider:' "$RESULTS/fi_info_cxi.txt" 2>/dev/null) provider blocks)"

# --- FI layer: fi_pingpong --------------------------------------------------
# fi_pingpong is a 2-process server/client test, not an MPI program: the
# server is started on the first node and the client dials it by address.
echo "[2/3] fi_pingpong"
SRV=$(echo "$NODES" | head -1)
CLI=$(echo "$NODES" | tail -1)

if command -v fi_pingpong >/dev/null 2>&1; then
    ssh "$SRV" "fi_pingpong -p cxi" > "$RESULTS/fi_pingpong_server.txt" 2>&1 &
    sleep 5
    ssh "$CLI" "fi_pingpong -p cxi $SRV" > "$RESULTS/fi_pingpong.txt" 2>&1
    echo "  fi_pingpong rc=$?"
    wait
else
    echo "  fi_pingpong not installed; FI layer will report as not measured"
    echo "fi_pingpong not available on this system" \
        > "$RESULTS/fi_pingpong_absent.txt"
fi

# --- NIXL layer -------------------------------------------------------------
# The repro script takes --dram/--vram and writes the bandwidth lines, the
# CXI octet deltas, and the byte-exact verdict that compare_layers reads.
echo "[3/3] NIXL"
: "${NIXL_REPRO:?set NIXL_REPRO to repro_nixl_2rank_transfer.py}"

export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1

for MEM in dram vram; do
    mpiexec -n 2 -ppn 1 --cpu-bind none \
        python3 "$NIXL_REPRO" --op READ --"$MEM" \
        --gib 0.25 --iters 5 --sync-dir "$RESULTS/sync" \
        > "$RESULTS/nixl_$MEM.txt" 2>&1
    echo "  nixl $MEM rc=$?  $(grep -c 'PASS: destination buffer is byte-exact' "$RESULTS/nixl_$MEM.txt") pass line(s)"
done

echo
echo "results: $RESULTS"
echo "compare: python -m dl_comm.analysis.compare_layers $RESULTS --ranks 2"
