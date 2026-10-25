#!/bin/bash
#PBS -N dlcomm_nixl_remote
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q workq
# -----------------------------------------------------------------------------
# Cross-node NIXL RDMA: one rank writes directly into a remote rank's VRAM.
#
# Submit:
#   qsub -A <account> -l select=2 jobscript_nixl_remote.sh
#   qsub -A <account> -l select=4 jobscript_nixl_remote.sh
#
# Why this exists next to 18_nixl_transfer: that example measures NIXL between
# buffers on ONE node. Every transfer there is node-local, so a 4-node run is
# just 4 independent node-local sets and the fabric never appears. This example
# is the one that crosses it -- the transfer a disaggregated serving job
# performs when a prefill node hands KV cache to a decode node.
#
# CHOOSING THE BACKEND IS THE WHOLE BALLGAME (measured on Tara, Slingshot +
# GH200, 2026-10-24). NIXL registers several backends; which one you get
# decides whether the transfer touches the NIC at all:
#
#   LIBFABRIC  -> CXI provider, real RDMA. ~11 GB/s cross-node.
#   UCX        -> on Slingshot, UCX's bundled transports (ib/mlx5/efa/cma/cuda)
#                 cannot drive a cxi NIC, so it silently falls back to TCP over
#                 the hsn IP interfaces: ~0.2-0.7 GB/s, every byte still
#                 correct. A wrong answer that passes every gate.
#
# LIBFABRIC refuses to construct without the FI_* settings below, and the CXI
# provider additionally needs a Slingshot VNI, which PALS only provisions when
# the allocation requests GPUs. Submit with `-l select=N:ngpus=4`; a plain
# `-l select=N` yields SLINGSHOT_VNIS unset and agent creation dies with
# "Failed to find valid default rgroup and vni for cxi0" -- which reads like a
# NIXL bug and is really a missing resource request.
#
# Submit:
#   qsub -A <account> -l select=2:ngpus=4 jobscript_nixl_remote.sh
#   qsub -A <account> -l select=4:ngpus=4 jobscript_nixl_remote.sh
#
# Step 1 below runs the MPI baseline every time so the comparison lives in the
# log rather than in someone's memory.
# -----------------------------------------------------------------------------
set -uo pipefail

PROJ="${PROJ:-$PWD}"
REPO="${REPO:-$PROJ/DLcomm_benchmark}"

# Activate the project venv unless the caller already supplied an interpreter.
# PBS starts a login shell whose `python3` on this system is old enough to
# reject `from __future__ import annotations`, so every rank dies with a
# SyntaxError before importing anything -- a failure that looks like broken
# code rather than a missing venv. VENV defaults to the layout used by the
# other examples; override it or PYBIN if yours differs.
VENV="${VENV:-$PROJ/tcvenv}"
if [[ -z "${PYBIN:-}" ]]; then
    if [[ -x "$VENV/bin/python3" ]]; then
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
        PYBIN="$VENV/bin/python3"
    else
        PYBIN="python3"
    fi
fi

# Fail loudly and early rather than letting every rank emit the same
# SyntaxError, which buries the real cause under 8 identical tracebacks.
if ! "$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "VERDICT=BAD_PYTHON ($PYBIN is $("$PYBIN" --version 2>&1); need >= 3.8 " \
         "-- set VENV=/path/to/venv or PYBIN=/path/to/python3)"
    exit 1
fi
RESULTS="${RESULTS:-$PROJ/results/nixl_remote_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS"
LOG="$RESULTS/nixl_remote.log"

# grep -c . not wc -l: a PBS nodefile with no trailing newline makes wc -l
# return nodes-1, and 0 for a single node, which then launches nothing.
if [ -n "${PBS_NODEFILE:-}" ]; then
    # grep -c . not `wc -l`: PBS nodefiles on some systems (Tara) have no
    # trailing newline, so wc -l undercounts by one and a 1-node allocation
    # builds `mpiexec --np 0`, which aborts with "Invalid -n/--np value 0".
    # NOTE: grep -c prints the count AND exits 1 when the count is 0, so a
    # `|| echo 1` fallback would emit two lines. Let it print 0 and let the
    # guard below reject it. See tools/nodefile.sh for the measurement.
    NNODES=$(grep -c . "$PBS_NODEFILE" 2>/dev/null || true)
    case "${NNODES:-}" in ""|*[!0-9]*) NNODES=0 ;; esac
    if [ "$NNODES" -lt 1 ]; then
        echo "FATAL: counted $NNODES nodes in $PBS_NODEFILE" >&2; exit 1
    fi
else
    NNODES=1
fi
[[ "$NNODES" -lt 1 ]] && NNODES=1
RANKS_PER_NODE="${RANKS_PER_NODE:-4}"
NRANKS=$(( NNODES * RANKS_PER_NODE ))

{
  echo "NODES=$NNODES RANKS_PER_NODE=$RANKS_PER_NODE NRANKS=$NRANKS"
  echo "REPO=$REPO"
  echo "DATE=$(date -Is)"
  echo "NODELIST=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | tr '\n' ' ')"
} | tee "$RESULTS/environment.txt"

if [[ "$NNODES" -lt 2 ]]; then
    echo "VERDICT=SINGLE_NODE (this example measures the fabric; submit -l select=2 or more)" | tee -a "$LOG"
    exit 1
fi

# EXTRA_PYTHONPATH: a NIXL venv built for fabric work often has no mpi4py,
# which this example needs only for the out-of-band peer exchange (NIXL has
# no bootstrap of its own). Rather than forcing it into that venv, point at a
# `pip install --target` directory. Without it the whole run exits 0 per rank
# after printing NIXL_REMOTE_SKIP, and the gates correctly report NO_RECORDS.
export PYTHONPATH="${PYTHONPATH:-}:$REPO${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"

# ---------------------------------------------------------------------------
# Slingshot/CXI fabric configuration for the LIBFABRIC backend.
#
# Every one of these is load-bearing; without them nixl_agent(...) raises
# NIXL_ERR_BACKEND before a single byte moves. Site-overridable, because none
# of it applies on an IB/RoCE cluster -- set NIXL_FABRIC=ucx there (or just
# leave CUSTOM_LIBFABRIC unset) and the block is skipped.
#
# CUSTOM_LIBFABRIC should point at a CXI-capable libfabric. The system
# /opt/cray/libfabric also works; a locally built shs-libfabric is used here
# because it is what this stack was validated against.
# ---------------------------------------------------------------------------
NIXL_FABRIC="${NIXL_FABRIC:-cxi}"
if [[ "$NIXL_FABRIC" == "cxi" ]]; then
    CUSTOM_LIBFABRIC="${CUSTOM_LIBFABRIC:-/opt/cray/libfabric/2.3.1}"
    if [[ -d "$CUSTOM_LIBFABRIC/lib64" ]]; then
        export LD_LIBRARY_PATH="$CUSTOM_LIBFABRIC/lib64:${LD_LIBRARY_PATH:-}"
    elif [[ -d "$CUSTOM_LIBFABRIC/lib" ]]; then
        export LD_LIBRARY_PATH="$CUSTOM_LIBFABRIC/lib:${LD_LIBRARY_PATH:-}"
    fi
    # NIXL_INSTALL_DIR: a source-built NIXL keeps libnixl.so and its plugins
    # outside site-packages, so they must be added explicitly. Do NOT mix a
    # source build's libfabric with the pip wheel's bundled plugin: the agent
    # still constructs, then every transfer fails with NIXL_ERR_BACKEND at
    # getXferStatus. Keep libfabric, libnixl and the plugins from ONE build.
    if [[ -n "${NIXL_INSTALL_DIR:-}" ]]; then
        export LD_LIBRARY_PATH="$NIXL_INSTALL_DIR/lib64:$NIXL_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"
        [[ -z "${NIXL_PLUGIN_DIR:-}" && -d "$NIXL_INSTALL_DIR/lib64/plugins" ]] && \
            export NIXL_PLUGIN_DIR="$NIXL_INSTALL_DIR/lib64/plugins"
    fi
    # NIXL_PLUGIN_DIR: without it the agent comes up with zero backends and
    # the LIBFABRIC request fails at construction. Resolve it from whichever
    # NIXL is actually importable rather than hardcoding a wheel layout.
    if [[ -z "${NIXL_PLUGIN_DIR:-}" ]]; then
        _plug=$("$PYBIN" - <<'PY' 2>/dev/null
import pathlib, importlib.util
spec = importlib.util.find_spec("nixl")
if spec and spec.origin:
    root = pathlib.Path(spec.origin).resolve().parent.parent
    for c in root.rglob("libplugin_LIBFABRIC.so"):
        print(c.parent); break
PY
)
        [[ -n "$_plug" ]] && export NIXL_PLUGIN_DIR="$_plug"
    fi
    # CUDA runtime on LD_LIBRARY_PATH. `module load cuda` is not enough when
    # the job runs under a venv that was built against a different CUDA, and
    # the failure is doubly confusing: mpi4py/cupy raise a plain
    # "libcudart.so.12: cannot open shared object file", while NIXL's
    # LIBFABRIC backend instead reports
    #   prepareAndSubmitTransfer failed for descriptor 0 device N
    # because it cannot register device memory it has no runtime for. That
    # message points at the fabric and the real cause is a loader path.
    if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]]; then
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    else
        for _c in /opt/nvidia/hpc_sdk/Linux_aarch64/*/cuda/*/lib64 \
                  /usr/local/cuda/lib64; do
            [[ -f "$_c/libcudart.so.12" ]] && { export LD_LIBRARY_PATH="$_c:${LD_LIBRARY_PATH:-}"; break; }
        done
    fi

    export FI_PROVIDER="${FI_PROVIDER:-cxi}"
    export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
    export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
    export FI_HMEM_CUDA_ENABLE="${FI_HMEM_CUDA_ENABLE:-1}"
    export FI_HMEM_CUDA_USE_GDRCOPY="${FI_HMEM_CUDA_USE_GDRCOPY:-0}"
    export FI_CXI_ENABLE_DMABUF="${FI_CXI_ENABLE_DMABUF:-1}"
    export FI_CXI_DISABLE_HMEM_DEV_REGISTER="${FI_CXI_DISABLE_HMEM_DEV_REGISTER:-1}"
    # PALS sets TMPDIR to /var/run/palsd/<uuid>, which the job cannot write;
    # anything opening a temp file then dies with PermissionError.
    export TMPDIR="${TMPDIR_OVERRIDE:-/tmp/${PBS_JOBID:-nixl}}"
    mkdir -p "$TMPDIR" 2>/dev/null

    echo "FABRIC=cxi LIBFABRIC=$CUSTOM_LIBFABRIC PLUGIN_DIR=${NIXL_PLUGIN_DIR:-<unset>}" | tee -a "$LOG"
    # Report the VNI rather than assume it. Absent VNI means the allocation
    # was made without ngpus and LIBFABRIC will fail; say so now, not in a
    # stack trace 8 ranks deep.
    #
    # Launch this across 2 nodes, not 1: PALS only provisions a Slingshot VNI
    # for a launch that actually spans nodes, so `mpiexec --np 1` reports
    # <none> even inside a perfectly good ngpus=4 allocation -- a false alarm
    # that sends you chasing a resource bug that isn't there.
    _vni=$(mpiexec --np 2 -ppn 1 sh -c 'echo "${SLINGSHOT_VNIS:-}"' 2>/dev/null \
           | tr -d '[:space:]' | head -c 64)
    echo "SLINGSHOT_VNI=${_vni:-<none>}" | tee -a "$LOG"
    if [[ -z "$_vni" ]]; then
        echo "WARNING: no Slingshot VNI in this allocation -- LIBFABRIC will fail." | tee -a "$LOG"
        echo "         Resubmit with -l select=N:ngpus=4 (see header)." | tee -a "$LOG"
    fi
fi

# Fail early on a missing mpi4py rather than letting every rank print
# NIXL_REMOTE_SKIP and bottom out in a generic NO_RECORDS, which reads like
# the fabric failed when the real cause is one absent module.
if ! PYTHONPATH="$PYTHONPATH" "$PYBIN" -c 'import mpi4py' 2>/dev/null; then
    echo "VERDICT=NO_MPI4PY ($PYBIN cannot import mpi4py; it is needed only for the" \
         "out-of-band peer exchange. pip install --target <dir> mpi4py and pass" \
         "EXTRA_PYTHONPATH=<dir>)" | tee -a "$LOG"
    exit 1
fi

# Report which NIXL actually wins the import race, and refuse to run if it is
# not the one whose plugins we just configured. EXTRA_PYTHONPATH makes this a
# live hazard: pointing it at another venv's full site-packages drags that
# venv's nixl in too, so the agent builds against one install's libfabric and
# another's plugins. That combination constructs cleanly and then fails every
# transfer with NIXL_ERR_BACKEND at postXferReq -- an error that says nothing
# about its real cause. Point EXTRA_PYTHONPATH at a directory holding ONLY
# the missing module.
_nixl_origin=$(PYTHONPATH="$PYTHONPATH" "$PYBIN" -c \
    'import nixl,pathlib;print(pathlib.Path(nixl.__file__).resolve().parent.parent)' 2>/dev/null)
echo "NIXL_IMPORTED_FROM=${_nixl_origin:-<none>}" | tee -a "$LOG"
if [[ -n "${NIXL_INSTALL_DIR:-}" && -n "$_nixl_origin" && "$_nixl_origin" != "$NIXL_INSTALL_DIR"* ]]; then
    _venv_sp="${VENV:-}/lib/python3.12/site-packages"
    if [[ "$_nixl_origin" != "$_venv_sp" ]]; then
        echo "VERDICT=NIXL_SHADOWED (configured NIXL_INSTALL_DIR=$NIXL_INSTALL_DIR and" \
             "VENV=$VENV, but 'import nixl' resolved to $_nixl_origin -- almost always" \
             "EXTRA_PYTHONPATH pointing at a full site-packages. Narrow it.)" | tee -a "$LOG"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: fabric baseline through MPI, NOT NIXL.
# Without this, a slow NIXL number is unattributable -- it could be the fabric,
# the placement, or NIXL. With it, the comparison is on the same two nodes in
# the same allocation, which is the only version of the claim worth making.
# ---------------------------------------------------------------------------
echo "=== MPI fabric baseline (same nodes, same allocation) ===" | tee -a "$LOG"
MPICH_GPU_SUPPORT_ENABLED=0 mpiexec --np 2 -ppn 1 \
    --env PYTHONPATH="$PYTHONPATH" \
    "$PYBIN" "$REPO/examples/19_nixl_remote_rdma/mpi_fabric_baseline.py" \
    2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Step 2: the NIXL cross-node measurement.
# NIXL moves device memory itself and does not use Cray MPICH's GPU path, so
# MPICH_GPU_SUPPORT_ENABLED must NOT be set here -- setting it globally aborts
# any step whose binary is not GTL-linked.
# ---------------------------------------------------------------------------
echo "=== NIXL cross-node RDMA ===" | tee -a "$LOG"
# The FI_*/plugin vars must be forwarded explicitly: PALS does not propagate
# the submitting shell's environment to launched ranks, so exporting them
# above is necessary but not sufficient.
mpiexec --np "$NRANKS" -ppn "$RANKS_PER_NODE" \
    --env PYTHONPATH="$PYTHONPATH" \
    --env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    --env NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR:-}" \
    --env FI_PROVIDER="${FI_PROVIDER:-}" \
    --env FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-}" \
    --env FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-}" \
    --env FI_HMEM_CUDA_ENABLE="${FI_HMEM_CUDA_ENABLE:-}" \
    --env FI_HMEM_CUDA_USE_GDRCOPY="${FI_HMEM_CUDA_USE_GDRCOPY:-}" \
    --env FI_CXI_ENABLE_DMABUF="${FI_CXI_ENABLE_DMABUF:-}" \
    --env FI_CXI_DISABLE_HMEM_DEV_REGISTER="${FI_CXI_DISABLE_HMEM_DEV_REGISTER:-}" \
    --env TMPDIR="${TMPDIR:-/tmp}" \
    "$PYBIN" "$REPO/examples/19_nixl_remote_rdma/run_nixl_remote.py" \
        --sizes "${SIZES:-1048576,16777216,67108864}" \
        --iterations "${ITERATIONS:-10}" \
        --pairing "${PAIRING:-split}" \
        --ranks-per-node "$RANKS_PER_NODE" \
        --backend-preference "${BACKEND_PREFERENCE:-LIBFABRIC,UCX,GDS,POSIX}" \
        --op "${NIXL_OP:-READ}" \
    2>&1 | tee -a "$LOG"

EXIT_STATUS=${PIPESTATUS[0]}

# --- Gates. Exit 0 is not evidence: a run that skipped everything also exits 0.
RECORDS=$(grep -c "^NIXL_REMOTE_RESULT" "$LOG" 2>/dev/null || true)
echo "REMOTE_RECORDS=${RECORDS:-0}" | tee -a "$LOG"
if [[ "${RECORDS:-0}" -eq 0 ]]; then
    echo "VERDICT=NO_RECORDS (exit $EXIT_STATUS, nothing measured)" | tee -a "$LOG"
    exit 1
fi

# Which backend actually carried the bytes. This gate exists because the
# original run of this example reported 0.67 GB/s with every correctness
# check green: UCX had silently fallen back to TCP. Correct bytes over the
# wrong transport is the failure mode this catches.
BACKENDS_USED=$(awk '/^NIXL_REMOTE_RESULT/ {print $3}' "$LOG" | sort -u | tr '\n' ',' | sed 's/,$//')
echo "BACKENDS_USED=${BACKENDS_USED:-<none>}" | tee -a "$LOG"
if [[ "$NIXL_FABRIC" == "cxi" && "$BACKENDS_USED" != *LIBFABRIC* ]]; then
    echo "VERDICT=WRONG_BACKEND (expected LIBFABRIC on a CXI fabric, got ${BACKENDS_USED:-<none>};" \
         "numbers below are a TCP fallback, not RDMA)" | tee -a "$LOG"
    exit 1
fi

# Corruption is a hard failure, never a slow-but-usable result.
if grep -q "^NIXL_REMOTE_CORRUPT" "$LOG"; then
    echo "VERDICT=CORRUPT (transfer completed but delivered wrong bytes)" | tee -a "$LOG"
    exit 1
fi

# Every result must be verified. An unverified row is a bandwidth number for a
# transfer nobody checked actually moved the sender's data.
UNVERIFIED=$(awk '/^NIXL_REMOTE_RESULT/ && $NF != "True"' "$LOG" | wc -l)
echo "UNVERIFIED_ROWS=$UNVERIFIED" | tee -a "$LOG"
if [[ "$UNVERIFIED" -ne 0 ]]; then
    echo "VERDICT=UNVERIFIED (byte-level check did not pass on every row)" | tee -a "$LOG"
    exit 1
fi

# Every pairing must actually cross a node boundary, or this measured the wrong
# thing while looking entirely healthy.
SAMENODE=$(awk '/^NIXL_RMAP/ && $NF != "True"' "$LOG" | wc -l)
echo "SAME_NODE_PAIRS=$SAMENODE EXPECTED=0" | tee -a "$LOG"
if [[ "$SAMENODE" -ne 0 ]]; then
    echo "VERDICT=NOT_CROSS_NODE (pairing degenerated to same-node traffic)" | tee -a "$LOG"
    exit 1
fi

# Each rank must own a distinct device; ranks sharing one inflate per-device
# bandwidth while every other check still passes.
DEVS=$(awk '/^NIXL_RMAP /{print $2, $6}' "$LOG" | sort -u | wc -l)
echo "DISTINCT_HOST_DEV=$DEVS EXPECTED=$NRANKS" | tee -a "$LOG"
if [[ "$DEVS" -ne "$NRANKS" ]]; then
    echo "VERDICT=DEVICE_COLLAPSE (ranks sharing devices; numbers not trustworthy)" | tee -a "$LOG"
    exit 1
fi

# Report the gap against the MPI baseline rather than leaving it to the reader.
BASE=$(awk '/^MPI_BASELINE/ {print $NF}' "$LOG" | tail -1 | sed 's/oneway_GB\/s=//')
NIXLBW=$(awk '/^NIXL_REMOTE_RESULT/ {print $5}' "$LOG" | sort -g | tail -1)
echo "FABRIC_BASELINE_GBPS=${BASE:-unknown} NIXL_BEST_GBPS=${NIXLBW:-unknown}" | tee -a "$LOG"

echo "VERDICT=OK" | tee -a "$LOG"
exit "$EXIT_STATUS"
