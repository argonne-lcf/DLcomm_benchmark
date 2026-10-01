#!/bin/bash
#PBS -N nccllife
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Lifecycle cost decomposition: init / first op / steady op / destroy,
# for MPI, NCCL and torch.distributed, on 1 node and 2 nodes.
#
# The point of this job is that a benchmark reporting only steady-state
# bandwidth hides the cost a real application actually pays at startup.

set -uo pipefail
W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
cd "$W" || exit 1

OUT=$W/ncclife_${PBS_JOBID%%.*}
mkdir -p "$OUT"

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
PY=$W/venv/bin/python
NCCLENV=$W/pals_nccl_env.sh
OSUENV=$W/pals_osu_env.sh

# --env LD_LIBRARY_PATH REPLACES the module value, so CUDA dirs must be
# restated or every rank dies with libcudart.so.13 not found (exit 127).
INSTR=$W/nccl_instr
FI=$W/shs-libfabric-install
# Must match the runtime the working GTL binary (OSU) resolves. Using the
# REDIST copy instead produced cxil_map write errors on inter-node device
# transfers at 16 MiB while 4 KiB worked -- job 7057 A/B.
CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0/lib64
# The CUDA driver library: required by the GTL to pin device memory for RDMA.
CUDA_DRIVER_LIB=$(dirname "$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')")
[ -d "$CUDA_DRIVER_LIB" ] || CUDA_DRIVER_LIB=/usr/lib64
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/lib
LDP="$INSTR/lib:$CUDA13_LIB:$CUDA12_LIB:$FI/lib:${LD_LIBRARY_PATH:-}"

if [ ! -f "$INSTR/lib/libnccl-net.so" ]; then
  echo "FATAL: $INSTR/lib/libnccl-net.so missing; NCCL would fall back to Socket" >&2
  exit 1
fi

CXI_ENV=(
  --env LD_LIBRARY_PATH="$LDP"
  --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so"
  --env NCCL_DEBUG=INFO
  --env NCCL_DEBUG_FILE=$OUT/nccldbg.%h.%p
)

# PALS: rank 0 is the FIRST line of the nodefile; PBS does not sort it.
mapfile -t HOSTS < <(awk '!seen[$0]++' "$PBS_NODEFILE")
echo "nodes: ${HOSTS[*]}"

# ---------------------------------------------------------------- build
# Built inside the job: the login node has a fork cap that breaks some
# builds, and the compute node is where the real CUDA arch lives.
echo "=== build ==="
BUILD_OK_MPI=0; BUILD_OK_NCCL=0
# Device pointers into MPI require the GPU Transport Layer. Without it
# MPICH reads device memory as host memory and the rank dies with signal 11.
GTL_DIR=$(ls -d /opt/cray/pe/mpich/9.1.0/gtl/lib 2>/dev/null | head -1)
# Compile against CUDA 12 headers, LINK the CUDA 13 runtime the GTL needs.
mpicc -O2 -o "$OUT/lifecycle_mpi" "$W/lifecycle_mpi.c" \
  -I"$CUDA12_LIB/../include" \
  ${GTL_DIR:+-L$GTL_DIR -lmpi_gtl_cuda -Wl,-rpath,$GTL_DIR} \
  -L"$CUDA13_LIB" -lcudart -Wl,-rpath,"$CUDA13_LIB" \
  -L"$CUDA_DRIVER_LIB" -lcuda -Wl,-rpath,"$CUDA_DRIVER_LIB" \
  > "$OUT/build_mpi.log" 2>&1 && BUILD_OK_MPI=1
echo "gtl dir: ${GTL_DIR:-NONE}"
if [ "$BUILD_OK_MPI" = 1 ]; then
  echo "--- ldd gtl/driver check ---"
  ldd "$OUT/lifecycle_mpi" | grep -iE 'gtl|libcuda\.so\.1|cudart' || true
  ldd "$OUT/lifecycle_mpi" | grep -q 'libmpi_gtl_cuda' || echo "WARNING: no GTL linked"
  # Inter-node device transfers above the eager limit fail without this.
  ldd "$OUT/lifecycle_mpi" | grep -q 'libcuda\.so\.1' || echo "WARNING: no CUDA driver linked"
fi
echo "mpi build: $BUILD_OK_MPI"
[ "$BUILD_OK_MPI" = 0 ] && tail -15 "$OUT/build_mpi.log"

nvcc -O2 -arch=sm_90 -o "$OUT/lifecycle_nccl" "$W/lifecycle_nccl.cu" \
  -I"$INSTR/include" -L"$INSTR/lib" -lnccl \
  -I/opt/cray/pe/mpich/9.1.0/ofi/nvidia/23.3/include \
  -L/opt/cray/pe/mpich/9.1.0/ofi/nvidia/23.3/lib -lmpi \
  > "$OUT/build_nccl.log" 2>&1 && BUILD_OK_NCCL=1
echo "nccl build: $BUILD_OK_NCCL"
[ "$BUILD_OK_NCCL" = 0 ] && tail -15 "$OUT/build_nccl.log"

CSV=$OUT/lifecycle.csv
echo "scale,nodes,ranks,lib,bytes,import_us,bootstrap_us,init_us,first_us,steady_us,p90_us,destroy_us,gbps,transport,status" > "$CSV"

row() { echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11},${12},${13},${14},${15}" >> "$CSV"; }

# Pull a key=value out of a RESULT line; '-' when absent.
kv() { grep -a '^RESULT' "$1" | tail -1 | grep -ao "$2=[0-9.]*" | cut -d= -f2 | tail -1; }
kvd() { local v; v=$(kv "$1" "$2"); echo "${v:--}"; }

transport_of() {
  if grep -qa 'Using network Libfabric' "$1" $OUT/nccldbg.* 2>/dev/null; then
    echo Libfabric
  elif grep -qa 'Using network Socket' "$1" $OUT/nccldbg.* 2>/dev/null; then
    echo Socket
  else
    echo n/a
  fi
}

SIZE=16777216
ITERS=20

run_scale() {
  local scale=$1 nodes=$2; shift 2
  for RANKS in "$@"; do
    local PPN=$(( RANKS / nodes )); [ "$PPN" -lt 1 ] && PPN=1

    # ---- MPI ----
    if [ "$BUILD_OK_MPI" = 1 ]; then
      L=$OUT/${scale}_r${RANKS}_mpi.txt
      timeout 900 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
        --env MPICH_GPU_SUPPORT_ENABLED=1 \
        "$OSUENV" "$OUT/lifecycle_mpi" --bytes $SIZE --iters $ITERS > "$L" 2>&1
      if grep -qa '^RESULT' "$L" && ! grep -qa 'error=' "$L"; then
        row "$scale" "$nodes" "$RANKS" mpi "$SIZE" - - \
          "$(kvd "$L" init_us)" "$(kvd "$L" first_us)" "$(kvd "$L" steady_us)" \
          "$(kvd "$L" p90_us)" "$(kvd "$L" destroy_us)" "$(kvd "$L" gbps)" mpi ok
      else
        row "$scale" "$nodes" "$RANKS" mpi "$SIZE" - - - - - - - - mpi no_result
      fi
    else
      row "$scale" "$nodes" "$RANKS" mpi "$SIZE" - - - - - - - - mpi build_failed
    fi

    # ---- NCCL ----
    if [ "$BUILD_OK_NCCL" = 1 ]; then
      rm -f $OUT/nccldbg.* 2>/dev/null
      L=$OUT/${scale}_r${RANKS}_nccl.txt
      timeout 900 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
        "${CXI_ENV[@]}" "$NCCLENV" "$OUT/lifecycle_nccl" \
        --bytes $SIZE --iters $ITERS > "$L" 2>&1
      TR=$(transport_of "$L")
      if grep -qa 'VALIDATION_FAILED' "$L"; then
        row "$scale" "$nodes" "$RANKS" nccl "$SIZE" - - - - - - - - "$TR" validation_failed
      elif grep -qa '^RESULT' "$L" && ! grep -qa 'error=' "$L"; then
        row "$scale" "$nodes" "$RANKS" nccl "$SIZE" - \
          "$(kvd "$L" bootstrap_us)" "$(kvd "$L" init_us)" "$(kvd "$L" first_us)" \
          "$(kvd "$L" steady_us)" "$(kvd "$L" p90_us)" "$(kvd "$L" destroy_us)" \
          "$(kvd "$L" gbps)" "$TR" ok
      else
        row "$scale" "$nodes" "$RANKS" nccl "$SIZE" - - - - - - - - "$TR" no_result
      fi
    else
      row "$scale" "$nodes" "$RANKS" nccl "$SIZE" - - - - - - - - n/a build_failed
    fi

    # ---- torch.distributed ----
    rm -f $OUT/nccldbg.* 2>/dev/null
    L=$OUT/${scale}_r${RANKS}_torch.txt
    MADDR=${HOSTS[0]}
    MPORT=$(( 29500 + (RANDOM % 2000) ))
    timeout 900 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
      "${CXI_ENV[@]}" \
      --env MASTER_ADDR="$MADDR" --env MASTER_PORT="$MPORT" \
      --env WORLD_SIZE="$RANKS" \
      "$NCCLENV" "$PY" "$W/lifecycle_torch.py" \
      --bytes $SIZE --iters $ITERS > "$L" 2>&1
    TR=$(transport_of "$L")
    if grep -qa 'VALIDATION_FAILED' "$L"; then
      row "$scale" "$nodes" "$RANKS" torch "$SIZE" - - - - - - - - "$TR" validation_failed
    elif grep -qa '^RESULT' "$L" && ! grep -qa 'error=' "$L"; then
      row "$scale" "$nodes" "$RANKS" torch "$SIZE" \
        "$(kvd "$L" import_us)" - "$(kvd "$L" init_us)" "$(kvd "$L" first_us)" \
        "$(kvd "$L" steady_us)" "$(kvd "$L" p90_us)" "$(kvd "$L" destroy_us)" \
        "$(kvd "$L" gbps)" "$TR" ok
    else
      row "$scale" "$nodes" "$RANKS" torch "$SIZE" - - - - - - - - "$TR" no_result
    fi
  done
}

echo "=== 1 node ==="
run_scale 1node 1 2 4
echo "=== 2 nodes ==="
run_scale 2node 2 2 4 8

echo
echo "=== lifecycle complete ==="
column -s, -t "$CSV"
