#!/bin/bash
#PBS -N mpibisect
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:40:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Two theories about the inter-node GPU allreduce failure were wrong
# (cudart provenance, cudaSetDevice ordering). Stop theorising: find the
# exact size threshold, then difference my program against OSU on the one
# axis that remains.

set -uo pipefail
W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
cd "$W" || exit 1
OUT=$W/mpibisect_${PBS_JOBID%%.*}
mkdir -p "$OUT"

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
OSU=$W/bench_install_gtl/libexec/osu-micro-benchmarks/mpi/collective
OSUENV=$W/pals_osu_env.sh
GTL_DIR=/opt/cray/pe/mpich/9.1.0/gtl/lib
CUDA12_INC=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/include
CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0/lib64

LC=$OUT/lifecycle_mpi
mpicc -O2 -o "$LC" "$W/lifecycle_mpi.c" -I"$CUDA12_INC" \
  -L"$GTL_DIR" -lmpi_gtl_cuda -Wl,-rpath,"$GTL_DIR" \
  -L"$CUDA13_LIB" -lcudart -Wl,-rpath,"$CUDA13_LIB" \
  > "$OUT/build.log" 2>&1 && echo "build: ok" || { echo "build FAILED"; tail -12 "$OUT/build.log"; exit 1; }

echo
echo "=== size threshold for MY binary, 2 nodes ==="
for S in 4096 65536 262144 524288 1048576 2097152 4194304 8388608 16777216; do
  timeout 180 "$MPIEXEC" -n 2 -ppn 1 --hostfile "$PBS_NODEFILE" \
    --env MPICH_GPU_SUPPORT_ENABLED=1 \
    "$OSUENV" "$LC" --bytes $S --iters 3 > "$OUT/s_$S.txt" 2>&1
  RC=$?
  if grep -qa '^RESULT' "$OUT/s_$S.txt" && ! grep -qa 'error=' "$OUT/s_$S.txt"; then
    echo "  $S bytes: OK   rc=$RC"
  else
    W1=$(grep -ao 'cxil_map: write error' "$OUT/s_$S.txt" | head -1)
    echo "  $S bytes: FAIL rc=$RC  ${W1:-}"
  fi
done

echo
echo "=== same sizes with OSU (control) ==="
for S in 1048576 4194304 16777216; do
  timeout 180 "$MPIEXEC" -n 2 -ppn 1 --hostfile "$PBS_NODEFILE" \
    --env MPICH_GPU_SUPPORT_ENABLED=1 \
    "$OSUENV" "$OSU/osu_allreduce" -d cuda -m $S:$S > "$OUT/o_$S.txt" 2>&1
  V=$(grep -aE "^$S " "$OUT/o_$S.txt" | awk '{print $2}')
  echo "  $S bytes: ${V:-FAIL}"
done

# OSU allocates ONE buffer and uses MPI_IN_PLACE-free sendrecv; my program
# allocates two. Also OSU touches the buffer from a kernel. Test whether an
# untouched (cudaMemset-only) buffer is the trigger.
echo
echo "=== does the failure follow the MPI call or the buffer? ==="
echo "(16 MiB with --iters 0 would skip the steady loop; first op alone)"
timeout 180 "$MPIEXEC" -n 2 -ppn 1 --hostfile "$PBS_NODEFILE" \
  --env MPICH_GPU_SUPPORT_ENABLED=1 \
  "$OSUENV" "$LC" --bytes 16777216 --iters 1 > "$OUT/firstonly.txt" 2>&1
echo "exit=$?"; grep -aE 'RESULT|cxil_map|Bad address' "$OUT/firstonly.txt" | head -3

echo
echo "=== MPICH env actually seen by the rank ==="
timeout 120 "$MPIEXEC" -n 2 -ppn 1 --hostfile "$PBS_NODEFILE" \
  --env MPICH_GPU_SUPPORT_ENABLED=1 \
  bash -c 'echo "rank $PALS_RANKID: GPU_SUPPORT=$MPICH_GPU_SUPPORT_ENABLED dev=$PALS_LOCAL_RANKID"' 2>&1 | head -4
