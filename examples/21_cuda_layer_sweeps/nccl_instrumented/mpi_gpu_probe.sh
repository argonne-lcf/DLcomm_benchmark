#!/bin/bash
#PBS -N mpigpu
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Is inter-node GPU-buffer MPI broken on this machine, or is it my program?
#
# OSU with -d cuda is the known-good GTL binary: it produced 19-51 GB/s in
# job 7047 on 2 nodes. If OSU also fails here, the fault is the stack, not
# lifecycle_mpi.c. Same node count, same launcher, back to back in ONE job
# so node assignment cannot confound the comparison.

set -uo pipefail
W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
cd "$W" || exit 1
OUT=$W/mpigpu_${PBS_JOBID%%.*}
mkdir -p "$OUT"

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
OSU=$W/bench_install_gtl/libexec/osu-micro-benchmarks/mpi/collective
OSUENV=$W/pals_osu_env.sh
SIZE=16777216

echo "### A: OSU allreduce, 2 nodes, device buffers (known-good binary)"
timeout 300 "$MPIEXEC" -n 2 -ppn 1 --hostfile "$PBS_NODEFILE" \
  --env MPICH_GPU_SUPPORT_ENABLED=1 \
  "$OSUENV" "$OSU/osu_allreduce" -d cuda -m $SIZE:$SIZE > "$OUT/a_osu.txt" 2>&1
echo "exit=$?"; tail -4 "$OUT/a_osu.txt"

echo
echo "### B: my lifecycle_mpi, 2 nodes, device buffers"
GTL_DIR=/opt/cray/pe/mpich/9.1.0/gtl/lib
CUDA12_INC=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/include
CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0/lib64
LC=$OUT/lifecycle_mpi
mpicc -O2 -o "$LC" "$W/lifecycle_mpi.c" -I"$CUDA12_INC" \
  -L"$GTL_DIR" -lmpi_gtl_cuda -Wl,-rpath,"$GTL_DIR" \
  -L"$CUDA13_LIB" -lcudart -Wl,-rpath,"$CUDA13_LIB" \
  > "$OUT/build.log" 2>&1 && echo "build: ok" || { echo "build: FAILED"; tail -12 "$OUT/build.log"; }
if [ -x "$LC" ]; then
  timeout 300 "$MPIEXEC" -n 2 -ppn 1 --hostfile "$PBS_NODEFILE" \
    --env MPICH_GPU_SUPPORT_ENABLED=1 \
    "$OSUENV" "$LC" --bytes $SIZE --iters 5 > "$OUT/b_mine.txt" 2>&1
  echo "exit=$?"; tail -4 "$OUT/b_mine.txt"
else
  echo "lifecycle_mpi not found at $LC"
fi

echo
echo "### C: my lifecycle_mpi, 2 nodes, SMALL message (4 KiB)"
if [ -x "$LC" ]; then
  timeout 300 "$MPIEXEC" -n 2 -ppn 1 --hostfile "$PBS_NODEFILE" \
    --env MPICH_GPU_SUPPORT_ENABLED=1 \
    "$OSUENV" "$LC" --bytes 4096 --iters 5 > "$OUT/c_small.txt" 2>&1
  echo "exit=$?"; tail -3 "$OUT/c_small.txt"
fi

echo
echo "### D: OSU at the same 16 MiB but 1 node (control)"
timeout 300 "$MPIEXEC" -n 2 -ppn 2 --hostfile "$PBS_NODEFILE" \
  --env MPICH_GPU_SUPPORT_ENABLED=1 \
  "$OSUENV" "$OSU/osu_allreduce" -d cuda -m $SIZE:$SIZE > "$OUT/d_1node.txt" 2>&1
echo "exit=$?"; tail -3 "$OUT/d_1node.txt"

echo
echo "### E: ldd comparison"
echo "-- OSU --"; ldd "$OSU/osu_allreduce" | grep -iE 'gtl|cudart' || echo "none"
echo "-- mine --"; [ -x "$LC" ] && ldd "$LC" | grep -iE 'gtl|cudart' || echo "none"
