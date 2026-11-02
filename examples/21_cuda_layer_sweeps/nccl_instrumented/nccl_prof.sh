#!/bin/bash
#PBS -N ncclprof
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# In-collective timers via NCCL's profiler plugin API, on the instrumented
# build, over CXI. Job 7012 proved the transport switch (Socket -> Libfabric,
# 6.39 -> 124.01 GB/s). This job answers WHERE the time goes inside the
# collective, which NCCL_DEBUG text cannot show.
#
# Event mask (from ext-profiler/example/nccl/profiler.h), default is 0:
#   Coll(1<<1=2) + ProxyOp(1<<3=8) + ProxyStep(1<<4=16) + KernelCh(1<<6=64) = 90
# KernelCh = GPU-side channel time, ProxyStep = host/network progress.
# Their ratio is the compute-vs-wire split we could not measure before.
set -uo pipefail

W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
INSTR=$W/nccl_instr
FI=$W/shs-libfabric-install
RUN=$W/ncclprof_${PBS_JOBID%%.*}
mkdir -p "$RUN"
cd "$RUN" || exit 1

module load cray-mpich/9.1.0 2>/dev/null
module load cuda/13.0 2>/dev/null

CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/REDIST/cuda/13.0/targets/sbsa-linux/lib
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/lib
LDP="$INSTR/lib:$CUDA13_LIB:$CUDA12_LIB:$FI/lib:${LD_LIBRARY_PATH:-}"

MPIEXEC=$(ls /opt/cray/pals/*/bin/mpiexec 2>/dev/null | head -1)
[ -x "$MPIEXEC" ] || MPIEXEC=$(ls /opt/cray/pe/pals/*/bin/mpiexec 2>/dev/null | head -1)
NB=$W/bench_src/nccl-tests/build/all_reduce_perf
NCCLENV=$W/pals_nccl_env.sh

echo "nodes=$(awk '!seen[$0]++' "$PBS_NODEFILE" | wc -l)"
echo "mpiexec=$MPIEXEC"

MASK=90

# One size per run: the profiler writes a trace per rank and a 256 MiB
# 20-iteration trace is already large. 1 MiB and 256 MiB bracket the range
# where the Socket-era cliff was worst.
for SZ in 1048576 268435456; do
  echo ""
  echo "########## size=$SZ mask=$MASK ##########"
  OUT=$RUN/prof_${SZ}.txt

  timeout 600 "$MPIEXEC" -n 8 -ppn 4 --hostfile "$PBS_NODEFILE" \
      --env LD_LIBRARY_PATH="$LDP" \
      --env NCCL_DEBUG=WARN \
      --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so" \
      --env NCCL_PROFILER_PLUGIN="$INSTR/lib/libnccl-profiler-example.so" \
      --env NCCL_PROFILE_EVENT_MASK=$MASK \
      --env NCCL_PROFILE_DUMP_FILE="$RUN/trace_${SZ}" \
      "$NCCLENV" "$NB" -b $SZ -e $SZ -f 1 -g 1 -c 0 \
      > "$OUT" 2>&1
  echo "exit=$?"

  if grep -qaE 'not found|error while loading' "$OUT"; then
    echo "!! library resolution failed"; grep -ahoE '[^ ]*\.so[^ ]* => not found' "$OUT" | sort -u | head -3
  fi
  grep -a 'Avg bus bandwidth' "$OUT"
  echo "--- traces written ---"
  ls -la "$RUN"/trace_${SZ}* 2>/dev/null | head -10
  echo "trace count: $(ls "$RUN"/trace_${SZ}* 2>/dev/null | wc -l)"
done

echo ""
echo "RUN=$RUN"
