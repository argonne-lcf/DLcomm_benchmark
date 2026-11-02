#!/bin/bash
#PBS -N ncclprof2
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Per-phase attribution across the RANK LADDER on CXI.
#
# Job 7013 traced only 8 ranks. The question "where does the cost go as you
# scale" needs the same trace at 2, 4 and 8 ranks, so the PROXY/NET/GPU split
# can be read as a function of rank count rather than a single point.
set -uo pipefail
W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
cd "$W"

module load cray-mpich/9.1.0 2>/dev/null
module load cuda/13.0 2>/dev/null
module unload darshan 2>/dev/null

export TMPDIR=/tmp/${PBS_JOBID}
mkdir -p "$TMPDIR"

INSTR=$W/nccl_instr
FI=$W/shs-libfabric-install
CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/REDIST/cuda/13.0/targets/sbsa-linux/lib
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/lib
LDP="$INSTR/lib:$CUDA13_LIB:$CUDA12_LIB:$FI/lib:${LD_LIBRARY_PATH:-}"

NB=$W/bench_src/nccl-tests/build/all_reduce_perf
MPIEXEC=$(ls /opt/cray/pals/*/bin/mpiexec 2>/dev/null | head -1)
[ -x "$MPIEXEC" ] || MPIEXEC=$(ls /opt/cray/pe/pals/*/bin/mpiexec 2>/dev/null | head -1)

OUT=$W/ncclprof_${PBS_JOBID}
mkdir -p "$OUT"

# Mask 90 = Coll(2) | ProxyOp(8) | ProxyStep(16) | KernelCh(64).
# It defaults to 0: without it the plugin loads, says so, and records nothing.
MASK=90

for PAIR in 2:1 4:2 8:4; do
  RANKS=${PAIR%%:*}
  PPN=${PAIR##*:}
  for SIZE in 1048576 268435456; do
    D="$OUT/r${RANKS}_${SIZE}"
    mkdir -p "$D"
    echo "########## ranks=$RANKS ppn=$PPN size=$SIZE ##########"
    timeout 600 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
        --env LD_LIBRARY_PATH="$LDP" \
        --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so" \
        --env NCCL_PROFILER_PLUGIN="$INSTR/lib/libnccl-profiler-example.so" \
        --env NCCL_PROFILE_EVENT_MASK="$MASK" \
        --env NCCL_PROFILE_DUMP_FILE="$D/trace" \
        --env NCCL_DEBUG=INFO \
        "$W/pals_nccl_env.sh" "$NB" \
        -b $SIZE -e $SIZE -f 1 -g 1 -n 20 > "$D/run.txt" 2>&1
    RC=$?

    # Library resolution failures exit 127 -> PALS 143, which looks exactly
    # like a fabric timeout. Name it rather than letting it pass as a result.
    if grep -qa 'error while loading shared libraries' "$D/run.txt"; then
      echo "  FATAL: library resolution failed"
      grep -am2 'error while loading shared libraries' "$D/run.txt"
      continue
    fi

    TR=$(grep -ao 'Using network [A-Za-z]*' "$D/run.txt" | head -1 | awk '{print $3}')
    echo "  exit=$RC transport=${TR:-unknown} traces=$(ls "$D"/trace_*.json 2>/dev/null | wc -l)"
    grep -a 'Avg bus bandwidth' "$D/run.txt" || true
  done
done

echo "=== profiler ladder complete: $OUT ==="
