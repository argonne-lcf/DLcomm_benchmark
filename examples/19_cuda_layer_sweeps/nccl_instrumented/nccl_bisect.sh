#!/bin/bash
#PBS -N nccbisect
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Isolate WHICH profiler event bit breaks the two failing cells.
#
# Job 7018 with mask=90 (Coll|ProxyOp|ProxyStep|KernelCh) produced:
#   2 ranks / 256 MiB -> SIGSEGV
#   8 ranks / 1 MiB   -> "Out of bounds values : 8 FAILED"
# Both pass with no profiler at all, so the plugin is implicated -- but
# "the plugin is buggy" is not actionable. Bisecting the mask says whether a
# usable subset exists, which decides if attribution can be collected safely.
#
# Bits (nccl/profiler.h): Group=1 Coll=2 P2p=4 ProxyOp=8 ProxyStep=16
#                         NetPlugin=32 KernelCh=64
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

OUT=$W/nccbisect_${PBS_JOBID}
mkdir -p "$OUT"

run_one() {
  local RANKS=$1 PPN=$2 SIZE=$3 MASK=$4 TAG=$5
  local D="$OUT/$TAG"
  mkdir -p "$D"
  local ENVARGS=(--env LD_LIBRARY_PATH="$LDP"
                 --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so"
                 --env NCCL_DEBUG=INFO)
  if [ "$MASK" != "none" ]; then
    ENVARGS+=(--env NCCL_PROFILER_PLUGIN="$INSTR/lib/libnccl-profiler-example.so"
              --env NCCL_PROFILE_EVENT_MASK="$MASK"
              --env NCCL_PROFILE_DUMP_FILE="$D/trace")
  fi

  timeout 400 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
      "${ENVARGS[@]}" "$W/pals_nccl_env.sh" "$NB" \
      -b "$SIZE" -e "$SIZE" -f 1 -g 1 -n 20 > "$D/run.txt" 2>&1
  local RC=$?

  local STATUS=ok
  grep -qa 'error while loading shared libraries' "$D/run.txt" && STATUS=libfail
  grep -qa 'signal 11' "$D/run.txt" && STATUS=sigsegv
  grep -qa 'Out of bounds values : [1-9]' "$D/run.txt" && STATUS=validation_failed
  grep -qa 'Out of bounds values : 0 OK' "$D/run.txt" || \
      [ "$STATUS" != ok ] || STATUS=no_result

  local BW
  BW=$(grep -a 'Avg bus bandwidth' "$D/run.txt" | awk '{print $5}' | head -1)
  printf 'ranks=%s size=%s mask=%-5s rc=%-3s status=%-18s busbw=%s traces=%s\n' \
      "$RANKS" "$SIZE" "$MASK" "$RC" "$STATUS" "${BW:-NA}" \
      "$(ls "$D"/trace_*.json 2>/dev/null | wc -l)"
}

echo "########## cell A: 2 ranks / 256 MiB (SIGSEGV at mask=90) ##########"
for M in none 0 2 8 16 64 24 90; do
  run_one 2 1 268435456 "$M" "A_mask_${M}"
done

echo "########## cell B: 8 ranks / 1 MiB (validation failure at mask=90) ##########"
for M in none 0 2 8 16 64 24 90; do
  run_one 8 4 1048576 "$M" "B_mask_${M}"
done

echo "########## control: 8 ranks / 256 MiB (passed at mask=90) ##########"
for M in none 90; do
  run_one 8 4 268435456 "$M" "C_mask_${M}"
done

echo "=== bisect complete: $OUT ==="
