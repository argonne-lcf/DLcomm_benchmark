#!/bin/bash
#PBS -N nccvar
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:40:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Run-to-run variance, measured properly.
#
# The earlier ~57% figure was never attributed. Within-run jitter was offered
# as a candidate, which is not a diagnosis. To separate the two sources this
# script is submitted N times as INDEPENDENT jobs:
#
#   within-job  variance = repeats inside one allocation (same nodes, same
#                          process launch) -> jitter
#   across-job  variance = spread of per-job medians (different allocations,
#                          possibly different nodes) -> placement/topology
#
# If across-job >> within-job, the 57% is placement, not jitter, and no amount
# of tuning inside a run will fix it. If they are comparable, jitter is the
# explanation.
#
# The node list is recorded per run, so a slow job can be traced to hardware.
# No profiler plugin: it perturbs what we are trying to measure, and it
# crashes some cells (job 7018).
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

OUT=$W/nccvar_${PBS_JOBID}
mkdir -p "$OUT"

SIZE=268435456
RANKS=8
PPN=4
REPS=5

# Which physical nodes this allocation got. A job that lands on a slow node
# should be identifiable rather than just "an outlier".
awk '!seen[$0]++' "$PBS_NODEFILE" > "$OUT/nodes.txt"
echo "JOBID=$PBS_JOBID"
echo "NODES=$(tr '\n' ',' < "$OUT/nodes.txt")"

for R in $(seq 1 $REPS); do
  F="$OUT/rep${R}.txt"
  timeout 400 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
      --env LD_LIBRARY_PATH="$LDP" \
      --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so" \
      --env NCCL_DEBUG=INFO \
      "$W/pals_nccl_env.sh" "$NB" \
      -b $SIZE -e $SIZE -f 1 -g 1 -n 20 > "$F" 2>&1
  RC=$?

  if grep -qa 'error while loading shared libraries' "$F"; then
    echo "rep=$R FATAL library resolution"
    continue
  fi
  # A wrong answer still prints a bandwidth (job 7018). Never record one.
  if grep -qa 'Out of bounds values : [1-9]' "$F"; then
    echo "rep=$R FAILED validation"
    continue
  fi

  TR=$(grep -ao 'Using network [A-Za-z]*' "$F" | head -1 | awk '{print $3}')
  read -r US BW < <(awk '
    /NCCL INFO|NET\/OFI/ { next }
    {
      n = 0
      for (i = 1; i <= NF; i++) if ($i ~ /^-?[0-9]+(\.[0-9]+)?$/) n++
      if (NF == 13 && $1 ~ /^[0-9]+$/ && $6 ~ /^[0-9.]+$/) { print $6, $8; exit }
      if (NF == 8 && n == 8 && $4 ~ /^[0-9]+$/ && $8 ~ /^[0-9]+$/) { print $1, $3; exit }
    }' "$F")
  echo "rep=$R rc=$RC transport=${TR:-unknown} time_us=${US:-NA} busbw=${BW:-NA}"
done

echo "=== variance run complete: $OUT ==="
