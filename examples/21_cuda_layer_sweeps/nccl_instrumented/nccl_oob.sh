#!/bin/bash
#PBS -N nccoob
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=00:50:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Is the 8-rank / 1 MiB validation failure real, and is it profiler-related?
#
# Job 7027 showed this cell failing WITHOUT the profiler (4 out-of-bounds
# values, 9.30773 GB/s printed anyway) and PASSING with mask=90. That
# inverts the earlier claim that the profiler caused it. Either it is
# intermittent, or something size-specific is wrong on the CXI path.
#
# This repeats the cell many times with no profiler, and sweeps nearby sizes,
# to establish a failure RATE rather than a single observation. A correctness
# bug that appears ~half the time is far more serious than a slow transport
# and must not be reported from one sample.
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

OUT=$W/nccoob_${PBS_JOBID}
mkdir -p "$OUT"

awk '!seen[$0]++' "$PBS_NODEFILE" > "$OUT/nodes.txt"
echo "NODES=$(tr '\n' ',' < "$OUT/nodes.txt")"

probe() {
  local SIZE=$1 REP=$2 STOCK=$3
  local TAG="s${SIZE}_r${REP}"
  local F="$OUT/${TAG}.txt"
  local LD="$LDP" PLUG=(--env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so")
  local WHICH=cxi
  if [ "$STOCK" = stock ]; then
    # Stock venv NCCL, no net plugin -> Socket. Does the failure follow the
    # transport or the size?
    LD="$W/venv/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA13_LIB:${LD_LIBRARY_PATH:-}"
    PLUG=()
    WHICH=stock
    TAG="${TAG}_stock"
    F="$OUT/${TAG}.txt"
  fi

  timeout 300 "$MPIEXEC" -n 8 -ppn 4 --hostfile "$PBS_NODEFILE" \
      --env LD_LIBRARY_PATH="$LD" "${PLUG[@]}" --env NCCL_DEBUG=INFO \
      "$W/pals_nccl_env.sh" "$NB" \
      -b "$SIZE" -e "$SIZE" -f 1 -g 1 -n 20 > "$F" 2>&1
  local RC=$?

  local OOB TR BW
  OOB=$(grep -ao 'Out of bounds values : [0-9]*' "$F" | head -1 | awk '{print $NF}')
  TR=$(grep -ao 'Using network [A-Za-z]*' "$F" | head -1 | awk '{print $3}')
  BW=$(grep -a 'Avg bus bandwidth' "$F" | awk '{print $5}' | head -1)
  printf 'size=%-10s rep=%-2s arm=%-5s rc=%-4s oob=%-4s transport=%-10s busbw=%s\n' \
      "$SIZE" "$REP" "$WHICH" "$RC" "${OOB:-NA}" "${TR:-unknown}" "${BW:-NA}"
}

echo "########## 8 ranks / 1 MiB, CXI, no profiler, 10 repeats ##########"
for R in $(seq 1 10); do probe 1048576 "$R" cxi; done

echo "########## size sweep, CXI, no profiler ##########"
for S in 262144 524288 1048576 2097152 4194304 8388608 16777216; do
  probe "$S" 1 cxi
done

echo "########## 8 ranks / 1 MiB, STOCK (Socket) -- does it follow transport? ##########"
for R in $(seq 1 5); do probe 1048576 "$R" stock; done

echo "=== oob probe complete: $OUT ==="
