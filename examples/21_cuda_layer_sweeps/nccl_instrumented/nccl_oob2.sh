#!/bin/bash
#PBS -N nccoob2
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Characterise the CXI allreduce correctness failure found in job 7030.
#
# Established there: 8 ranks / 1 MiB / CXI produced wrong results in 5 of 10
# runs. Socket at the same size and rank count: 0 of 5. Sizes 256 KiB..16 MiB
# at one sample each: clean. So it tracks the transport AND is size-specific.
#
# Open questions this job answers, each with a failure RATE not one sample:
#   1. Is 1 MiB really special, or did neighbouring sizes just get lucky?
#      -> 6 repeats at each nearby size.
#   2. Does it need 8 ranks / 2 nodes, or does it appear at 4 and 2?
#   3. Does NCCL_PROTO change it? 1 MiB is near the LL128->Simple threshold,
#      and proto_LL128 failed in the 7028 sweep. If pinning Simple fixes it,
#      that is both a strong hypothesis and a usable workaround.
#   4. Does NCCL_ALGO (Ring vs Tree) matter?
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

OUT=$W/nccoob2_${PBS_JOBID}
mkdir -p "$OUT"
CSV=$OUT/oob2.csv
echo "group,size,ranks,ppn,proto,algo,rep,rc,oob,transport" > "$CSV"

awk '!seen[$0]++' "$PBS_NODEFILE" > "$OUT/nodes.txt"
echo "NODES=$(tr '\n' ',' < "$OUT/nodes.txt")"

probe() {
  local GROUP=$1 SIZE=$2 RANKS=$3 PPN=$4 PROTO=$5 ALGO=$6 REP=$7
  local F="$OUT/${GROUP}_s${SIZE}_n${RANKS}_p${PROTO}_a${ALGO}_r${REP}.txt"
  local EXTRA=()
  [ "$PROTO" != "-" ] && EXTRA+=(--env NCCL_PROTO="$PROTO")
  [ "$ALGO"  != "-" ] && EXTRA+=(--env NCCL_ALGO="$ALGO")

  timeout 300 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
      --env LD_LIBRARY_PATH="$LDP" \
      --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so" \
      --env NCCL_DEBUG=WARN "${EXTRA[@]}" \
      "$W/pals_nccl_env.sh" "$NB" \
      -b "$SIZE" -e "$SIZE" -f 1 -g 1 -n 20 > "$F" 2>&1
  local RC=$?

  local OOB TR
  OOB=$(grep -ao 'Out of bounds values : [0-9]*' "$F" | head -1 | awk '{print $NF}')
  TR=$(grep -ao 'Using network [A-Za-z]*' "$F" | head -1 | awk '{print $3}')
  echo "$GROUP,$SIZE,$RANKS,$PPN,$PROTO,$ALGO,$REP,$RC,${OOB:-NA},${TR:-unknown}" >> "$CSV"
  printf '%-10s size=%-9s n=%-2s proto=%-7s algo=%-5s rep=%-2s oob=%s\n' \
      "$GROUP" "$SIZE" "$RANKS" "$PROTO" "$ALGO" "$REP" "${OOB:-NA}"
}

echo "########## 1: size boundary, 6 repeats each, 8 ranks ##########"
for S in 524288 786432 1048576 1310720 2097152; do
  for R in 1 2 3 4 5 6; do probe sizescan "$S" 8 4 - - "$R"; done
done

echo "########## 2: rank scaling at 1 MiB ##########"
for R in 1 2 3 4 5 6; do probe rank2 1048576 2 1 - - "$R"; done
for R in 1 2 3 4 5 6; do probe rank4 1048576 4 2 - - "$R"; done
for R in 1 2 3 4 5 6; do probe rank8_1node 1048576 4 4 - - "$R"; done

echo "########## 3: NCCL_PROTO at 1 MiB, 8 ranks ##########"
for P in Simple LL LL128; do
  for R in 1 2 3 4 5 6; do probe "proto_$P" 1048576 8 4 "$P" - "$R"; done
done

echo "########## 4: NCCL_ALGO at 1 MiB, 8 ranks ##########"
for A in Ring Tree; do
  for R in 1 2 3 4 5 6; do probe "algo_$A" 1048576 8 4 - "$A" "$R"; done
done

echo "=== done: $CSV ==="
awk -F, 'NR>1 { t[$1]++; if ($9 != "0" && $9 != "NA") f[$1]++ }
  END { printf "\n%-14s %8s\n", "group", "fail/total";
        for (g in t) printf "%-14s %5d/%-4d\n", g, f[g] + 0, t[g] }' "$CSV" | sort
