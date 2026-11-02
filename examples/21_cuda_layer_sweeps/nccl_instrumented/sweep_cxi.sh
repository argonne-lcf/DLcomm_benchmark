#!/bin/bash
#PBS -N nccvar_cxi
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# NCCL variant sweep (collective / dtype / redop / algo / proto) on CXI.
#
# The original sweep ran these 19 cells over NET/Socket, so every absolute
# number in it is a TCP number. Relative ordering within an axis may survive
# the transport change -- but that is an assumption, and this job tests it
# rather than asserting it. Same axes, same sizes, CXI transport.
#
# Per cell we record: transport, validation status, peak busbw. A cell whose
# transport is unknown, or whose output failed validation, is recorded as
# such and never contributes a number.
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

NCCL_TESTS=$W/bench_src/nccl-tests/build
MPIEXEC=$(ls /opt/cray/pals/*/bin/mpiexec 2>/dev/null | head -1)
[ -x "$MPIEXEC" ] || MPIEXEC=$(ls /opt/cray/pe/pals/*/bin/mpiexec 2>/dev/null | head -1)

OUT=$W/sweepcxi_${PBS_JOBID}
mkdir -p "$OUT"
CSV=$OUT/sweep_cxi.csv
echo "cell,axis,value,coll,dtype,ranks,peak_busbw,peak_algbw,at_bytes,status,transport" > "$CSV"

RANKS=8
PPN=4
MSG_MIN=1048576
MSG_MAX=268435456

run_nccl() {
  local cell=$1 axis=$2 value=$3 bin=$4 coll=$5 dt=$6 op=$7
  shift 7
  local F="$OUT/${cell}.txt"
  local OPARG=()
  [ "$op" != "-" ] && OPARG=(-o "$op")

  # Extra NCCL_* knobs (NCCL_ALGO / NCCL_PROTO) arrive as KEY=VAL and must be
  # passed via --env. LD_LIBRARY_PATH is REPLACED by --env, not extended, so
  # every needed dir is listed explicitly or ranks die with exit 127 -> 143,
  # which reads exactly like a fabric hang.
  local ENVARGS=(--env LD_LIBRARY_PATH="$LDP"
                 --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so"
                 --env NCCL_DEBUG=INFO)
  local kv
  for kv in "$@"; do ENVARGS+=(--env "$kv"); done

  timeout 420 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
      "${ENVARGS[@]}" "$W/pals_nccl_env.sh" "$NCCL_TESTS/$bin" \
      -b $MSG_MIN -e $MSG_MAX -f 4 -g 1 -d "$dt" "${OPARG[@]}" > "$F" 2>&1
  local RC=$?

  local TR
  TR=$(grep -ao 'Using network [A-Za-z]*' "$F" | head -1 | awk '{print $3}')
  [ -n "$TR" ] || TR=unknown

  if grep -qa 'error while loading shared libraries' "$F"; then
    echo "$cell,$axis,$value,$coll,$dt,$RANKS,-,-,-,libfail,$TR" >> "$CSV"
    printf '  %-26s %s\n' "$cell" "LIBFAIL"
    return
  fi
  # A run that computed the wrong answer still prints a bandwidth (job 7018).
  if grep -qa 'Out of bounds values : [1-9]' "$F"; then
    local OOB
    OOB=$(grep -ao 'Out of bounds values : [0-9]*' "$F" | head -1 | awk '{print $NF}')
    echo "$cell,$axis,$value,$coll,$dt,$RANKS,-,-,-,validation_oob_$OOB,$TR" >> "$CSV"
    printf '  %-26s %s\n' "$cell" "VALIDATION_FAILED($OOB)"
    return
  fi

  # Peak busbw across the size sweep. Rows are matched structurally: stderr
  # interleaves into stdout under mpiexec and eats the leading fields, so
  # "^ +<size> " matches nothing or matches an INFO line.
  read -r PB PA PS < <(awk '
    /NCCL INFO|NET\/OFI/ { next }
    {
      n = 0
      for (i = 1; i <= NF; i++) if ($i ~ /^-?[0-9]+(\.[0-9]+)?$/) n++
      bw = ""; alg = ""; sz = ""
      if (NF == 13 && $1 ~ /^[0-9]+$/ && $6 ~ /^[0-9.]+$/) { alg = $7; bw = $8; sz = $1 }
      else if (NF == 8 && n == 8 && $4 ~ /^[0-9]+$/) { alg = $2; bw = $3; sz = "unknown" }
      if (bw != "" && bw + 0 > best) { best = bw + 0; balg = alg; bsz = sz }
    }
    END { if (best > 0) print best, balg, bsz }' "$F")

  if [ -z "${PB:-}" ]; then
    echo "$cell,$axis,$value,$coll,$dt,$RANKS,-,-,-,no_rows,$TR" >> "$CSV"
    printf '  %-26s %s\n' "$cell" "NO_ROWS rc=$RC"
    return
  fi

  echo "$cell,$axis,$value,$coll,$dt,$RANKS,$PB,$PA,$PS,ok,$TR" >> "$CSV"
  printf '  %-26s peak_busbw=%-9s transport=%s\n' "$cell" "$PB" "$TR"
}

echo "=== axis: collective ==="
for pair in "all_reduce:all_reduce_perf" "all_gather:all_gather_perf" \
            "alltoall:alltoall_perf" "broadcast:broadcast_perf" \
            "reduce_scatter:reduce_scatter_perf" "reduce:reduce_perf" \
            "sendrecv:sendrecv_perf" "scatter:scatter_perf" \
            "gather:gather_perf"; do
    run_nccl "coll_${pair%%:*}" collective "${pair%%:*}" "${pair##*:}" "${pair%%:*}" float sum
done

echo "=== axis: dtype ==="
for dt in half bfloat16; do
    run_nccl "dtype_$dt" dtype "$dt" all_reduce_perf all_reduce "$dt" sum
done

echo "=== axis: reduction op ==="
for op in max min prod; do
    run_nccl "redop_$op" redop "$op" all_reduce_perf all_reduce float "$op"
done

echo "=== axis: NCCL_ALGO ==="
for algo in Ring Tree; do
    run_nccl "algo_$algo" algo "$algo" all_reduce_perf all_reduce float sum \
        NCCL_ALGO="$algo"
done

echo "=== axis: NCCL_PROTO ==="
for proto in Simple LL LL128; do
    run_nccl "proto_$proto" proto "$proto" all_reduce_perf all_reduce float sum \
        NCCL_PROTO="$proto"
done

echo "=== sweep complete: $CSV ==="
cat "$CSV"
