#!/bin/bash
#PBS -N nccgap
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Fill the two real holes in the layer ladder:
#   1. fi_pingpong  -- the NIC ceiling. Never measured; without it there is
#                      no denominator for "how much does each layer give up".
#   2. collectives other than allreduce at 2/4/8 ranks on BOTH scales.
#      The variant sweep only ran them at 8 ranks / 2 nodes.
#
# Layers per cell: osu (MPI) -> nccl_cxi (NCCL over CXI) -> torch.distributed.
# Every NCCL/torch run asserts transport and validation; anything unproven is
# recorded as such and never published as a measurement.

set -u
W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
OUT=$W/nccgap_${PBS_JOBID%%.*}
mkdir -p "$OUT"
CSV=$OUT/gap.csv
echo "scale,nodes,ranks,layer,op,bytes,usec,gbps,transport,oob,status" > "$CSV"

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
NCCLT=$W/bench_src/nccl-tests/build
NCCLENV=$W/pals_nccl_env.sh
# OSU 7.5 puts blocking collectives in collective/blocking, and the tree is
# named -7.5 not -7.5-1. Job 7038 recorded binary_missing for every OSU cell
# because of that path.
OSU=$W/bench_install_gtl/libexec/osu-micro-benchmarks/mpi/collective
FIB=$W/shs-libfabric-install/bin
PY=$W/venv/bin/python

# --env LD_LIBRARY_PATH REPLACES the module's value rather than extending it,
# so the CUDA runtime dirs must be restated explicitly. Job 7038 lost them and
# every rank died with "libcudart.so.13: cannot open shared object file"
# (exit 127), which looks exactly like a fabric fault in the job log.
INSTR=$W/nccl_instr
FI=$W/shs-libfabric-install
CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/REDIST/cuda/13.0/targets/sbsa-linux/lib
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/lib
NCCL_CXI_LDP="$INSTR/lib:$CUDA13_LIB:$CUDA12_LIB:$FI/lib:${LD_LIBRARY_PATH:-}"

if [ ! -f "$INSTR/lib/libnccl-net.so" ]; then
  echo "FATAL: $INSTR/lib/libnccl-net.so missing; would silently use Socket" >&2
  exit 1
fi

CXI_ENV=(
  --env LD_LIBRARY_PATH="$NCCL_CXI_LDP"
  --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so"
  --env NCCL_DEBUG=INFO
  --env NCCL_DEBUG_FILE=$OUT/nccldbg.%h.%p
)

# PALS: rank 0 is the FIRST line of the nodefile, not the sorted-first host.
mapfile -t HOSTS < <(awk '!seen[$0]++' "$PBS_NODEFILE")
echo "nodes: ${HOSTS[*]}"

row() { echo "$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11}" >> "$CSV"; }

# ---- parse one nccl-tests run: busbw, transport, validation --------------
nccl_cell() {
  local scale=$1 nodes=$2 ranks=$3 ppn=$4 bin=$5 op=$6 size=$7 log=$8
  local bw usec tr oob st
  # Structural parse: the data row has 13 fields and numeric size/time.
  read -r usec bw < <(awk 'NF==13 && $1 ~ /^[0-9]+$/ && $6 ~ /^[0-9.]+$/ {print $6, $8}' "$log" | tail -1)
  tr=unknown
  # NCCL_DEBUG_FILE keeps INFO out of the data stream, so attest from there.
  if grep -qa 'Using network Libfabric' "$log" $OUT/nccldbg.* 2>/dev/null; then
    tr=Libfabric
  elif grep -qa 'Using network Socket' "$log" $OUT/nccldbg.* 2>/dev/null; then
    tr=Socket
  fi
  oob=$(grep -ao 'Out of bounds values : [0-9]*' "$log" | tail -1 | awk '{print $NF}')
  [ -z "$oob" ] && oob=NA
  st=ok
  [ -z "${bw:-}" ] && { st=no_result; bw=-; usec=-; }
  [ "$oob" != "0" ] && [ "$oob" != "NA" ] && st=validation_oob_$oob
  [ "$tr" = "unknown" ] && [ "$st" = "ok" ] && st=transport_unproven
  row "$scale" "$nodes" "$ranks" nccl_cxi "$op" "$size" "$usec" "$bw" "$tr" "$oob" "$st"
}

# ---- layer 1: fi_pingpong, the NIC ceiling -------------------------------
# 2 processes, host memory, no collective, no GPU. This is the number every
# higher layer is measured against.
echo "=== fi_pingpong NIC ceiling ==="
if [ -x "$FIB/fi_pingpong" ]; then
  SRV=${HOSTS[0]}; CLI=${HOSTS[1]}
  for SZ in 1048576 16777216; do
    L=$OUT/fi_pp_$SZ.txt
    timeout 180 ssh "$SRV" "$FIB/fi_pingpong -p cxi -E -S $SZ" > "$L" 2>&1 &
    sleep 3
    timeout 180 ssh "$CLI" "$FIB/fi_pingpong -p cxi -E -S $SZ $SRV" >> "$L" 2>&1
    wait 2>/dev/null
    # fi_pingpong prints: <iters> <bytes> ... <MB/s>
    GB=$(awk '/MB\/sec|MB\/s/ {for(i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/) v=$i} END{if (v) printf "%.2f", v/1000}' "$L")
    [ -z "$GB" ] && { row fi_pingpong 2 2 fi pingpong "$SZ" - - cxi NA no_result; } \
                 || { row fi_pingpong 2 2 fi pingpong "$SZ" - "$GB" cxi NA ok; }
  done
else
  row fi_pingpong 2 2 fi pingpong 0 - - cxi NA binary_missing
fi

# ---- layers 2-4 across the rank ladder, several collectives --------------
# 1 node: 2 and 4 ranks. 2 nodes: 2, 4 and 8 ranks.
run_scale() {
  local scale=$1 nodes=$2; shift 2
  for RANKS in "$@"; do
    PPN=$(( RANKS / nodes )); [ "$PPN" -lt 1 ] && PPN=1
    for SIZE in 16777216; do
      for OP in all_reduce all_gather reduce_scatter broadcast alltoall; do
        BIN=$NCCLT/${OP}_perf
        [ -x "$BIN" ] || { row "$scale" "$nodes" "$RANKS" nccl_cxi "$OP" "$SIZE" - - - NA binary_missing; continue; }
        L=$OUT/${scale}_r${RANKS}_${OP}.txt
        rm -f $OUT/nccldbg.* 2>/dev/null
        timeout 420 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
          "${CXI_ENV[@]}" "$NCCLENV" "$BIN" -b $SIZE -e $SIZE -f 1 -g 1 -c 1 > "$L" 2>&1
        nccl_cell "$scale" "$nodes" "$RANKS" "$PPN" "$BIN" "$OP" "$SIZE" "$L"
      done

      # MPI reference for the same shapes (osu)
      for PAIR in "osu_allreduce all_reduce" "osu_allgather all_gather" "osu_alltoall alltoall"; do
        set -- $PAIR; OB=$1; OP=$2
        [ -x "$OSU/$OB" ] || { row "$scale" "$nodes" "$RANKS" osu "$OP" "$SIZE" - - mpi NA binary_missing; continue; }
        L=$OUT/${scale}_r${RANKS}_${OB}.txt
        MPICH_GPU_SUPPORT_ENABLED=1 \
        timeout 420 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
          "$W/pals_osu_env.sh" "$OSU/$OB" -d cuda -m $SIZE:$SIZE > "$L" 2>&1
        US=$(awk -v s=$SIZE '$1==s {print $2}' "$L" | tail -1)
        if [ -n "$US" ]; then
          # allreduce busbw = 2*(n-1)/n * bytes / t  (same as nccl-tests).
          # allgather/alltoall move (n-1)/n * bytes, so the factor differs.
          case "$OP" in
            all_reduce) FACT=2 ;;
            *)          FACT=1 ;;
          esac
          GB=$($PY -c "print(f'{$FACT*$SIZE*($RANKS-1)/$RANKS/($US*1e-6)/1e9:.2f}')" 2>/dev/null || echo -)
          row "$scale" "$nodes" "$RANKS" osu "$OP" "$SIZE" "$US" "$GB" mpi NA ok
        else
          row "$scale" "$nodes" "$RANKS" osu "$OP" "$SIZE" - - mpi NA no_result
        fi
      done

      # ---- layer 4: torch.distributed over the SAME CXI plugin -----------
      # This is the one layer with no attested CXI number anywhere in the
      # study: every previous torch row ran on the TCP fallback, so the
      # "torch overhead" in the old report was really a transport delta.
      TL=$OUT/${scale}_r${RANKS}_torch.txt
      rm -f $OUT/nccldbg.* 2>/dev/null
      MADDR=${HOSTS[0]}
      MPORT=$(( 29500 + (RANDOM % 2000) ))
      timeout 420 "$MPIEXEC" -n "$RANKS" -ppn "$PPN" --hostfile "$PBS_NODEFILE" \
        "${CXI_ENV[@]}" \
        --env MASTER_ADDR="$MADDR" --env MASTER_PORT="$MPORT" \
        --env WORLD_SIZE="$RANKS" \
        "$NCCLENV" "$PY" "$W/torch_allreduce.py" \
        --bytes $SIZE --iters 20 > "$TL" 2>&1
      TUS=$(awk '/^RESULT usec=/ {print $2}' "$TL" | tail -1 | cut -d= -f2)
      TGB=$(awk '/^RESULT usec=/ {print $3}' "$TL" | tail -1 | cut -d= -f2)
      TTR=unknown
      if grep -qa 'Using network Libfabric' "$TL" $OUT/nccldbg.* 2>/dev/null; then
        TTR=Libfabric
      elif grep -qa 'Using network Socket' "$TL" $OUT/nccldbg.* 2>/dev/null; then
        TTR=Socket
      fi
      if [ -n "${TGB:-}" ] && [ "$TGB" != INVALID ]; then
        TST=ok; [ "$TTR" = unknown ] && TST=transport_unproven
        row "$scale" "$nodes" "$RANKS" torch all_reduce "$SIZE" "$TUS" "$TGB" "$TTR" NA "$TST"
      elif [ "${TGB:-}" = INVALID ]; then
        WR=$(awk '/^VALIDATION_FAILED/ {print $2}' "$TL" | cut -d= -f2)
        row "$scale" "$nodes" "$RANKS" torch all_reduce "$SIZE" "$TUS" - "$TTR" "$WR" validation_failed
      else
        row "$scale" "$nodes" "$RANKS" torch all_reduce "$SIZE" - - "$TTR" NA no_result
      fi
    done
  done
}

echo "=== 1 node ladder ==="
run_scale 1node 1 2 4
echo "=== 2 node ladder ==="
run_scale 2node 2 2 4 8

echo ""
echo "=== gap ladder complete ==="
column -s, -t "$CSV"
