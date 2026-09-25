#!/bin/bash
# sweep_layers_cuda.sh -- per-layer option sweeps on CUDA/aarch64.
#
# run_all_scales_cuda.sh answers "does every layer work?" with one setting per
# collective. This answers the different question "which of the options each
# layer exposes actually matter, and by how much?" -- by varying ONE option at
# a time against a fixed baseline.
#
# One-at-a-time is deliberate. A full cross product of the axes below is
# thousands of runs and, when something regresses, tells you nothing about
# which axis caused it. Each cell here differs from the baseline in exactly one
# variable, so a slow cell names its own cause.
#
# Axes per layer:
#
#   osu         collective x {allreduce,allgather,alltoall,reduce,bcast,
#                             reduce_scatter,barrier,gather,scatter}
#               buffer     x {device (cuda), host}
#               message    4 KiB .. 256 MiB
#
#   nccl        collective x {all_reduce,all_gather,alltoall,broadcast,
#                             reduce_scatter,reduce,sendrecv,scatter,gather}
#               dtype      x {float,half,bfloat16}      (payload/precision)
#               redop      x {sum,max}                  (reduction kernels)
#               algo       x {default,Ring,Tree}        (NCCL_ALGO)
#               proto      x {default,Simple,LL,LL128}  (NCCL_PROTO)
#
#   torch_dist  collective x {allreduce,allgather,reducescatter,alltoall,
#                             broadcast,reduce,barrier,send_recv}
#               dtype      x {float32,float16,bfloat16}
#               async      x {sync, async_op}           (overlap behaviour)
#
#   nixl        op         x {READ,WRITE}
#               memory     x {VRAM,DRAM}
#               size       4 KiB .. 1 GiB
#
#   fi          provider/domain inventory + per-rail pingpong
#
# Scale is a parameter, not a separate script: DLCOMM_NNODES=1 uses all 4 GPUs
# on one node, DLCOMM_NNODES=2 uses all 8 across two.
#
# Every cell writes one file and one CSV row. A cell that cannot run records
# status=unavailable with a reason; it never records a zero.

set -uo pipefail

W=${DLCOMM_WORKDIR:-$HOME/Dynamo_Slingshot_tara}
NNODES=${DLCOMM_NNODES:?set DLCOMM_NNODES to 1 or 2}
GPUS_PER_NODE=${DLCOMM_GPUS_PER_NODE:-4}
RUN=${DLCOMM_RUN_DIR:-$W/sweeps_${PBS_JOBID:-manual}/${NNODES}node}
LAYERS=${DLCOMM_LAYERS:-fi,osu,nccl,torch_dist,nixl}
QUICK=${DLCOMM_QUICK:-0}      # 1 = tiny message range, for a fast smoke test

mkdir -p "$RUN"
CSV=$RUN/sweep_results.csv
[ -f "$CSV" ] || echo "layer,cell,axis,value,collective,dtype,extra,nodes,ranks,exit,rows,status,detail" > "$CSV"

OSU_GTL=${DLCOMM_OSU_GTL:-$W/bench_install_gtl/libexec/osu-micro-benchmarks/mpi}
NCCL_TESTS=${DLCOMM_NCCL_TESTS:-$W/bench_src/nccl-tests/build}
TORCH_BENCH=${DLCOMM_TORCH_BENCH:-$W/torch_dist_bench.py}
NIXL_BENCH=${DLCOMM_NIXL_BENCH:-$W/nixl_putget_bench.py}
NIXL_MODES=${DLCOMM_NIXL_MODES:-$W/nixl_modes_bench.py}
PY=${DLCOMM_PYTHON:-$W/venv/bin/python}

# PALS lives at /opt/cray/pals on Tara and /opt/cray/pe/pals elsewhere.
# Hardcoding either one makes every cell exit 127.
if [ -n "${DLCOMM_MPIEXEC:-}" ]; then
    MPIEXEC=$DLCOMM_MPIEXEC
else
    MPIEXEC=$(command -v mpiexec 2>/dev/null)
    [ -z "$MPIEXEC" ] && MPIEXEC=$(ls -d /opt/cray/pals/*/bin/mpiexec 2>/dev/null | tail -1)
    [ -z "$MPIEXEC" ] && MPIEXEC=$(ls -d /opt/cray/pe/pals/*/bin/mpiexec 2>/dev/null | tail -1)
fi
[ -x "${MPIEXEC:-}" ] || { echo "FATAL: no usable mpiexec found"; exit 2; }

RANKS=$(( NNODES * GPUS_PER_NODE ))

# Never sort $PBS_NODEFILE: PALS gives rank 0 to its first line as written,
# and reordering points MASTER_ADDR at a node with no rendezvous listening.
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "$PBS_NODEFILE" ]; then
    NODES_ORDERED=$(awk '!seen[$0]++' "$PBS_NODEFILE")
    export MASTER_ADDR=$(echo "$NODES_ORDERED" | head -1)
    echo "$NODES_ORDERED" | head -1 > "$RUN/node1.txt"
    echo "$NODES_ORDERED" | sed -n 2p > "$RUN/node2.txt"
else
    NODES_ORDERED=$(hostname); export MASTER_ADDR=$(hostname)
fi
export MASTER_PORT=${MASTER_PORT:-29531}

if [ "$QUICK" = "1" ]; then
    MSG_MIN=4096;  MSG_MAX=1048576;   NIXL_SIZES=4096,1048576
    # dense ladder still needs >1 point to show a curve
    NIXL_LADDER=4096,65536,1048576
    NIXL_MODE_SIZES=65536,1048576
    NIXL_BATCH_COUNTS=1,4,16
else
    MSG_MIN=4096;  MSG_MAX=268435456; NIXL_SIZES=4096,1048576,268435456,1073741824
    # Dense power-of-2 ladder 4 KiB -> 1 GiB (19 points). The old 4-point set
    # jumped 256x between 1 MiB and 256 MiB, straddling the eager/rendezvous
    # transition and hiding where the fixed per-transfer cost stops dominating.
    NIXL_LADDER=4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912,1073741824
    # three points for the API-coverage modes: below, around, and above the
    # size where the fixed cost stops dominating
    NIXL_MODE_SIZES=65536,16777216,1073741824
    NIXL_BATCH_COUNTS=1,4,16,64,256
fi

echo "=================================================================="
echo " per-layer option sweep | nodes=$NNODES ranks=$RANKS layers=$LAYERS"
echo " run dir : $RUN"
echo " rank0   : $MASTER_ADDR (first nodefile line, unsorted)"
echo " msg     : $MSG_MIN .. $MSG_MAX   quick=$QUICK"
echo "=================================================================="

want() { case ",$LAYERS," in *",$1,"*) return 0;; *) return 1;; esac; }

# csv <layer> <cell> <axis> <value> <coll> <dtype> <extra> <exit> <rows> <status> <detail>
csv() {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$NNODES" "$RANKS" "$8" "$9" "${10}" "${11}" >> "$CSV"
}

# ==================================================================
# LAYER fi
# ==================================================================
if want fi; then
echo ""; echo "########## fi: provider inventory + per-rail pingpong ##########"
FI_BIN_DIR=${DLCOMM_FI_BIN:-$W/shs-libfabric-install/bin}
if [ -x "$FI_BIN_DIR/fi_info" ]; then
    # The custom libfabric build is CUDA-aware, so fi_info links
    # libcudart.so.12. Run it through the NIXL wrapper, which assembles
    # libfabric + NIXL + UCX + CUDA; with only the libfabric dir it exits 127
    # (job 6952) and that was being miscounted as "0 providers".
    bash "$W/pals_nixl_env.sh" "$FI_BIN_DIR/fi_info" -p cxi \
        > "$RUN/fi_info.txt" 2>&1
    FI_INFO_RC=$?
    NPROV=$(grep -ci '^provider:' "$RUN/fi_info.txt")
    echo "fi_info providers=$NPROV rc=$FI_INFO_RC"
    if [ "$FI_INFO_RC" -ne 0 ]; then
        # A loader error is not a measurement. Recording it as providers=0
        # reads as "this node has no CXI fabric", which is how job 6952's
        # missing libcudart.so.12 nearly got filed as a hardware finding.
        WHY=$(grep -aoE '[a-zA-Z0-9_.+-]+\.so[0-9.]*: cannot open shared object' \
              "$RUN/fi_info.txt" | head -1)
        csv fi inventory provider - - - - "$FI_INFO_RC" 0 \
            failed "${WHY:-fi_info_rc_$FI_INFO_RC}"
    else
        csv fi inventory provider cxi - - - 0 "$NPROV" \
            "$([ "$NPROV" -gt 0 ] && echo ok || echo failed)" "providers=$NPROV"
    fi

    # One pingpong per CXI rail. Same-node pingpong is not a wire measurement
    # (CXI has no loopback: fi_domain() returns -38), so it is only run at 2
    # nodes and explicitly skipped at 1 with a reason.
    if [ "$NNODES" -ge 2 ] && [ -x "$FI_BIN_DIR/fi_pingpong" ]; then
        # rank 0 listens, rank 1 connects to it. Running identical args on
        # both ranks makes both sides servers and the cell hangs to SIGTERM.
        SRV_SHORT=$(head -1 "$RUN/node1.txt" | cut -d. -f1)
        FI_LIB=${DLCOMM_FI_LIB:-$W/shs-libfabric-install/lib}
        for rail in 0 1 2 3; do
            F=$RUN/fi_pingpong_cxi$rail.txt
            cat > "$RUN/pp_wrap.sh" <<EOS
#!/bin/bash
# Same CUDA-aware libfabric, so source the full env wrapper rather than
# hand-rolling LD_LIBRARY_PATH (that produced a 127, job 6952).
export FI_CXI_DEVICE_NAME=cxi$rail
if [ "\${PALS_RANKID:-0}" = "0" ]; then
    exec bash "$W/pals_nixl_env.sh" "$FI_BIN_DIR/fi_pingpong" -e rdm -p cxi -S 6291456
else
    sleep 8
    exec bash "$W/pals_nixl_env.sh" "$FI_BIN_DIR/fi_pingpong" -e rdm -p cxi -S 6291456 "$SRV_SHORT"
fi
EOS
            chmod +x "$RUN/pp_wrap.sh"
            timeout 180 $MPIEXEC -n 2 -ppn 1 "$RUN/pp_wrap.sh" > "$F" 2>&1
            RC=$?; ROWS=$(grep -cE '^[[:space:]]*[0-9]' "$F")
            echo "  cxi$rail exit=$RC rows=$ROWS"
            csv fi pingpong rail "cxi$rail" pingpong - - "$RC" "$ROWS" \
                "$([ "$RC" -eq 0 ] && [ "$ROWS" -gt 0 ] && echo ok || echo failed)" "-"
        done
    else
        csv fi pingpong rail all pingpong - - 0 0 unavailable \
            "needs_2_nodes_CXI_has_no_loopback_fi_domain_ret_-38"
        echo "  pingpong skipped: CXI has no loopback, needs 2 nodes"
    fi
else
    csv fi inventory provider cxi - - - 0 0 unavailable "fi_info_not_found"
fi
fi

# ==================================================================
# LAYER osu -- collective x buffer type
# ==================================================================
if want osu; then
echo ""; echo "########## osu: 9 collectives x {device,host} ##########"
OSU_COLL=$OSU_GTL/collective
if [ -x "$OSU_COLL/osu_allreduce" ]; then
    GTL_N=$(ldd "$OSU_COLL/osu_allreduce" 2>/dev/null | grep -ci gtl)
    echo "GTL linked: $GTL_N"
    if [ "$GTL_N" -eq 0 ]; then
        csv osu all buffer device - - - 0 0 unavailable "no_GTL_device_buffers_segfault"
    else
      for coll in allreduce allgather alltoall reduce bcast reduce_scatter barrier gather scatter; do
        BIN=$OSU_COLL/osu_$coll
        [ -x "$BIN" ] || { csv osu "$coll" collective "$coll" "$coll" - - 0 0 unavailable "binary_missing"; continue; }
        for buf in device host; do
            F=$RUN/osu_${coll}_${buf}.txt
            if [ "$buf" = device ]; then
                # MPICH_GPU_SUPPORT_ENABLED is set per-cell, never globally:
                # leaking it into a non-GTL step aborts that step.
                MPICH_GPU_SUPPORT_ENABLED=1 timeout 420 $MPIEXEC -n "$RANKS" -ppn "$GPUS_PER_NODE" \
                    "$W/pals_osu_env.sh" "$BIN" -d cuda -m $MSG_MIN:$MSG_MAX > "$F" 2>&1
            else
                timeout 420 $MPIEXEC -n "$RANKS" -ppn "$GPUS_PER_NODE" \
                    "$W/pals_osu_env.sh" "$BIN" -m $MSG_MIN:$MSG_MAX > "$F" 2>&1
            fi
            # Count lines whose first non-space character is a digit: OSU
            # indents osu_barrier's single latency value, and anchoring on
            # column 1 scored a real measurement as rows=0 -> failed.
            RC=$?; ROWS=$(grep -cE '^[[:space:]]*[0-9]' "$F")
            # An invalid CLI form prints usage and exits 0 -- that is not a run.
            if grep -qE "Usage:|\(null\) \[-d\]" "$F"; then
                ST=failed; DET=usage_printed_invalid_args; ROWS=0
            elif [ "$RC" -eq 0 ] && [ "$ROWS" -gt 0 ]; then ST=ok; DET=-
            else ST=failed; DET="exit_$RC"; fi
            printf '  %-16s %-6s exit=%-3s rows=%-3s %s\n' "$coll" "$buf" "$RC" "$ROWS" "$ST"
            csv osu "${coll}_${buf}" buffer "$buf" "$coll" - - "$RC" "$ROWS" "$ST" "$DET"
        done
      done
    fi
else
    csv osu all collective all - - - 0 0 unavailable "osu_gtl_build_missing"
fi
fi

# ==================================================================
# LAYER nccl -- collective, dtype, redop, algo, proto
# ==================================================================
if want nccl; then
echo ""; echo "########## nccl: collectives x dtype x redop x algo x proto ##########"
if [ -x "$NCCL_TESTS/all_reduce_perf" ]; then
  export LD_LIBRARY_PATH="$W/venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
  # run_nccl <cell> <axis> <value> <bin> <coll> <dtype> <extra> <redop> [env...]
  run_nccl() {
      local cell=$1 axis=$2 value=$3 bin=$4 coll=$5 dt=$6 extra=$7 redop=$8; shift 8
      local F=$RUN/nccl_${cell}.txt
      [ -x "$NCCL_TESTS/$bin" ] || { csv nccl "$cell" "$axis" "$value" "$coll" "$dt" "$extra" 0 0 unavailable "binary_missing"; return; }
      # -o is only meaningful for reducing collectives; passing it to e.g.
      # all_gather makes nccl-tests reject the argument.
      local OPARG=()
      case "$bin" in
        all_reduce_perf|reduce_perf|reduce_scatter_perf) OPARG=(-o "$redop") ;;
      esac
      env "$@" timeout 420 $MPIEXEC -n "$RANKS" -ppn "$GPUS_PER_NODE" \
          "$W/pals_nccl_env.sh" "$NCCL_TESTS/$bin" \
          -b $MSG_MIN -e $MSG_MAX -f 4 -g 1 -d "$dt" "${OPARG[@]}" > "$F" 2>&1
      local RC=$? ROWS; ROWS=$(grep -cE '^ *[0-9]+ ' "$F")
      local ST; { [ "$RC" -eq 0 ] && [ "$ROWS" -gt 0 ]; } && ST=ok || ST=failed
      printf '  %-34s exit=%-3s rows=%-3s %s\n' "$cell" "$RC" "$ROWS" "$ST"
      csv nccl "$cell" "$axis" "$value" "$coll" "$dt" "$extra" "$RC" "$ROWS" "$ST" "-"
  }

  # axis: collective (baseline float/sum/default)
  for pair in "all_reduce:all_reduce_perf" "all_gather:all_gather_perf" \
              "alltoall:alltoall_perf" "broadcast:broadcast_perf" \
              "reduce_scatter:reduce_scatter_perf" "reduce:reduce_perf" \
              "sendrecv:sendrecv_perf" "scatter:scatter_perf" "gather:gather_perf"; do
      run_nccl "coll_${pair%%:*}" collective "${pair%%:*}" "${pair##*:}" "${pair%%:*}" float - sum
  done

  # axis: dtype, on allreduce
  for dt in half bfloat16; do
      run_nccl "dtype_$dt" dtype "$dt" all_reduce_perf all_reduce "$dt" - sum
  done

  # axis: reduction op
  for op in max min prod; do
      run_nccl "redop_$op" redop "$op" all_reduce_perf all_reduce float "op=$op" "$op"
  done

  # axis: NCCL_ALGO -- Ring vs Tree changes the collective's shape
  for algo in Ring Tree; do
      run_nccl "algo_$algo" algo "$algo" all_reduce_perf all_reduce float "algo=$algo" sum \
          NCCL_ALGO="$algo"
  done

  # axis: NCCL_PROTO -- LL/LL128 trade latency against bandwidth
  for proto in Simple LL LL128; do
      run_nccl "proto_$proto" proto "$proto" all_reduce_perf all_reduce float "proto=$proto" sum \
          NCCL_PROTO="$proto"
  done
else
  csv nccl all collective all - - - 0 0 unavailable "nccl_tests_not_built"
fi
fi

# ==================================================================
# LAYER torch_dist -- collective, dtype, async
# ==================================================================
if want torch_dist; then
echo ""; echo "########## torch_dist: collectives x dtype x async ##########"
if [ -f "$TORCH_BENCH" ]; then
  run_torch() {  # run_torch <cell> <axis> <value> <coll> <dtype> <extra-args...>
      local cell=$1 axis=$2 value=$3 coll=$4 dt=$5; shift 5
      local F=$RUN/torch_${cell}.txt
      timeout 420 $MPIEXEC -n "$RANKS" -ppn "$GPUS_PER_NODE" \
          "$W/pals_torch_env.sh" "$PY" "$TORCH_BENCH" \
          --collective "$coll" --dtype "$dt" \
          --min-bytes $MSG_MIN --max-bytes $MSG_MAX \
          --json-out "$RUN/torch_${cell}.json" "$@" > "$F" 2>&1
      local RC=$? ROWS; ROWS=$(grep -cE '^ *[0-9]+ ' "$F")
      local ST; { [ "$RC" -eq 0 ] && [ "$ROWS" -gt 0 ]; } && ST=ok || ST=failed
      printf '  %-34s exit=%-3s rows=%-3s %s\n' "$cell" "$RC" "$ROWS" "$ST"
      csv torch_dist "$cell" "$axis" "$value" "$coll" "$dt" "${*:-}" "$RC" "$ROWS" "$ST" "-"
  }
  for coll in allreduce allgather reducescatter alltoall broadcast reduce barrier send_recv; do
      run_torch "coll_$coll" collective "$coll" "$coll" float32
  done
  for dt in float16 bfloat16; do
      run_torch "dtype_$dt" dtype "$dt" allreduce "$dt"
  done
  run_torch "async_on" async async allreduce float32 --async-op
else
  csv torch_dist all collective all - - - 0 0 unavailable "torch_dist_bench_missing"
fi
fi

# ==================================================================
# LAYER nixl -- op x memory
# ==================================================================
if want nixl; then
echo ""; echo "########## nixl: {READ,WRITE} x {VRAM,DRAM} ##########"
# CXI has no loopback, so a SECOND LIBFABRIC agent on the same node cannot
# construct: createBackend returns NIXL_ERR_BACKEND for every op/mem pair
# (jobs 6951, 6953, placement=same-node). One agent per node across two nodes
# is healthy -- job 6949 got AGENT=OK with both DRAM and VRAM registration OK
# on each node, and 2-node runs report placement=cross-node with no error.
# This is the same loopback limit that already makes fi_pingpong unavailable
# at 1 node (fi_domain() ret=-38).
#
# So report 1 node as out of scope, not as a failure: "failed" sends the next
# reader hunting for a defect that is not there.
if [ "$NNODES" -lt 2 ] && [ -f "$NIXL_BENCH" ]; then
  for op in READ WRITE; do
    for mem in VRAM DRAM; do
      echo "  ${op}_${mem} unsupported (same-node CXI has no loopback)"
      csv nixl "${op}_${mem}" op "$op" "$mem" - - 0 0 \
        unsupported "needs_2_nodes_second_LIBFABRIC_agent_cannot_construct_same_node"
    done
  done
  # The ladder and the API-coverage modes need two agents just as much, so
  # declare them here too. Omitting them would make the 1-node and 2-node
  # cell counts differ for no stated reason.
  echo "  LADDER_VRAM unsupported (same-node CXI has no loopback)"
  csv nixl LADDER_VRAM size ladder putget VRAM - 0 0 \
    unsupported "needs_2_nodes_second_LIBFABRIC_agent_cannot_construct_same_node"
  for MODE in prepped batch notif introspect; do
    echo "  MODE_${MODE} unsupported (same-node CXI has no loopback)"
    csv nixl "MODE_${MODE}" api "$MODE" modes VRAM - 0 0 \
      unsupported "needs_2_nodes_second_LIBFABRIC_agent_cannot_construct_same_node"
  done
elif [ -f "$NIXL_BENCH" ]; then
  EXPECT=$([ "$NNODES" -eq 1 ] && echo same-node || echo cross-node)
  PPN=$([ "$NNODES" -eq 1 ] && echo 2 || echo 1)
  for op in READ WRITE; do
    for mem in VRAM DRAM; do
      CELL="${op}_${mem}"
      SYNC=$RUN/nixl_sync_$CELL; rm -rf "$SYNC"; mkdir -p "$SYNC"
      F=$RUN/nixl_${CELL}.txt
      timeout 600 $MPIEXEC -n 2 -ppn "$PPN" \
          "$W/pals_nixl_env.sh" "$PY" "$NIXL_BENCH" \
          --op "$op" --mem "$mem" --sync-dir "$SYNC" --expect "$EXPECT" \
          --sizes "$NIXL_SIZES" --json-out "$RUN/nixl_${CELL}.json" > "$F" 2>&1
      RC=$?; NP=$(grep -c 'byte-exact check: PASS' "$F" 2>/dev/null)
      # WRITE is unsupported on this CXI stack. It surfaces two ways: the
      # reference repro's -260 "Flags not supported", and postXferReq raising
      # NIXL_ERR_BACKEND (job 6955). READ over the identical path is healthy
      # (22.4 GB/s at 1 GiB, byte-exact), so this is a transport capability
      # limit, not a broken benchmark. Restrict the NIXL_ERR_BACKEND reading
      # to WRITE: on READ that same error means something genuinely wrong.
      if grep -qaE 'Flags not supported|-260' "$F" 2>/dev/null ||
         { [ "$op" = "WRITE" ] && grep -qa 'NIXL_ERR_BACKEND' "$F" 2>/dev/null; }; then
          ST=unsupported; DET="cxi_rma_write_unsupported"
      elif [ "$RC" -eq 0 ] && [ "$NP" -gt 0 ]; then ST=ok; DET=-
      else ST=failed; DET="exit_$RC"; fi
      printf '  %-14s exit=%-3s pass=%-3s %s %s\n' "$CELL" "$RC" "$NP" "$ST" "$DET"
      csv nixl "$CELL" op "$op" putget "$mem" "$EXPECT" "$RC" "$NP" "$ST" "$DET"
    done
  done

  # ---- dense size ladder -------------------------------------------------
  # 19 power-of-2 points, READ/VRAM only. The 4-point set above jumps 256x
  # between 1 MiB and 256 MiB; this resolves where the fixed per-transfer
  # cost stops dominating. WRITE is skipped: unsupported on CXI.
  SYNC=$RUN/nixl_sync_ladder; rm -rf "$SYNC"; mkdir -p "$SYNC"
  F=$RUN/nixl_LADDER_VRAM.txt
  timeout 900 $MPIEXEC -n 2 -ppn "$PPN" \
      "$W/pals_nixl_env.sh" "$PY" "$NIXL_BENCH" \
      --op READ --mem VRAM --sync-dir "$SYNC" --expect "$EXPECT" \
      --sizes "$NIXL_LADDER" --json-out "$RUN/nixl_LADDER_VRAM.json" > "$F" 2>&1
  RC=$?; NP=$(grep -c 'byte-exact check: PASS' "$F" 2>/dev/null)
  NPTS=$(echo "$NIXL_LADDER" | tr ',' '\n' | grep -c .)
  if [ "$RC" -eq 0 ] && [ "$NP" -eq "$NPTS" ]; then ST=ok; DET="${NP}_of_${NPTS}_sizes"
  elif [ "$RC" -eq 0 ]; then ST=failed; DET="only_${NP}_of_${NPTS}_sizes_verified"
  else ST=failed; DET="exit_$RC"; fi
  printf '  %-14s exit=%-3s pass=%-3s %s %s\n' "LADDER_VRAM" "$RC" "$NP" "$ST" "$DET"
  csv nixl LADDER_VRAM size ladder putget VRAM "$EXPECT" "$RC" "$NP" "$ST" "$DET"

  # ---- API-coverage modes ------------------------------------------------
  # prepped/batch/notif/introspect cover the 26 nixl_agent methods the basic
  # lifecycle never touches. Separate cells so a slow or broken mode names
  # itself instead of hiding inside an aggregate.
  if [ -f "$NIXL_MODES" ]; then
    for MODE in prepped batch notif introspect; do
      CELL="MODE_${MODE}"
      SYNC=$RUN/nixl_sync_$CELL; rm -rf "$SYNC"; mkdir -p "$SYNC"
      F=$RUN/nixl_${CELL}.txt
      # batch registers up to 256 buffers per point and walks 5 points, so it
      # needs a longer leash than the others. Job 6958 died at the n=16
      # barrier on the 300 s default while its transfers were healthy.
      case "$MODE" in
        batch) CT=2400; BT=900 ;;
        *)     CT=900;  BT=300 ;;
      esac
      timeout $CT env NIXL_SYNC_TIMEOUT=$BT $MPIEXEC -n 2 -ppn "$PPN" \
          "$W/pals_nixl_env.sh" "$PY" "$NIXL_MODES" \
          --mode "$MODE" --op READ --mem VRAM --sync-dir "$SYNC" \
          --expect "$EXPECT" --sizes "$NIXL_MODE_SIZES" \
          --batch-counts "$NIXL_BATCH_COUNTS" \
          --json-out "$RUN/nixl_${CELL}.json" > "$F" 2>&1
      RC=$?; NP=$(grep -c 'byte-exact check: PASS' "$F" 2>/dev/null)
      # introspect makes one transfer, so one PASS is the full score there.
      if [ "$RC" -eq 0 ] && [ "$NP" -gt 0 ]; then ST=ok; DET="${NP}_verified"
      elif [ "$RC" -eq 124 ] || [ "$RC" -eq 143 ]; then ST=failed; DET="timeout_$RC"
      else ST=failed; DET="exit_$RC"; fi
      printf '  %-14s exit=%-3s pass=%-3s %s %s\n' "$CELL" "$RC" "$NP" "$ST" "$DET"
      csv nixl "$CELL" api "$MODE" modes VRAM "$EXPECT" "$RC" "$NP" "$ST" "$DET"
    done
  else
    csv nixl MODE_all api all modes - - 0 0 unavailable "nixl_modes_bench_missing"
  fi
else
  csv nixl all op all - - - 0 0 unavailable "nixl_putget_bench_missing"
fi
fi

echo ""
echo "=================================================================="
echo "CSV=$CSV"
printf 'cells: %s ok, %s failed, %s unavailable, %s unsupported\n' \
    "$(grep -c ',ok,' "$CSV")" "$(grep -c ',failed,' "$CSV")" \
    "$(grep -c ',unavailable,' "$CSV")" "$(grep -c ',unsupported,' "$CSV")"
echo "SWEEP_DONE nodes=$NNODES"
echo "=================================================================="
