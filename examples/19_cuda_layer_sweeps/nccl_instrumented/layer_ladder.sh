#!/bin/bash
# Rank ladder: compare LAYERS against each other at MATCHED rank counts.
#
# The per-layer sweep compared 1 node vs 2 nodes within a layer. That cannot
# answer "where is the cost going between layers", because each layer was
# run at a different rank count (NIXL/fi at 2 ranks, OSU/NCCL/torch at 4-8).
# Comparing them directly was apples-to-oranges.
#
# Here every layer runs the SAME operation at the SAME message size at the
# SAME rank count, stepping 2 -> 4 -> 8 GPUs. Then the difference between two
# layers in a row is attributable to the software layer, not the geometry.
#
# Ladder:
#   1 node : 2 ranks (ppn 2), 4 ranks (ppn 4)
#   2 nodes: 2 ranks (ppn 1), 4 ranks (ppn 2), 8 ranks (ppn 4)
#
# Layers that are structurally 2-rank only (fi_pingpong, NIXL pairwise READ)
# are recorded as n/a above 2 ranks rather than silently omitted -- an empty
# cell must never be confusable with a slow one.
set -uo pipefail

W=${DLCOMM_WORKDIR:?}
RUN=${DLCOMM_RUN_DIR:?}
NNODES=${DLCOMM_NNODES:?}
SIZES=${DLCOMM_LADDER_SIZES:-16777216,268435456}
ITERS=${DLCOMM_LADDER_ITERS:-20}
mkdir -p "$RUN"

CSV=$RUN/ladder_results.csv
[ -f "$CSV" ] || echo "layer,op,bytes,nodes,ranks,ppn,usec,gbps,status,detail" > "$CSV"

PY=${DLCOMM_PYTHON:-$W/venv/bin/python}
OSU=$W/bench_install_gtl/libexec/osu-micro-benchmarks/mpi/collective
NCCLT=$W/bench_src/nccl-tests/build
MPIEXEC=$(command -v mpiexec)
[ -x "$MPIEXEC" ] || MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec

# Instrumented NCCL + aws-ofi-nccl (jobs 7002/7010). INSTR/lib must come
# FIRST so its libnccl.so.2 wins over the venv copy.
INSTR=$W/nccl_instr
FI=$W/shs-libfabric-install
CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/REDIST/cuda/13.0/targets/sbsa-linux/lib
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/lib
NCCL_CXI_LDP="$INSTR/lib:$CUDA13_LIB:$CUDA12_LIB:$FI/lib:${LD_LIBRARY_PATH:-}"

# Fail loudly if the plugin is absent. Without this the nccl_cxi arm would
# just fall back to Socket and quietly produce a duplicate of the stock arm,
# which is the exact failure mode this whole exercise exists to expose.
if [ ! -f "$INSTR/lib/libnccl-net.so" ]; then
  echo "FATAL: $INSTR/lib/libnccl-net.so missing; nccl_cxi arm would silently" >&2
  echo "       fall back to Socket and duplicate the stock arm. Build it first." >&2
  exit 2
fi

# rank 0 must be the FIRST nodefile line; PBS does not sort it.
if [ -n "${PBS_NODEFILE:-}" ]; then
  NODES_ORDERED=$(awk '!seen[$0]++' "$PBS_NODEFILE")
  export MASTER_ADDR=$(echo "$NODES_ORDERED" | head -1)
else
  export MASTER_ADDR=$(hostname)
fi
export MASTER_PORT=${MASTER_PORT:-29537}

row() {  # row <layer> <op> <bytes> <ranks> <ppn> <usec> <gbps> <status> <detail>
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$1" "$2" "$3" "$NNODES" "$4" "$5" "$6" "$7" "$8" "$9" >> "$CSV"
  printf '  %-12s %-10s %10s B  ranks=%-2s ppn=%-2s %12s us %10s GB/s  %s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
}

# ---------------------------------------------------------------
# Which (ranks, ppn) pairs are legal at this node count?
# ---------------------------------------------------------------
if [ "$NNODES" = "1" ]; then
  LADDER="2:2 4:4"
else
  LADDER="2:1 4:2 8:4"
fi

echo "=============================================================="
echo " RANK LADDER  nodes=$NNODES  sizes=$SIZES  iters=$ITERS"
echo " ladder (ranks:ppn) = $LADDER"
echo "=============================================================="

for PAIR in $LADDER; do
  RANKS=${PAIR%%:*}
  PPN=${PAIR##*:}
  echo ""
  echo "########## ranks=$RANKS  ppn=$PPN  nodes=$NNODES ##########"

  IFS=',' read -ra SZ <<< "$SIZES"
  for B in "${SZ[@]}"; do

    # ---------- layer 2: OSU allreduce (MPI, device buffers) ----------
    # MPICH_GPU_SUPPORT_ENABLED must be set for a GTL-linked binary passing
    # DEVICE pointers; without it MPICH reads device memory as host memory
    # and the rank dies with signal 11. Scoped per-cell, never globally --
    # leaking it into a non-GTL step aborts that step instead.
    F=$RUN/ladder_osu_r${RANKS}_${B}.txt
    MPICH_GPU_SUPPORT_ENABLED=1 timeout 300 $MPIEXEC -n "$RANKS" -ppn "$PPN" \
        "$W/pals_osu_env.sh" "$OSU/osu_allreduce" -d cuda -m "${B}:${B}" \
        > "$F" 2>&1
    RC=$?
    US=$(grep -aE "^$B " "$F" | awk '{print $2}' | head -1)
    if [ "$RC" = 0 ] && [ -n "$US" ]; then
      GB=$(awk -v b="$B" -v u="$US" 'BEGIN{printf "%.3f", (b/1e9)/(u/1e6)}')
      row osu allreduce "$B" "$RANKS" "$PPN" "$US" "$GB" ok measured
    else
      row osu allreduce "$B" "$RANKS" "$PPN" - - failed "exit_$RC"
    fi

    # ---------- layer 3: NCCL all_reduce ----------
    # Two arms. 'nccl' is the stock venv build, which has no libnccl-net.so
    # and therefore silently selects NET/Socket -- kernel TCP over Slingshot.
    # 'nccl_cxi' is the instrumented build plus aws-ofi-nccl, which reaches
    # the CXI provider. Job 7012 measured 6.39 -> 124.01 GB/s between them at
    # 8 ranks, so every previously published 2-node NCCL number is a TCP
    # number. Both arms run here, back to back, at the same rank count.
    for ARM in nccl nccl_cxi; do
      F=$RUN/ladder_${ARM}_r${RANKS}_${B}.txt
      if [ "$ARM" = "nccl_cxi" ]; then
        # --env LD_LIBRARY_PATH REPLACES the module value rather than
        # extending it, so both CUDA runtimes must be named explicitly:
        # .so.13 for all_reduce_perf, .so.12 for libfabric and the plugin.
        # Omitting them kills every rank with exit 127 -> job 143, which
        # reads exactly like a fabric hang (job 7011).
        #
        # NCCL_DEBUG=INFO is required, not cosmetic: the wrapper defaults to
        # WARN, which suppresses the "Using network X" line and leaves every
        # cell recorded as net_unknown -- i.e. a fast number with no proof of
        # which transport produced it.
        timeout 300 $MPIEXEC -n "$RANKS" -ppn "$PPN" \
            --env LD_LIBRARY_PATH="$NCCL_CXI_LDP" \
            --env NCCL_NET_PLUGIN="$INSTR/lib/libnccl-net.so" \
            --env NCCL_DEBUG=INFO \
            "$W/pals_nccl_env.sh" "$NCCLT/all_reduce_perf" \
            -b "$B" -e "$B" -f 1 -g 1 -n "$ITERS" > "$F" 2>&1
      else
        timeout 300 $MPIEXEC -n "$RANKS" -ppn "$PPN" \
            --env NCCL_DEBUG=INFO \
            "$W/pals_nccl_env.sh" "$NCCLT/all_reduce_perf" \
            -b "$B" -e "$B" -f 1 -g 1 -n "$ITERS" > "$F" 2>&1
      fi
      RC=$?

      # A library-resolution failure exits 127 and PALS reports 143, which is
      # indistinguishable from a fabric timeout in the CSV. Catch it by name.
      if grep -qa 'error while loading shared libraries' "$F"; then
        row "$ARM" allreduce "$B" "$RANKS" "$PPN" - - failed lib_resolution
        continue
      fi

      # Which transport actually carried it. Without this a fast number is
      # unattributable: the whole point is Socket vs Libfabric.
      TR=$(grep -ao 'Using network [A-Za-z]*' "$F" | head -1 | awk '{print $3}')
      [ -n "$TR" ] || TR=unknown

      # nccl-tests validates every result against a reference reduction and
      # reports "Out of bounds values : N FAILED". It still prints a bandwidth
      # figure and can still exit 0, so a numerically WRONG run looks like a
      # good measurement. Job 7018 hit exactly this (8 ranks, 1 MiB, 8 FAILED,
      # 3.148 GB/s printed). Never publish a cell whose output was wrong.
      if grep -qa 'Out of bounds values : [1-9]' "$F"; then
        OOB=$(grep -ao 'Out of bounds values : [0-9]*' "$F" | head -1 | awk '{print $NF}')
        row "$ARM" allreduce "$B" "$RANKS" "$PPN" - - failed "validation_oob_${OOB}_net_$TR"
        continue
      fi

      # Structural parse. NCCL_DEBUG writes to stderr and mpiexec interleaves
      # it into stdout mid-line, so the row's leading fields (size, count,
      # dtype) are sometimes overwritten and "^ +<size> " matches nothing --
      # or matches an INFO line, yielding busbw=INFO.
      #
      # A complete nccl-tests row has 13 fields (in-place and out-of-place
      # blocks, each time/algbw/busbw/#wrong, after size/count/type/redop/
      # root): time=$6, busbw=$8. When stderr has eaten the leading fields
      # only the trailing 8-number block survives: time=$1, busbw=$3.
      # Verified against both shapes from jobs 7012/7015.
      read -r US BW < <(awk '
        /NCCL INFO|NET\/OFI/ { next }
        {
          n = 0
          for (i = 1; i <= NF; i++) if ($i ~ /^-?[0-9]+(\.[0-9]+)?$/) n++
          if (NF == 13 && $1 ~ /^[0-9]+$/ && $6 ~ /^[0-9.]+$/) { print $6, $8; exit }
          if (NF == 8 && n == 8 && $4 ~ /^[0-9]+$/ && $8 ~ /^[0-9]+$/) { print $1, $3; exit }
        }' "$F")

      if [ "$RC" = 0 ] && [ -n "${US:-}" ]; then
        row "$ARM" allreduce "$B" "$RANKS" "$PPN" "$US" "$BW" ok "busbw_net_$TR"
      else
        row "$ARM" allreduce "$B" "$RANKS" "$PPN" - - failed "exit_${RC}_net_$TR"
      fi
    done

    # ---------- layer 4: torch.distributed allreduce ----------
    F=$RUN/ladder_torch_r${RANKS}_${B}.txt
    timeout 300 $MPIEXEC -n "$RANKS" -ppn "$PPN" \
        "$W/pals_torch_env.sh" "$PY" "$W/torch_dist_bench.py" \
        --collective allreduce --dtype float32 \
        --min-bytes "$B" --max-bytes "$B" --iters "$ITERS" \
        > "$F" 2>&1
    RC=$?
    # torch_dist_bench columns: bytes iters ms/iter algbw_GB/s busbw_GB/s
    # NOT bytes,usec,... -- $2 is the iteration count. Convert ms -> us.
    LINE=$(grep -aE "^ *$B " "$F" | head -1)
    MS=$(echo "$LINE" | awk '{print $3}')
    BW=$(echo "$LINE" | awk '{print $5}')
    US=$(awk -v m="$MS" 'BEGIN{ if (m=="") print ""; else printf "%.1f", m*1000 }')
    if [ "$RC" = 0 ] && [ -n "$US" ]; then
      row torch allreduce "$B" "$RANKS" "$PPN" "$US" "$BW" ok busbw
    else
      row torch allreduce "$B" "$RANKS" "$PPN" - - failed "exit_$RC"
    fi

    # ---------- layer 5: NIXL READ (pairwise, cross-node only) ----------
    # NIXL over LIBFABRIC/CXI has no same-node loopback: fi_domain() returns
    # -38 and createBackend raises NIXL_ERR_BACKEND. That is a known
    # transport limit, already recorded for this layer -- not a ladder
    # failure, so it is marked n/a with the reason rather than 'failed'.
    if [ "$RANKS" = "2" ] && [ "$NNODES" = "2" ]; then
      SD=$RUN/sync_nixl_r2_${B}; mkdir -p "$SD"
      F=$RUN/ladder_nixl_r${RANKS}_${B}.txt
      EXPECT=cross-node
      timeout 600 $MPIEXEC -n 2 -ppn "$PPN" \
          "$W/pals_nixl_env.sh" "$PY" "$W/nixl_putget_bench.py" \
          --op READ --mem VRAM --sizes "$B" --iters "$ITERS" \
          --sync-dir "$SD" --expect "$EXPECT" \
          --json-out "$RUN/ladder_nixl_r${RANKS}_${B}.json" > "$F" 2>&1
      RC=$?
      # Parse the JSON, not stdout: NIXL interleaves libfabric NUMA warnings
      # with results, and its text line is "best ... | mean ...", which no
      # single awk column maps to. best_s is the like-for-like number.
      read -r US BW < <("$PY" - "$RUN/ladder_nixl_r${RANKS}_${B}.json" "$B" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    want = int(sys.argv[2])
    for r in d.get("rows", []):
        if int(r.get("bytes", -1)) == want:
            s = r.get("best_s") or r.get("transfer_best_s")
            if s:
                print(f"{s*1e6:.1f} {want/s/1e9:.3f}")
                break
    else:
        print(" ")
except Exception:
    print(" ")
PYEOF
)
      if [ "$RC" = 0 ] && [ -n "${US// /}" ]; then
        row nixl read "$B" "$RANKS" "$PPN" "$US" "$BW" ok best_of_iters
      else
        row nixl read "$B" "$RANKS" "$PPN" - - failed "exit_$RC"
      fi
    elif [ "$RANKS" = "2" ]; then
      row nixl read "$B" "$RANKS" "$PPN" - - n/a no_cxi_loopback_same_node
    else
      row nixl read "$B" "$RANKS" "$PPN" - - n/a pairwise_2_rank_api_only
    fi

    # ---------- layer 1: fi_pingpong (NIC ceiling reference) ----------
    # fi_pingpong needs an asymmetric server/client launch (rank 0 listens,
    # rank 1 connects), which the sweep already does per-rail. It is a
    # 2-process host-memory NIC test with no collective and no GPU, so it
    # does not belong on a GPU rank ladder at all. It is carried here as a
    # fixed CEILING reference, read from the sweep's own measurement, and
    # never presented as a ladder point.
    row fi pingpong "$B" "$RANKS" "$PPN" - - n/a see_layer1_per_rail_ceiling

  done
done

echo ""
echo "=== ladder complete: $CSV ==="
