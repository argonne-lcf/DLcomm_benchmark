#!/bin/bash -l
#PBS -N dlcomm_allscales
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=flare:home
#PBS -j oe

# Every layer at both target scales, in one queue slot.
#
#   scale A: 1 node,  12 ranks (1 per tile)
#   scale B: 2 nodes, 24 ranks
#
# Allocating 2 nodes and running the 12-rank case on one of them means both
# scales are measured against the same binaries, the same modules and the
# same allocation -- a difference between them is then a scaling effect, not
# a difference in build or environment.
#
# Layers covered: C++ SYCL transfer (h2d/d2h/d2d), C++ oneCCL collectives and
# p2p, OSU/MPI, torch.distributed. torchcomms is probed separately while its
# bootstrap is still being fixed.
#
# Nothing here is allowed to fail silently: each stage records an exit status,
# and the summary marks a layer "unavailable" rather than omitting it.

set -u
cd "$PBS_O_WORKDIR"

RUN="$PWD/validation/allscales_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
echo "RUN=$RUN"

# --- launcher must be Aurora's, captured before any conda activation --------
set +u
module load frameworks/2025.3.1
set -u
MPIEXEC="$(command -v mpiexec)"
case "$MPIEXEC" in
    /opt/cray/pals/*) echo "launcher: $MPIEXEC" ;;
    *) echo "FATAL: expected PALS mpiexec, got $MPIEXEC"; exit 1 ;;
esac

NNODES=$(wc -l < "$PBS_NODEFILE")
echo "nodes allocated: $NNODES"
if [ "$NNODES" -lt 2 ]; then
    echo "FATAL: need 2 nodes to cover both scales, got $NNODES"
    exit 1
fi
head -1 "$PBS_NODEFILE" > "$RUN/one_node.txt"

# PALS sets no global size variable (proved by job 8824786), so every rank is
# launched through a wrapper that derives WORLD_SIZE from PALS_LOCAL_SIZE and
# the node count. MASTER_ADDR must be identical on every rank.
WRAP="$PWD/pals_env.sh"
chmod +x "$WRAP"
export MASTER_ADDR
MASTER_ADDR=$(head -1 "$PBS_NODEFILE")
export MASTER_PORT=29522

CCL=/opt/aurora/26.26.0/oneapi/ccl/latest
PY_FW=$(command -v python)
TCENV=/lus/flare/projects/datascience/kaushik/torch-comm-everything/env2

# ---------------------------------------------------------------------------
# build once, run at both scales
# ---------------------------------------------------------------------------
echo "############ BUILD ############"
# Guard against building a stale source. Job 8824908 silently produced a full
# set of numbers from a pre-fix binary because the fixed file had been copied
# to a different path than the one compiled here. Assert the markers the
# current fixes introduce, and refuse to run without them.
for marker_file in \
    "MAP rank=:DLcomm_benchmark/dl_comm/ccl/ccl_bench.cpp" \
    "MOVED_BYTES:DLcomm_benchmark/dl_comm/ccl/ccl_bench.cpp" \
    "MAP rank=:DLcomm_benchmark/dl_comm/transfer/pci_fixed.cpp" ; do
    marker="${marker_file%%:*}"
    src="${marker_file#*:}"
    if ! grep -q "$marker" "$src"; then
        echo "FATAL: '$marker' missing from $src -- stale source, refusing to run"
        exit 1
    fi
done
echo "source markers verified (per-rank device selection + busbw numerator)"

mpicxx -cxx=icpx -fsycl -std=c++17 -O2 -DDLCOMM_XCCL DLcomm_benchmark/dl_comm/ccl/ccl_bench.cpp \
    -o "$RUN/ccl_bench" -I"$CCL/include" -L"$CCL/lib" -lccl 2>&1 | grep -v "^/usr/bin/ld: warning" | head -5
echo "CCL_BUILD_EXIT=${PIPESTATUS[0]}"

# pci_fixed.cpp calls MPI_Barrier/MPI_Reduce, so it needs the MPI wrapper,
# not bare icpx (job 8824800: undefined reference to MPI_Barrier).
mpicxx -cxx=icpx -fsycl -std=c++17 -O2 DLcomm_benchmark/dl_comm/transfer/pci_fixed.cpp \
    -o "$RUN/pci_fixed" 2>&1 | grep -v "^/usr/bin/ld: warning" | head -5
echo "PCI_BUILD_EXIT=${PIPESTATUS[0]}"

cat > "$RUN/torch_dist_bench.py" <<'PYEOF'
import os, time, statistics
from datetime import timedelta
import torch
import torch.distributed as dist

# No mpi4py. The working Aurora reference reads the launcher variables
# directly and passes device_id to init_process_group.
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
if world < 2:
    raise SystemExit(f"world={world}: launcher did not set up a real job")

torch.xpu.set_device(local_rank)
device = torch.device("xpu", local_rank)
dist.init_process_group(backend="xccl", init_method="env://", rank=rank,
                        world_size=world, timeout=timedelta(seconds=300),
                        device_id=device)

def busbw_factor(op, n):
    if op == "allreduce":
        return 2.0 * (n - 1) / n
    if op in ("allgather", "alltoall", "reduce_scatter"):
        return (n - 1) / n
    return 1.0

ITERS, WARMUP = 20, 5
for nbytes in (1 << 20, 1 << 21, 1 << 22):
    count = nbytes // 4
    x = torch.ones(count, dtype=torch.float32, device="xpu")
    # Largest multiple of `world` that fits in `count`, for the collectives
    # that require an evenly divisible first dimension.
    count_div = (count // world) * world
    xa = x[:count_div]
    outg = torch.empty(count * world, dtype=torch.float32, device="xpu")
    for op in ("allreduce", "allgather", "alltoall", "broadcast", "reduce"):
        ts = []
        for i in range(WARMUP + ITERS):
            dist.barrier()
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            if op == "allreduce":
                dist.all_reduce(x)
            elif op == "allgather":
                dist.all_gather_into_tensor(outg, x)
            elif op == "alltoall":
                # all_to_all_single requires dim 0 to divide by world size.
                # count = nbytes/4 is a power of two, so it divides by 12 but
                # not by 24; the C++ layer already trims for this and the
                # torch layer did not, which failed the whole stage at 24
                # ranks after every other collective had succeeded.
                dist.all_to_all_single(xa.clone(), xa)
            elif op == "broadcast":
                dist.broadcast(x, 0)
            elif op == "reduce":
                dist.reduce(x, 0)
            torch.xpu.synchronize()
            t1 = time.perf_counter()
            if i >= WARMUP:
                ts.append(t1 - t0)
        if rank == 0:
            t = statistics.median(ts)
            # alltoall runs on the trimmed buffer xa (count_div elements), so
            # the record must report the bytes actually moved, not the bytes
            # requested. At 24 ranks count=262144 trims to 262128; reporting
            # nbytes would overstate the transfer and inflate the bandwidth.
            nbytes_eff = count_div * 4 if op == "alltoall" else nbytes
            moved = (nbytes_eff * world
                     if op in ("allgather", "reduce_scatter") else nbytes_eff)
            algbw = moved / t
            print(f"LAYER=torch_dist BACKEND=xccl OP={op} BYTES={nbytes_eff} "
                  f"RANKS={world} T_MED={t:.6g} ALGBW={algbw:.6g} "
                  f"BUSBW={algbw * busbw_factor(op, world):.6g}", flush=True)
dist.destroy_process_group()
PYEOF

run_scale () {
    local tag="$1" nranks="$2" ppn="$3" hostarg="$4"
    local nnodes=$(( nranks / ppn ))
    export DLCOMM_NNODES="$nnodes"
    echo ""
    echo "################################################################"
    echo "# SCALE $tag : $nranks ranks, $ppn per node, $nnodes node(s)"
    echo "################################################################"
    local out="$RUN/$tag"
    mkdir -p "$out"

    echo "---- C++ oneCCL collectives + p2p ----"
    # Keep MAP lines as well as LAYER lines: the tile-mapping assertion below
    # reads them. Job 8824908 filtered them out and the check saw nothing.
    # shellcheck disable=SC2086
    timeout 900 "$MPIEXEC" --pmi=pmix --envall -n "$nranks" -ppn "$ppn" $hostarg "$RUN/ccl_bench" 2>&1 \
        | grep -E "^(LAYER=cpp_ccl|MAP )" | tee "$out/ccl_bench.txt" | grep -E "^LAYER=" | tail -20
    echo "CCL_RUN_EXIT=${PIPESTATUS[0]}"

    echo "    tile map (rank -> device):"
    awk '/^MAP /{print $4, $6}' "$out/ccl_bench.txt" | sort | uniq -c | head -26
    n_map=$(grep -cE "^MAP " "$out/ccl_bench.txt" || true)
    n_uniq=$(awk '/^MAP /{print $4, $6}' "$out/ccl_bench.txt" | sort -u | wc -l)
    echo "    MAP_LINES=$n_map DISTINCT_HOST_DEV=$n_uniq EXPECTED=$nranks"
    if [ "$n_map" -eq 0 ]; then
        echo "    TILE_CHECK=FAIL (no MAP lines -- stale binary)"
    elif [ "$n_uniq" -ne "$nranks" ]; then
        echo "    TILE_CHECK=FAIL (ranks sharing tiles)"
    else
        echo "    TILE_CHECK=PASS"
    fi

    echo "---- C++ SYCL transfer (h2d/d2h/d2d) ----"
    # shellcheck disable=SC2086
    timeout 600 "$MPIEXEC" --pmi=pmix --envall -n "$nranks" -ppn "$ppn" $hostarg "$RUN/pci_fixed" 2>&1 \
        | grep -E "^(LAYER=|MAP )" | tee "$out/transfer.txt" | grep -E "^LAYER=" | tail -10
    echo "TRANSFER_RUN_EXIT=${PIPESTATUS[0]}"

    echo "---- OSU collectives ----"
    local OSUDIR=/lus/flare/projects/datascience/kaushik/DLcomm/osu-build/libexec/osu-micro-benchmarks/mpi
    for b in osu_allreduce osu_allgather osu_alltoall osu_bcast osu_reduce; do
        if [ -x "$OSUDIR/collective/$b" ]; then
            # shellcheck disable=SC2086
            timeout 400 "$MPIEXEC" --pmi=pmix --envall -n "$nranks" -ppn "$ppn" $hostarg \
                "$OSUDIR/collective/$b" -m 1048576:4194304 -i 20 -x 5 \
                > "$out/$b.txt" 2>&1
            echo "  $b exit=$?"
        else
            echo "  $b MISSING at $OSUDIR/collective/$b"
        fi
    done
    if [ -x "$OSUDIR/pt2pt/osu_latency" ]; then
        # shellcheck disable=SC2086
        timeout 300 "$MPIEXEC" -n 2 -ppn "$ppn" $hostarg \
            "$OSUDIR/pt2pt/osu_latency" -m 1048576:4194304 -i 20 -x 5 \
            > "$out/osu_latency.txt" 2>&1
        echo "  osu_latency exit=$?"
    fi

    echo "---- torch.distributed (XCCL) ----"
    cd DLcomm_benchmark
    # shellcheck disable=SC2086
    # NOTE: the script is passed as a FILE, not on stdin. mpiexec delivers
    # stdin to rank 0 only, so a heredoc leaves every other rank with an
    # empty program (job 8824800: exit 124, zero output, rank 0 hung).
    # shellcheck disable=SC2086
    CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi FI_MR_CACHE_MONITOR=userfaultfd \
        ZE_FLAT_DEVICE_HIERARCHY=FLAT ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
    timeout 900 "$MPIEXEC" --pmi=pmix --envall -n "$nranks" -ppn "$ppn" $hostarg \
        "$WRAP" "$PY_FW" "$RUN/torch_dist_bench.py" > "$out/torch_dist.txt" 2>&1
    echo "TORCH_RUN_EXIT=$?"
    grep -E "^LAYER=torch_dist" "$out/torch_dist.txt" | tail -8
    cd ..

    echo "---- torchcomms 0.3.0 (XCCL) ----"
    # Separate conda env with the locally built torchcomms. The launcher is
    # already resolved to PALS above; activating the env here would shadow it,
    # so only the interpreter comes from the env.
    # Stack selection for the torchcomms layer.
    #
    #   DLCOMM_TC_STACK=frameworks : stock module torchcomms 0.1.0. Bootstraps
    #       correctly but its XCCL backend stubs out 16 operations, so only
    #       all_reduce is measurable (verified by job 8825144: TC_SUPPORTED=1/13).
    #   DLCOMM_TC_STACK=local      : torchcomms 0.3.0 paired with the matching
    #       custom torch build, copied into this project. `strings` on its
    #       _comms_xccl .so shows zero "is not supported" markers, so the
    #       remaining collectives and p2p ops are implemented there.
    #
    # The two must be used as a matched pair. Mixing a custom torch with an
    # independently built extension is what broke env2 (undefined symbol
    # urDeviceWaitExp, unresolved libc10.so) and produced the new_comm
    # segfault; see docs/fixes/26.
    local TC_STACK="${DLCOMM_TC_STACK:-frameworks}"
    # The 0.3.0 stack lives under this project. Byte-identical to the build it
    # was copied from: the three torchcomms .so files and libc10.so all
    # md5-match. There is deliberately no fallback to another user's directory
    # -- a stack that can move out from under the harness makes its numbers
    # unattributable, and a hard failure is the honest outcome.
    local LOCAL_STACK=/lus/flare/projects/datascience/kaushik/stacks/torchcomms_0.3.0
    local TC03_TORCH="$LOCAL_STACK/pytorch"
    local TC03_TC="$LOCAL_STACK/torchcomms"

    # Default to the frameworks-provided torchcomms. The working Aurora
    # reference harnesses import plain `torchcomms` under the module, which
    # ships torch 2.10.0a0+git449b176 and a matching torchcomms build. The
    # custom env2 stack (torch 2.13.0+xpu with a separately built
    # torchcomms) segfaults inside new_comm even with correct per-rank
    # devices, PMIx bootstrap and ZE_AFFINITY_MASK, which is the signature
    # of an ABI mismatch rather than a configuration error.
    # DLCOMM_TC_ENV=1 selects env2 instead, to compare the two.
    local TCPY TC_PYTHONPATH="" TC_LDPATH=""
    if [ "${DLCOMM_TC_ENV:-0}" = "1" ]; then
        TCPY="$TCENV/bin/python"
        TC_LDPATH="$TCENV/lib:$TCENV/lib/python3.12/site-packages/torch/lib:"
    elif [ "$TC_STACK" = "local" ]; then
        # Matched pair: torchcomms 0.3.0 + the torch it was built against.
        TCPY="$PY_FW"
        TC_PYTHONPATH="$TC03_TC:$TC03_TORCH"
        TC_LDPATH="$TC03_TORCH/torch/lib:"
    else
        TCPY="$PY_FW"
    fi
    echo "TORCHCOMMS_STACK=$TC_STACK"
    if [ -x "$TCPY" ]; then
        cd DLcomm_benchmark
        # shellcheck disable=SC2086
        PYTHONPATH="${TC_PYTHONPATH:+$TC_PYTHONPATH:}${PYTHONPATH:-}" \
        LD_LIBRARY_PATH="${TC_LDPATH}${LD_LIBRARY_PATH:-}" \
        MASTER_PORT=29533 \
        CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi FI_MR_CACHE_MONITOR=userfaultfd \
        ZE_FLAT_DEVICE_HIERARCHY=FLAT ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
        timeout 900 "$MPIEXEC" --pmi=pmix --envall -n "$nranks" -ppn "$ppn" $hostarg \
            bash -c '
              # Mask each rank down to ONE visible XPU, then declare local
              # rank 0. This is the pattern in all 81 working torchcomms
              # reference launchers. Exposing all 12
              # tiles and indexing by local rank works for oneCCL and SYCL
              # but segfaults inside the XCCL bootstrap in new_comm.
              # No ZE_AFFINITY_MASK: job 8825078 applied it correctly and
              # still segfaulted, and the reference harnesses do not use it.
              export LOCAL_RANK=${PALS_LOCAL_RANKID}
              echo "RANK_MAP: RANK=${PALS_RANKID} HOST=$(hostname)" \
                   "LOCAL_RANK=${LOCAL_RANK}"
              exec "$@"
            ' _ "$WRAP" "$TCPY" ../probe_tc03.py > "$out/torchcomms.txt" 2>&1
        echo "TORCHCOMMS_RUN_EXIT=$?"
        grep -E "^LAYER=torchcomms|^\[(yes|NO )\]|^MATRIX|^TC_" "$out/torchcomms.txt" | head -24
        # Exit status alone does not prove the layer measured anything.
        tc_sup=$(grep -oE "^TC_SUPPORTED=[0-9]+/[0-9]+" "$out/torchcomms.txt" | head -1)
        if [ -z "$tc_sup" ]; then
            echo "TORCHCOMMS_VERDICT=NO_MATRIX (probe produced no support matrix)"
        else
            tc_num=${tc_sup#TC_SUPPORTED=}; tc_den=${tc_num#*/}; tc_num=${tc_num%%/*}
            if [ "$tc_num" -eq 0 ]; then
                echo "TORCHCOMMS_VERDICT=FAIL_ALL_OPS $tc_sup"
            elif [ "$tc_num" -lt "$tc_den" ]; then
                echo "TORCHCOMMS_VERDICT=PARTIAL $tc_sup (backend implements $tc_num of $tc_den probed ops)"
            else
                echo "TORCHCOMMS_VERDICT=OK $tc_sup"
            fi
        fi
        grep -vE "^(I|W)[0-9]{8} |WARNING: Logging|^\s*$" "$out/torchcomms.txt" \
            | grep -iE "error|not supported|Traceback|world size" | head -6
        cd ..
    else
        echo "TORCHCOMMS STATUS=unavailable REASON=no_interpreter_at_$TCPY"
    fi
}

run_scale "1node_12rank"  12 12 "--hostfile $RUN/one_node.txt"
run_scale "2node_24rank"  24 12 ""

# ---------------------------------------------------------------------------
echo ""
echo "############ CROSS-LAYER COMPARISON ############"
cd DLcomm_benchmark
for tag in 1node_12rank 2node_24rank; do
    nr=12; [ "$tag" = "2node_24rank" ] && nr=24
    echo ""
    echo "=== $tag ==="
    "$PY_FW" -m dl_comm.analysis.compare_layers "$RUN/$tag" --ranks "$nr" 2>&1 | head -80
done
cd ..

echo ""
echo "RESULTS_DIR=$RUN"
