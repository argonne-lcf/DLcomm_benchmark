#!/bin/bash
#PBS -N ncclab
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=2:ngpus=4
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# A/B: stock NCCL (TCP fallback) vs instrumented NCCL + aws-ofi-nccl (CXI).
# Both arms run in ONE job on the SAME nodes, back to back. Split across two
# jobs the comparison is confounded by node assignment.
#
# Arm A: stock  -> the 2-node numbers reported so far
# Arm B: instrumented libnccl.so + libnccl-net.so -> should select CXI
#
# The primary result is not a speedup, it is the transport line NCCL prints.
# A faster number with "Using network Socket" would mean the plugin did not
# load and the delta came from something else.
set -uo pipefail

W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
INSTR=$W/nccl_instr
FI=$W/shs-libfabric-install
RUN=$W/ncclab_${PBS_JOBID%%.*}
mkdir -p "$RUN"
cd "$RUN" || exit 1

MPIEXEC=$(ls /opt/cray/pals/*/bin/mpiexec 2>/dev/null | head -1)
[ -x "$MPIEXEC" ] || MPIEXEC=$(ls /opt/cray/pe/pals/*/bin/mpiexec 2>/dev/null | head -1)
NB=$W/bench_src/nccl-tests/build/all_reduce_perf

# all_reduce_perf needs libcudart.so.13 (module cuda/13.0) and a libnccl.so.2.
# The existing sweep gets the stock NCCL from the venv's pip package; that is
# arm A's library. Load the same modules the working sweep loads, or the
# binary dies at startup with "not found" and the arm looks like a fabric
# failure instead of a missing module.
module load cray-mpich/9.1.0 2>/dev/null
module load cuda/13.0 2>/dev/null
STOCK_NCCL=$W/venv/lib/python3.12/site-packages/nvidia/nccl/lib

# Passing --env LD_LIBRARY_PATH to mpiexec REPLACES the value the modules set,
# it does not extend it. Job 7011 failed exactly here: every rank died with
# "libcudart.so.12: cannot open shared object file" (exit 127), which my guard
# caught rather than reporting 143 as a fabric result.
#
# Two different CUDA runtimes are genuinely needed in one process:
#   .so.13 -> all_reduce_perf itself (built against CUDA 13)
#   .so.12 -> our CXI libfabric and the aws-ofi-nccl plugin
# Both directories must be on the path explicitly.
CUDA13_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/REDIST/cuda/13.0/targets/sbsa-linux/lib
CUDA12_LIB=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9/targets/sbsa-linux/lib
CUDA_LIBS="$CUDA13_LIB:$CUDA12_LIB"
for d in "$CUDA13_LIB" "$CUDA12_LIB"; do
  [ -d "$d" ] || echo "WARN: missing CUDA lib dir $d"
done

# Rank->GPU wrapper: PALS exports none of the local-rank vars nccl-tests reads.
NCCLENV=$W/pals_nccl_env.sh

# rank 0 must be the FIRST line of the nodefile; PBS does not sort it.
HOSTS=$(awk '!seen[$0]++' "$PBS_NODEFILE" | tr '\n' ',' | sed 's/,$//')
NN=$(awk '!seen[$0]++' "$PBS_NODEFILE" | wc -l)

echo "nodes=$NN hosts=$HOSTS"
echo "mpiexec=$MPIEXEC"
echo "nccl-tests=$NB"
ls -la "$NB" 2>/dev/null || { echo "FATAL: no all_reduce_perf"; exit 1; }

SIZE=268435456   # 256 MiB, the size with the worst measured cliff

run_arm () {
  local TAG=$1 LDP=$2 PLUGDIR=$3
  local OUT=$RUN/${TAG}.txt
  echo "" ; echo "########## arm=$TAG ##########"

  # NCCL_DEBUG=INFO makes NCCL name its transport. Without this the run
  # cannot be interpreted, only timed.
  env_args=(
    --env LD_LIBRARY_PATH="$LDP"
    --env NCCL_DEBUG=INFO
    --env NCCL_DEBUG_SUBSYS=INIT,NET,ENV
  )
  # Arm B additionally points NCCL at the plugin directory.
  if [ -n "$PLUGDIR" ]; then
    env_args+=( --env NCCL_NET_PLUGIN="$PLUGDIR/libnccl-net.so" )
    # profiler plugin = the per-event timers
    env_args+=( --env NCCL_PROFILER_PLUGIN="$PLUGDIR/libnccl-profiler-example.so" )
  fi

  timeout 600 "$MPIEXEC" -n 8 -ppn 4 --hostfile "$PBS_NODEFILE" \
      "${env_args[@]}" \
      "$NCCLENV" "$NB" -b $SIZE -e $SIZE -f 1 -g 1 -c 1 \
      > "$OUT" 2>&1
  local RC=$?
  echo "exit=$RC"

  # A startup failure must not be read as a slow fabric.
  if grep -qaE 'not found|error while loading shared libraries' "$OUT"; then
    echo "!! library resolution failed in this arm:"
    grep -ahoE '[^ ]*\.so[^ ]* => not found|error while loading[^\n]*' "$OUT" | sort -u | head -4
  fi

  echo "--- transport chosen ---"
  grep -ahoE 'Using network [A-Za-z_/]+' "$OUT" | sort -u | head -3
  grep -ahoE 'NET/Plugin: (Loaded|Could not find)[^,]*' "$OUT" | sort -u | head -3
  grep -ahoE 'NET/OFI [^,]*' "$OUT" | sort -u | head -4

  echo "--- measured ---"
  grep -aE "^ *$SIZE" "$OUT" | head -2
  echo "--- busbw (col 8 in-place) ---"
  grep -aE "^ *$SIZE" "$OUT" | awk '{print "time_us="$6"  algbw="$7"  busbw="$8}' | head -1
}

# ---- arm A: stock ----
# Stock NCCL from the venv, no plugin var -> reproduces current behaviour.
run_arm stock "$STOCK_NCCL:$CUDA_LIBS:$FI/lib:${LD_LIBRARY_PATH:-}" ""

# ---- arm B: instrumented + CXI plugin ----
# INSTR/lib FIRST so the instrumented libnccl.so wins over the venv copy.
run_arm instr "$INSTR/lib:$CUDA_LIBS:$FI/lib:${LD_LIBRARY_PATH:-}" "$INSTR/lib"

echo ""
echo "########## which libnccl did each arm load ##########"
for t in stock instr; do
  echo "== $t"
  grep -ahoE 'NCCL version [0-9.]+[^ ]*' "$RUN/$t.txt" | sort -u | head -2
done
echo ""
echo "RUN=$RUN"
