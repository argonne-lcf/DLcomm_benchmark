#!/bin/bash
#PBS -N ncclbuild
#PBS -A tara_deployment
#PBS -q workq
#PBS -l select=1:ngpus=4
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -o /home/kaushikvelusamy/Dynamo_Slingshot_tara/logs/

# Instrumented NCCL toolchain build.
#
# Three artifacts, each closing a specific gap in the current measurements:
#
#   1. libnccl.so built with TRACE=1  -> NCCL_DEBUG=TRACE timing records,
#      so init/bootstrap/algorithm selection is visible instead of inferred.
#   2. libnccl-profiler.so (ext-profiler) -> per-event TIMERS. This is the
#      one that actually answers "where does the time go": KernelCh events
#      are GPU time, ProxyStep events are network time, so compute and wire
#      are separated by measurement rather than by argument.
#   3. libnccl-net.so (aws-ofi-nccl) -> binds NCCL to libfabric/CXI, which
#      is the fix for the TCP fallback that makes every 2-node number a
#      TCP number today.
#
# Built in a PBS job, not on the login node: vendored bootstraps fork past
# the login-node process cap (errno 11 vfork).
set -uo pipefail

W=/home/kaushikvelusamy/Dynamo_Slingshot_tara
SRC=$W/nccl_instr_src
PREFIX=$W/nccl_instr
LOGD=$W/logs/ncclbuild_${PBS_JOBID:-manual}
mkdir -p "$PREFIX" "$LOGD"

module load cray-mpich/9.1.0 2>/dev/null
module unload darshan 2>/dev/null

# Pin a real host compiler. Under PrgEnv-nvidia the vendored C++ is compiled
# by NVHPC and fails in ways that name nothing useful.
export CC=$(command -v gcc)
export CXX=$(command -v g++)
CUDA_HOME=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/12.9}
export CUDA_HOME

echo "=============================================="
echo " host:      $(hostname)"
echo " ulimit -u: $(ulimit -u)"
echo " CC:        $($CC --version | head -1)"
echo " nvcc:      $($CUDA_HOME/bin/nvcc --version | tail -2 | head -1)"
echo " prefix:    $PREFIX"
echo "=============================================="

# ---------------------------------------------------------------
# 1. NCCL with TRACE
# ---------------------------------------------------------------
# GH200 is sm_90. Building only that target keeps the compile to minutes
# instead of an hour across every architecture.
echo ""
echo "########## 1/3  libnccl.so (TRACE=1, sm_90) ##########"
cd "$SRC/nccl" || exit 1
make -j8 src.build \
     CUDA_HOME="$CUDA_HOME" \
     NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" \
     TRACE=1 VERBOSE=1 \
     > "$LOGD/build_nccl.log" 2>&1
NCCL_RC=$?
echo "nccl build exit: $NCCL_RC"
if [ "$NCCL_RC" -ne 0 ]; then
  echo "--- first real errors ---"
  grep -nE 'error:|fatal error|Error 1' "$LOGD/build_nccl.log" | head -15
  exit 1
fi
cp -a "$SRC/nccl/build/lib/." "$PREFIX/lib/" 2>/dev/null || {
  mkdir -p "$PREFIX/lib"; cp -a "$SRC/nccl/build/lib/." "$PREFIX/lib/"; }
mkdir -p "$PREFIX/include"
cp -a "$SRC/nccl/build/include/." "$PREFIX/include/"
echo "installed: $(ls -la $PREFIX/lib/libnccl.so.2* 2>/dev/null | head -2)"

# ---------------------------------------------------------------
# 2. profiler plugin (the timers)
# ---------------------------------------------------------------
echo ""
echo "########## 2/3  libnccl-profiler.so (ext-profiler) ##########"
cd "$SRC/nccl/ext-profiler/example" || exit 1
make -j4 NCCL_HOME="$PREFIX" > "$LOGD/build_profiler.log" 2>&1
PROF_RC=$?
echo "profiler build exit: $PROF_RC"
if [ "$PROF_RC" -ne 0 ]; then
  grep -nE 'error:|fatal error|Error 1' "$LOGD/build_profiler.log" | head -15
else
  find . -name 'libnccl-profiler*.so' -exec cp -v {} "$PREFIX/lib/" \;
fi

# ---------------------------------------------------------------
# 3. aws-ofi-nccl against the CXI libfabric
# ---------------------------------------------------------------
echo ""
echo "########## 3/3  libnccl-net.so (aws-ofi-nccl -> CXI) ##########"
cd "$SRC/aws-ofi-nccl" || exit 1
if [ ! -f configure ]; then
  ./autogen.sh > "$LOGD/autogen.log" 2>&1
  echo "autogen exit: $?"
fi
./configure \
    --prefix="$PREFIX" \
    --with-libfabric="$W/shs-libfabric-install" \
    --with-cuda="$CUDA_HOME" \
    --with-nccl="$PREFIX" \
    --disable-tests \
    > "$LOGD/configure_ofi.log" 2>&1
CFG_RC=$?
echo "configure exit: $CFG_RC"
if [ "$CFG_RC" -ne 0 ]; then
  echo "--- configure failure ---"
  grep -nE 'error:|cannot|not found' "$LOGD/configure_ofi.log" | tail -15
else
  make -j8 > "$LOGD/build_ofi.log" 2>&1
  OFI_RC=$?
  echo "make exit: $OFI_RC"
  if [ "$OFI_RC" -eq 0 ]; then
    make install >> "$LOGD/build_ofi.log" 2>&1
    echo "install exit: $?"
  else
    grep -nE 'error:|fatal error|Error 1' "$LOGD/build_ofi.log" | head -15
  fi
fi

# ---------------------------------------------------------------
# verification -- an install proves nothing on its own
# ---------------------------------------------------------------
echo ""
echo "########## verification ##########"
echo "--- produced libraries ---"
ls -la "$PREFIX/lib/" 2>/dev/null | grep -E '\.so' | head -12
echo ""
echo "--- libnccl.so version symbol ---"
strings "$PREFIX/lib/libnccl.so.2" 2>/dev/null | grep -oE '^2\.27\.[0-9]+' | head -2
echo ""
echo "--- is the plugin the CXI one? (must show OUR libfabric) ---"
for so in "$PREFIX/lib/libnccl-net.so" "$PREFIX/lib/libnccl-net-ofi.so"; do
  [ -e "$so" ] && { echo "== $so"; ldd "$so" | grep -iE 'fabric|cuda' | head -4; }
done
echo ""
echo "--- profiler plugin ---"
ls -la "$PREFIX"/lib/libnccl-profiler*.so 2>/dev/null
echo ""
echo "BUILD_LOGS=$LOGD"
