#!/bin/bash
# Build NIXL with the LIBFABRIC plugin against Cray libfabric 2.3.1 on tara-n.
#
# Blockers fixed vs the Oct-26 attempt:
#  A) fi_mr_attr.device.rocr -- NIXL gated it on FI_VERSION(2,3); Cray libfabric
#     2.3.1 reports API 2.3 but has NO rocr member. Patched to a meson member
#     probe (tools/patch_nixl_cray.py). ROCr is AMD-only: dead code on this box.
#  B) nixl-meta wheel target dies on Python 3.12 (distutils removed). That
#     target is gated on `find_program('uv', required: false)`, so we build with
#     uv hidden from PATH -> meson warns and skips it. The C++ lib + plugin,
#     which is all we need, is unaffected.
set -uo pipefail
W=/home/kaushikvelusamy/nixl_cxi_20261029
mkdir -p "$W/logs"
exec > >(tee "$W/logs/build.log") 2>&1
echo "# nixl libfabric build start $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

export CC=/opt/cray/pe/gcc-native/14/bin/gcc
export CXX=/opt/cray/pe/gcc-native/14/bin/g++
C13=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
# Blocker C: do NOT pass -Dlibfabric_path. That option hardcodes PREFIX/lib,
#  but Cray ships libfabric.so in PREFIX/lib64 -> "ld: cannot find -lfabric".
#  Omitting it takes meson's pkg-config branch, which reports lib64 correctly.
FI=/opt/cray/libfabric/2.3.1

source /home/kaushikvelusamy/node_health_20261026/venv/bin/activate
export PKG_CONFIG_PATH="$FI/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"

# Blocker B: hide uv so the meta-wheel custom_target is never created.
UVDIR=$(dirname "$(command -v uv)")
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vxF "$UVDIR" | paste -sd:)
command -v uv && { echo "FATAL: uv still on PATH"; exit 1; }
echo "uv hidden OK"

cd "$W/src/nixl" || exit 1
rm -rf build
echo
echo "=== configure ==="
meson setup build \
  --prefix="$W/nixl_cxi" \
  -Denable_plugins=LIBFABRIC \
  -Dcudapath_inc="$C13/include" \
  -Dcudapath_lib="$C13/lib64" \
  -Dbuild_docs=false \
  -Dwith_trace=false \
  -Dbuild_tests=false \
  -Dbuild_examples=false \
  -Dnixl_cuda_arch_list=90 2>&1 | tail -25

[ -f build/build.ninja ] || { echo "FATAL: configure failed"; exit 1; }

echo
echo "=== probe results (the two blockers) ==="
grep -iE "rocr|UNSOLICITED|uv not found" build/meson-logs/meson-log.txt | tail -8

echo
echo "=== build ==="
ninja -C build
BUILD_RC=$?
echo "BUILD_RC=$BUILD_RC"
[ $BUILD_RC -eq 0 ] || { echo "FATAL: build failed"; exit 1; }

echo
echo "=== install ==="
ninja -C build install
echo "INSTALL_RC=$?"

echo
echo "=== resulting plugin libs ==="
find "$W/nixl_cxi" -name "*.so*" | head -20
LFSO=$(find "$W/nixl_cxi" -iname "*libfabric*.so*" | head -1)
if [ -n "$LFSO" ]; then
  echo "LIBFABRIC_PLUGIN_BUILT=$LFSO"
  echo "--- links ---"
  ldd "$LFSO" | grep -E "fabric|cuda|not found"
else
  echo "LIBFABRIC_PLUGIN_MISSING"
fi
echo "# nixl libfabric build end $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
