#!/bin/bash
# Environment wrapper for the NIXL put/get benchmark under PALS.
#
# NIXL needs its own install tree on LD_LIBRARY_PATH (libnixl.so), its plugin
# directory, and the custom CXI-enabled libfabric this stack was built against.
# Without NIXL_PLUGIN_DIR the agent comes up with no backends and the
# LIBFABRIC backend request fails at agent construction.
W=${DLCOMM_WORKDIR:-$HOME/Dynamo_Slingshot_tara}
export NIXL_INSTALL_DIR=$W/nixl/nixl_install
export UCX_INSTALL=$W/ucx_install
export CUSTOM_LIBFABRIC=$W/shs-libfabric-install

CUDA_LIB=$(dirname "$(find /opt/nvidia -name 'libcudart.so.12' 2>/dev/null | head -1)")
export LD_LIBRARY_PATH="$CUSTOM_LIBFABRIC/lib:$NIXL_INSTALL_DIR/lib64:$NIXL_INSTALL_DIR/lib:$UCX_INSTALL/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$NIXL_INSTALL_DIR/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NIXL_PLUGIN_DIR="$NIXL_INSTALL_DIR/lib64/plugins"

# CXI provider selection and HMEM tuning. Without these the LIBFABRIC backend
# fails to construct and nixl_agent() raises NIXL_ERR_BACKEND -- the agent
# never reaches a transfer, so the failure looks like "NIXL is broken" when it
# is really an unconfigured provider. Values match the known-good job 6931.
export FI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_HMEM_CUDA_ENABLE=1
export FI_HMEM_CUDA_USE_GDRCOPY=0
export FI_CXI_ENABLE_DMABUF=1
export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1

# PALS supplies the Slingshot VNI. TMPDIR defaults to /var/run/palsd/<uuid>,
# which is not writable by the job and makes anything that opens a temp file
# fail with PermissionError [Errno 13]. Override it per job.
export TMPDIR="${TMPDIR_OVERRIDE:-/tmp/${PBS_JOBID:-nixl}}"
mkdir -p "$TMPDIR" 2>/dev/null

export RANK="${PALS_RANKID:-0}"
export LOCAL_RANK="${PALS_LOCAL_RANKID:-0}"

exec "$@"
