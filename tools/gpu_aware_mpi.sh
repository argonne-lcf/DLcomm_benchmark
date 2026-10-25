# shellcheck shell=bash
#
# GPU-aware MPI (Cray GTL) setup for Tara.
#
# ---------------------------------------------------------------------------
# THE PROBLEM
# ---------------------------------------------------------------------------
# Setting MPICH_GPU_SUPPORT_ENABLED=1 without the GTL library actually loaded
# aborts every MPI rank at init:
#
#     MPICH ERROR ... MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested,
#     but GTL library is not linked
#
# The repo worked around this by exporting MPICH_GPU_SUPPORT_ENABLED=0
# globally in the NCCL jobscripts. That stops the abort but silently disables
# GPU-aware MPI, so any MPI call on a device buffer either fails or is staged
# through host memory.
#
# ---------------------------------------------------------------------------
# WHY MERELY LOADING THE MODULE IS NOT ENOUGH  (measured on tara-n)
# ---------------------------------------------------------------------------
# craype-accel-nvidia90 sets the link variables:
#
#     PE_MPICH_GTL_DIR_nvidia90  = -L/opt/cray/pe/mpich/9.1.0/gtl/lib
#     PE_MPICH_GTL_LIBS_nvidia90 = -lmpi_gtl_cuda
#
# but those only matter when the *compiler wrapper* links your binary. Python
# processes (mpi4py, torch) are not linked by cc, so the GTL never enters the
# process and MPICH aborts. Putting the GTL directory on LD_LIBRARY_PATH is
# also not enough -- MPICH probes for an already-loaded symbol, so the library
# must be present at process start. Measured:
#
#     LD_LIBRARY_PATH only  -> "GTL library is not linked"  (abort, rc=255)
#     LD_PRELOAD the GTL    -> DEVICE_MPI_OK, result correct (rc=0)
#
# And the repo's current workaround is not merely slow, it is unsafe. With
# MPICH_GPU_SUPPORT_ENABLED=0 an MPI call on a device buffer does not fall
# back to staging -- it dies (measured, 2 nodes, PBS job 8574):
#
#     CELL A  =1, no GTL preload  -> rc=255  GTL_ABORT
#     CELL B  =1, GTL preloaded   -> rc=0    DEVICE_MPI_OK val=2.0
#     CELL C  =0, device buffer   -> rc=139  cxil_map: write error (SIGSEGV)
#
# Cell C is what every committed NCCL jobscript does today.
#
# ---------------------------------------------------------------------------
# THE CUDA 12 / 13 TRAP
# ---------------------------------------------------------------------------
# Tara's system GTL is built against CUDA 13:
#
#     objdump -p libmpi_gtl_cuda.so | grep NEEDED  ->  libcudart.so.13
#
# while the project's torch wheels are cu126 and ship libcudart.so.12. So
# linking a C probe with -lmpi_gtl_cuda under cuda/12.9 fails:
#
#     undefined reference to `__cudaRegisterFatBinary@libcudart.so.13'
#
# The two runtimes DO coexist in one process -- verified via /proc/self/maps:
#
#     gtl_mapped=True  cudart13_mapped=True  cudart12_mapped=True  result=1.0
#
# torch keeps using cudart 12, the GTL uses cudart 13. So the fix is to make
# BOTH visible, not to force everything onto one CUDA version.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#     source tools/gpu_aware_mpi.sh
#     dlcomm_enable_gpu_aware_mpi   # sets LD_PRELOAD/LD_LIBRARY_PATH, exports =1
#  or dlcomm_disable_gpu_aware_mpi  # exports =0, for NCCL-only steps
#
# Then pass the environment through to the ranks explicitly:
#
#     mpiexec ... --env MPICH_GPU_SUPPORT_ENABLED="$MPICH_GPU_SUPPORT_ENABLED" \
#                 --env LD_PRELOAD="${LD_PRELOAD:-}" \
#                 --env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ...
#
# NEVER `export MPICH_GPU_SUPPORT_ENABLED=1` at the top of a jobscript that
# also runs a non-GTL step: the abort is per-process and unconditional.

# Locate the Cray GTL shared object. Prefers the versioned mpich directory
# that matches the loaded cray-mpich, falls back to `default`.
dlcomm_find_gtl() {
    local cand
    for cand in \
        ${PE_MPICH_GTL_DIR_nvidia90#-L} \
        ${CRAY_MPICH_DIR:+$CRAY_MPICH_DIR/../gtl/lib} \
        /opt/cray/pe/mpich/*/gtl/lib \
        /opt/cray/pe/mpich/default/gtl/lib
    do
        [ -n "$cand" ] || continue
        if [ -f "$cand/libmpi_gtl_cuda.so" ]; then
            echo "$cand/libmpi_gtl_cuda.so"
            return 0
        fi
    done
    return 1
}

# Find the libcudart major version the GTL was built against (e.g. "13"),
# then the directory that actually provides it.
dlcomm_find_gtl_cudart_dir() {
    local gtl="$1" major dir
    major=$(objdump -p "$gtl" 2>/dev/null \
            | awk '/NEEDED.*libcudart/ {print $2}' \
            | sed -n 's/.*libcudart\.so\.\([0-9]\+\).*/\1/p' | head -1)
    [ -n "$major" ] || return 1
    for dir in \
        "${CUDA_HOME:-}/targets/sbsa-linux/lib" \
        "${CUDA_HOME:-}/lib64" \
        /opt/nvidia/hpc_sdk/Linux_"$(uname -m)"/*/cuda/*/targets/*/lib
    do
        [ -n "$dir" ] || continue
        if [ -f "$dir/libcudart.so.$major" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

# Turn GPU-aware MPI ON for the steps that need it.
# Returns non-zero (and leaves MPICH_GPU_SUPPORT_ENABLED=0) if the GTL cannot
# be found, so a caller never requests GPU support it cannot satisfy.
dlcomm_enable_gpu_aware_mpi() {
    local gtl cudart_dir
    if ! gtl=$(dlcomm_find_gtl); then
        echo "WARNING: no libmpi_gtl_cuda.so found; GPU-aware MPI stays off" >&2
        export MPICH_GPU_SUPPORT_ENABLED=0
        return 1
    fi

    # The GTL's own CUDA runtime must be resolvable. It is frequently a
    # different major version than the one torch ships; both can be mapped.
    if cudart_dir=$(dlcomm_find_gtl_cudart_dir "$gtl"); then
        export LD_LIBRARY_PATH="$cudart_dir:${LD_LIBRARY_PATH:-}"
    else
        echo "WARNING: cudart for $(basename "$gtl") not found; may fail" >&2
    fi

    export LD_LIBRARY_PATH="$(dirname "$gtl"):${LD_LIBRARY_PATH:-}"
    case ":${LD_PRELOAD:-}:" in
        *":$gtl:"*) ;;                                   # already preloaded
        *) export LD_PRELOAD="$gtl${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
    export MPICH_GPU_SUPPORT_ENABLED=1
    echo "GPU-aware MPI enabled via $gtl"
    return 0
}

# Turn GPU-aware MPI OFF for NCCL-only / host-buffer steps, and drop the GTL
# preload so the step does not carry a library it does not need.
dlcomm_disable_gpu_aware_mpi() {
    local gtl
    if gtl=$(dlcomm_find_gtl) && [ -n "${LD_PRELOAD:-}" ]; then
        LD_PRELOAD=$(echo "$LD_PRELOAD" | tr ':' '\n' | grep -v "^${gtl}$" \
                     | paste -sd: -)
        export LD_PRELOAD="${LD_PRELOAD}"
        [ -z "$LD_PRELOAD" ] && unset LD_PRELOAD
    fi
    export MPICH_GPU_SUPPORT_ENABLED=0
}

# Emit the --env flags needed to pass this configuration to mpiexec ranks.
# PALS does not forward LD_PRELOAD by default.
dlcomm_mpi_gpu_env_flags() {
    printf -- '--env MPICH_GPU_SUPPORT_ENABLED=%s ' "${MPICH_GPU_SUPPORT_ENABLED:-0}"
    printf -- '--env LD_LIBRARY_PATH=%s ' "${LD_LIBRARY_PATH:-}"
    [ -n "${LD_PRELOAD:-}" ] && printf -- '--env LD_PRELOAD=%s ' "$LD_PRELOAD"
}
