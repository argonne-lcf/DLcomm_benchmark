#!/bin/bash
# Install dlcomm on Polaris without the three problems that plain
# `pip install dlcomm` hits:
#   1. PyPI torch is a CUDA 13 build; the working Polaris env uses cu128.
#      -> install torch from the PyTorch cu128 index first.
#   2. PyPI mpi4py ships a manylinux wheel with its own bundled MPICH, not
#      Cray MPICH, so it cannot talk to the PALS launcher / Slingshot.
#      -> build mpi4py from source against Cray MPICH (PrgEnv-gnu, cc).
#   3. Install onto loaded Lustre is extremely slow (byte-compiling torch).
#      -> --no-compile; use a single pip transaction per step.
# Usage: install_dlcomm_polaris.sh <venv_dir> [pypi|git]
set -euo pipefail
VENV=${1:?venv dir}
SRC=${2:-pypi}
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
export OMP_NUM_THREADS=1 PIP_DISABLE_PIP_VERSION_CHECK=1
#   4. `conda create` on /grand hung >4 min on overloaded polaris-login-02;
#      `python -m venv` from miniforge takes ~49 s on login-01.
#      -> plain venv, no conda solve. System /usr/bin/python3.11 is NOT usable:
#      it has no Python.h, so mpi4py cannot be built against it.
#   6. miniforge base python is 3.14; hydra-core 1.3.x crashes on 3.14
#      argparse ("ValueError: badly formed help string" on every dlcomm run).
#      -> use the Sept-2026 env's Python 3.12 (has headers) as the venv base.
# Default is the ALCF project env used to validate this script; override with PYBASE.
PYBASE=${PYBASE:-/lus/grand/projects/datascience/kaushikv/DLcomm-Test-Sept2026/env/dlcomm/bin/python3.12}
[ -x "$PYBASE" ] || { echo "FATAL: PYBASE=$PYBASE not found; set PYBASE to a Python 3.9-3.13 with headers"; exit 2; }
t0=$(date +%s); step(){ echo "[$(( $(date +%s)-t0 ))s] $*"; }
INC=$("$PYBASE" -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
[ -f "$INC/Python.h" ] || { echo "FATAL: $PYBASE has no Python.h ($INC); mpi4py cannot build"; exit 2; }

step "create venv ($PYBASE)"
[ -x "$VENV/bin/python" ] || "$PYBASE" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
PIP="$VENV/bin/pip"

step "torch (cu128)"
$PIP install -q --no-compile --index-url https://download.pytorch.org/whl/cu128 torch numpy

step "mpi4py from source against Cray MPICH"
module swap PrgEnv-nvidia PrgEnv-gnu >/dev/null 2>&1 || true
#   5. The default darshan module injects libdarshan into the cc link line,
#      and libdarshan NEEDs libmpi_gnu_123.so.12, which the upgraded Cray PE
#      no longer ships -> `ImportError: libmpi_gnu_123.so.12` at import time.
module unload darshan >/dev/null 2>&1 || true
MPICC=cc $PIP install -q --no-compile --no-binary=mpi4py --no-cache-dir --force-reinstall --no-deps mpi4py
SO=$(ls "$VENV"/lib/python3*/site-packages/mpi4py/MPI*.so)
if ldd "$SO" | grep -q "not found"; then ldd "$SO" | grep "not found"; echo "FATAL: mpi4py has unresolved libs"; exit 3; fi

step "dlcomm ($SRC)"
if [ "$SRC" = git ]; then
  $PIP install -q --no-compile "git+https://github.com/argonne-lcf/DLcomm_benchmark.git"
else
  $PIP install -q --no-compile dlcomm
fi
step "verify"
"$VENV/bin/python" - <<'EOF'
import torch, mpi4py, dl_comm
from mpi4py import MPI
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("mpi4py", mpi4py.__version__, "|", MPI.Get_library_version().splitlines()[0][:70])
print("dl_comm", dl_comm.__file__)
EOF
# `dlcomm --help` always fails (hydra needs --config-path/--config-name), so
# check that a shipped example config loads instead. Needs a git clone for
# the example YAMLs: set DLCOMM_EXAMPLE_YAML to one of them.
if [ -n "${DLCOMM_EXAMPLE_YAML:-}" ]; then
  (cd /tmp && "$VENV/bin/dlcomm" --config-path="$(dirname "$DLCOMM_EXAMPLE_YAML")" \
     --config-name="$(basename "$DLCOMM_EXAMPLE_YAML" .yaml)" --cfg job | head -5) \
     || { echo "FATAL: dlcomm could not load $DLCOMM_EXAMPLE_YAML"; exit 4; }
fi
step "done"
