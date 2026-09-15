#!/bin/bash -l
#PBS -N dlcomm_mpiprobe
#PBS -l select=1:ncpus=208
#PBS -l walltime=00:10:00
#PBS -q debug-scaling
#PBS -l filesystems=home:flare
#PBS -A datascience
#PBS -j oe

set -u
cd /lus/flare/projects/datascience/kaushik/DLcomm
set +u; module load frameworks/2025.3.1 >/dev/null 2>&1; set -u

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
PY=$(command -v python)

echo "############ A: bare mpi4py, no CCL vars ############"
timeout 120 "$MPIEXEC" -n 4 -ppn 4 "$PY" -c '
from mpi4py import MPI
print("rank", MPI.COMM_WORLD.Get_rank(), "of", MPI.COMM_WORLD.Get_size(), flush=True)
' 2>&1 | head -8

echo "############ B: mpi4py + ALCF minimal set ############"
CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi FI_MR_CACHE_MONITOR=userfaultfd \
timeout 120 "$MPIEXEC" -n 4 -ppn 4 "$PY" -c '
from mpi4py import MPI
print("rank", MPI.COMM_WORLD.Get_rank(), "of", MPI.COMM_WORLD.Get_size(), flush=True)
' 2>&1 | head -8

echo "############ C: no mpi4py, torch xccl only, ALCF set ############"
CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi FI_MR_CACHE_MONITOR=userfaultfd \
MASTER_ADDR=$(hostname) MASTER_PORT=29711 \
timeout 200 "$MPIEXEC" -n 4 -ppn 4 ./pals_env.sh "$PY" -c '
import os, torch, torch.distributed as dist
torch.xpu.set_device(int(os.environ["RANK"]) % torch.xpu.device_count())
dist.init_process_group(backend="xccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
x = torch.ones(1024, device="xpu"); dist.all_reduce(x)
print("rank", dist.get_rank(), "allreduce ok, sum =", float(x[0]), flush=True)
dist.destroy_process_group()
' 2>&1 | grep -vE "^(I|W)[0-9]{8} |CCL_WARN|UserWarning|return func" | head -10

echo "############ D: which MPI does mpi4py link? ############"
"$PY" -c 'from mpi4py import MPI; print(MPI.Get_library_version()[:120])' 2>&1 | head -3
echo "PROBE_DONE"
