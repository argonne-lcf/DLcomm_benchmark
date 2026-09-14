#!/bin/bash -l
#PBS -N dlcomm_mpiprobe2
#PBS -l select=1:ncpus=208
#PBS -l walltime=00:15:00
#PBS -q debug-scaling
#PBS -l filesystems=home:flare
#PBS -A datascience
#PBS -j oe

set -u
cd /lus/flare/projects/datascience/kaushik/DLcomm
set +u; module load frameworks/2025.3.1 >/dev/null 2>&1; set -u

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
PY=$(command -v python)

# The previous probe's case C failed on my own missing export, not on torch.
# The guard in pals_env.sh fired correctly; supply the value this time.
export DLCOMM_NNODES=1
export MASTER_ADDR=$(hostname)
NR=12

ALCF="CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi FI_MR_CACHE_MONITOR=userfaultfd"

# Filter noise, but never truncate before the result line: case G of probe
# 8825011 printed seven glog warnings, which consumed a head -8 budget and
# discarded the CASE_G_OK/traceback line entirely, making a failed case
# look silent. Drop glog banners explicitly and keep a generous tail.
filt () { grep -vE "^(I|W)[0-9]{8} |CCL_WARN|UserWarning|return func|WARNING: Logging before InitGoogleLogging|^\s*$" | head -40; }

echo "############ C: torch xccl, NO mpi4py, ALCF set, ${NR} ranks ############"
env $ALCF MASTER_PORT=29721 \
timeout 240 "$MPIEXEC" -n $NR -ppn $NR ./pals_env.sh "$PY" -c '
import os, torch, torch.distributed as dist
r, w = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.xpu.set_device(r % torch.xpu.device_count())
dist.init_process_group(backend="xccl", rank=r, world_size=w)
x = torch.ones(1024, device="xpu"); dist.all_reduce(x)
if r == 0: print("CASE_C_OK sum =", float(x[0]), flush=True)
dist.destroy_process_group()
' 2>&1 | filt

echo "############ E: mpi4py + torch xccl, ALCF set (exact real-job combo) ############"
env $ALCF MASTER_PORT=29722 \
timeout 240 "$MPIEXEC" -n $NR -ppn $NR ./pals_env.sh "$PY" -c '
from mpi4py import MPI
import os, torch, torch.distributed as dist
r, w = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
torch.xpu.set_device(r % torch.xpu.device_count())
dist.init_process_group(backend="xccl", rank=r, world_size=w)
x = torch.ones(1024, device="xpu"); dist.all_reduce(x)
if r == 0: print("CASE_E_OK sum =", float(x[0]), flush=True)
dist.destroy_process_group()
' 2>&1 | filt

echo "############ F: mpi4py + torch xccl, NO CCL_ATL_TRANSPORT override ############"
env CCL_PROCESS_LAUNCHER=pmix MASTER_PORT=29723 \
timeout 240 "$MPIEXEC" -n $NR -ppn $NR ./pals_env.sh "$PY" -c '
from mpi4py import MPI
import os, torch, torch.distributed as dist
r, w = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
torch.xpu.set_device(r % torch.xpu.device_count())
dist.init_process_group(backend="xccl", rank=r, world_size=w)
x = torch.ones(1024, device="xpu"); dist.all_reduce(x)
if r == 0: print("CASE_F_OK sum =", float(x[0]), flush=True)
dist.destroy_process_group()
' 2>&1 | filt

echo "############ G: torchcomms env2, mpi4py + new_comm, ALCF set ############"
TCENV=/lus/flare/projects/datascience/kaushik/torch-comm-everything/env2
env $ALCF MASTER_PORT=29724 \
    LD_LIBRARY_PATH="$TCENV/lib:$TCENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}" \
timeout 240 "$MPIEXEC" -n $NR -ppn $NR ./pals_env.sh "$TCENV/bin/python" -c '
from mpi4py import MPI
import os, torch, torchcomms as tc
r = MPI.COMM_WORLD.Get_rank()
d = torch.device(f"xpu:{r % torch.xpu.device_count()}")
torch.xpu.set_device(d)
comm = tc.new_comm("xccl", d, name="probe")
if r == 0: print("CASE_G_OK new_comm returned", flush=True)
' 2>&1 | filt

echo "PROBE2_DONE"
