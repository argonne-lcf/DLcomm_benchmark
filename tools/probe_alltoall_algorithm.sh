#!/bin/bash
#PBS -N a2a_algo
#PBS -l walltime=00:20:00
#PBS -q debug-scaling
#PBS -l filesystems=flare
#PBS -A datascience
#PBS -k doe

# Which oneCCL algorithm is selected for alltoall at 12 ranks vs 24 ranks?
#
# docs/findings/02 records alltoall losing 4.3x (C++) to 6.7x (torch.distributed)
# crossing the node boundary, while sendrecv stays flat (25.9 -> 26.2 GB/s). A
# flat p2p path with a collapsing collective points at algorithm selection
# rather than at raw off-node bandwidth. This probe reads the selection
# directly instead of inferring it.
#
# Submit once per scale from the repository root:
#   qsub -l select=1:ncpus=208 tools/probe_alltoall_algorithm.sh
#   qsub -l select=2:ncpus=208 tools/probe_alltoall_algorithm.sh
#
# Exit status is not the verdict; read the ALGO/SELECTION lines.

set -uo pipefail
cd "$PBS_O_WORKDIR"

set +u
module load frameworks
set -u

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ATL_TRANSPORT=mpi
export CCL_PROCESS_LAUNCHER=pmix
export CCL_KVS_MODE=mpi
export PALS_PMI=pmix
export FI_MR_CACHE_MONITOR=userfaultfd

# The point of the probe: make oneCCL report its algorithm choice.
export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-info}"

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec

export MASTER_ADDR=$(head -1 "$PBS_NODEFILE")
export MASTER_PORT=29517
export WORLD_SIZE=$NRANKS

FULL_LOG="$PBS_O_WORKDIR/tools/a2a_algo_full_${NNODES}node_${CCL_LOG_LEVEL}.log"
echo "=== SCALE: ${NNODES} node(s), ${NRANKS} ranks, CCL_LOG_LEVEL=$CCL_LOG_LEVEL ==="
echo "full log: $FULL_LOG"

# NOTE: /tmp on Aurora compute nodes is node-local tmpfs, so a heredoc written
# here is visible only on the node that runs the jobscript. Ranks on every
# other node would fail with 'No such file or directory'. Write to the shared
# filesystem ($PBS_O_WORKDIR, on flare) so all ranks can read it.
PROBE_PY="$PBS_O_WORKDIR/tools/_a2a_algo_generated.py"
cat > "$PROBE_PY" <<'PYEOF'
import os

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])

torch.xpu.set_device(0)
dev = torch.device("xpu:0")
dist.init_process_group(backend="xccl", rank=rank, world_size=world)

# 4 MiB per rank, matching the finding-02 measurements.
per_rank = 4 * 1024 * 1024 // 4          # float32 elements
total = per_rank * world
send = torch.ones(total, dtype=torch.float32, device=dev)
recv = torch.empty_like(send)

# alltoall: the collective under investigation.
for _ in range(3):
    dist.all_to_all_single(recv, send)
torch.xpu.synchronize()

# sendrecv control: flat across the boundary in the C++ layer, so if the
# fabric itself were degraded this would fall too.
if world >= 2:
    peer = (rank + world // 2) % world
    buf = torch.ones(per_rank, dtype=torch.float32, device=dev)
    for _ in range(3):
        if rank < world // 2:
            dist.send(buf, peer)
            dist.recv(buf, peer)
        else:
            dist.recv(buf, peer)
            dist.send(buf, peer)
    torch.xpu.synchronize()

if rank == 0:
    print(f"[probe] completed at world={world}, {per_rank * 4} bytes per rank")
PYEOF

export PROBE_PY

"$MPIEXEC" --np ${NRANKS} --ppn ${RANKS_PER_NODE} \
    --envall \
    --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100 \
    bash -c 'export RANK=$PALS_RANKID; export LOCAL_RANK=$PALS_LOCAL_RANKID; exec python "$PROBE_PY"' 2>&1 \
  | tee "$FULL_LOG" \
  | grep -iE "selected algo|alltoall|all_to_all|algo|selection|topo|scaleout|scaleup|\[probe\]" \
  | sort -u

echo "=== end ${NNODES}-node selection dump ==="
