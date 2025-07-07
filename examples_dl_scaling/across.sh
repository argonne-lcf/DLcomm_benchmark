export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((RANDOM + 1024))

# Environment Variables
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600 
export PALS_PMI=pmix # Required by Aurora mpich
export CCL_ATL_TRANSPORT=mpi # Required by Aurora mpich
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export CCL_OP_SYNC=1
export CCL_ENABLE_AUTO_CACHE=0
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=4096

export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_MR_CACHE_MONITOR=kdreg2 #disabled
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=30

export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7

export PALS_PING_PERIOD=480
export PALS_RPC_TIMEOUT=480


mpiexec --envall -np 4 -ppn 2 \
    --cpu-bind $CPU_BIND \
    python \
    "/home/mcim/workspace/gpu-comm-bench/examples_dl_scaling/across.py"

