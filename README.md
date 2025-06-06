# Deep Learning I/O (DLIO) Benchmark
![test status] ???

This README provides an abbreviated documentation of the DL_COMM_code. Please refer to ... for full user documentation. 


## Overview

DL COMM is a lightweight benchmark for testing common communication patterns in large‐scale deep‐learning ( all‐reduce, broadcast, all‐gather). You run it from a single executable and configure everything with a simple YAML file. It’s modular—so adding new frameworks, back-ends, or algorithms is easy. DL COMM reports per‐iteration latency and bandwidth, can optionally generate Perfetto traces via DFTracer, and uses a small logger (DLIOLogger) to keep all output consistent across ranks.

## Installation and running DLIO
### Bare metal installation 


### Bare metal installation with profiler



## Container

## PowerPC


## Lassen, LLNL



## Running the benchmark
 


## YAML configuration file 
 

Workload characteristics for DL COMM are specified by a YAML configuration file. Below is an example of a YAML file for a DL COMM run that executes a PyTorch+XCCL ring all-reduce across 4 nodes with 8 GPUs each, sending a 1 MB float32 buffer for 10 iterations:

```yaml
# contents of dl_comm_run.yaml

# --- Which framework and backend to use ---
framework: pytorch
ccl_backend: xccl       # options: xccl, nccl, …

# --- Collective settings ---
collective:
  name: allreduce       # options: allreduce, reduce, broadcast, allgather, …
  algo: ring            # options: ring, tree ...
  op: sum               # valid when collective is a reduction (sum, max, min, prod)
  iterations: 10
  payload:
    buffer_size: 1MB    # can be specified as “1MB”, “512KB”, “1048576B”, etc.
    dtype: float32      # options: float32, float64, int32, int64, …

# --- Tensor-parallel (horizontal) configuration ---
horizontal:
  tp_degree: 8          # number of GPUs per node
  gpu_ids: [0,1,2,3,4,5,6,7]

# --- Data-parallel (vertical) configuration ---
vertical:
  dp_degree: 4          # number of nodes

# --- If true, skips TP/DP grouping and use a single “flat” communicator 
flatview: false

# --- Enable Tracer/Perfetto 
use_unitrace: true



```



## How to contribute 

## Citation and Reference



## Acknowledgments

## License
