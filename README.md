# Deep Learning Communication (DLcomm) Benchmark

This README provides an abbreviated documentation of the DL_COMM_code. Please refer to ... for full user documentation.

## Overview

DL COMM is a lightweight benchmark for testing common communication patterns in large‐scale deep‐learning ( all‐reduce, broadcast, all‐gather). You run it from a single executable and configure everything with a simple YAML file. It’s modular—so adding new frameworks, back-ends, or algorithms is easy. DL COMM reports per‐iteration latency and bandwidth.

update - x axis is num_gpus_per_node and y axis is num_compute_nodes

![Alt text](tools/dl_comm_logo.gif)

## Installation and running DLCOMM

pip install -r requirements.txt

pip install DLcomm

## Running the benchmark

## YAML configuration file

Workload characteristics for DL COMM are specified by a YAML configuration file. Below is an example of a YAML file for a DL COMM run that executes a PyTorch+XCCL ring all-reduce across 4 nodes with 8 GPUs each, sending a 1 MB float32 buffer for 10 iterations:

```yaml
# contents of dl_comm_run.yaml
framework.  : pytorch  # tensorflow / jax / titan / monarch
ccl_backend : xccl   # rccl / nccl
use_profiler: unitrace

collective:
  name: allreduce   # allgather / reducescatter / broadcast
  op: prod          # max / min 
  scale_up_algorithm: topo
  scale_out_algorithm: ring        # rabinseifner 
  iterations: 5
  payload:
    dtype: float32  # float64 / int32 / int64 / bfloat16 / float8 
    count: 1024
    buffer_size: 4096 # in Bytes -> float32(4B) x 1024 elements
  
  verify_correctness: on
  
  comm_group:
    mode: combined  # within_node/across_node/combined/flatview -> Only one out of four should be used
  
    flatview: off
  
    within_node: 
      num_compute_nodes: 2 
      num_gpus_per_node: 4
      gpu_ids_per_node: [8, 9, 10, 11]   
  
    across_node: 
      num_compute_nodes: 2
      num_gpus_per_node: 2
      gpu_ids_per_node: [0,1] 
  
    combined:
      within_node:
        num_compute_nodes: 2
        num_gpus_per_node: 12  
        gpu_ids_per_node: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
      across_node:
        num_compute_nodes: 2
        num_gpus_per_node: 2  
        gpu_ids_per_node: [5, 7]



```

## How to contribute

## Citation and Reference

## Acknowledgments

## License
