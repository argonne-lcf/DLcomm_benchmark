# Deep Learning Communication (DLcomm) Benchmark

**DLComm** is a communication benchmark designed for Deep Learning and AI workloads. Collective communication performance is often the primary bottleneck in AI training, inference, reasoning, and large-scale applications. DLComm emulates the communication patterns of the latest large language models (LLMs) and AI applications at scale, specifically targeting deployments of 50,000 GPUs and beyond.

The benchmark is provided as an executable that can be configured to test various communication patterns within different AI distributed runtime frameworks. It uses a modular design to support all levels of communicator groups across GPUs, with flexible configurations for payload sizes, AI frameworks, and collective communication backends. DLComm enables testing on diverse systems, supports modifying scale-up and scale-out algorithms, and verifies correctness after communication operations.

Unlike traditional communication benchmarks, DLComm is built with the philosophy of reflecting real-world communication performance of the application as accurately as possible. It captures the interplay between Python runtimes, AI frameworks, and collective communication libraries (CCL) to provide insights that are directly relevant to actual AI workloads.

The below gif shows a simple model of how different collective communications are performed over a group of GPUs. Update the below gif with a note - x axis is num_gpus_per_node and y axis is num_compute_nodes. Each sqaure is a GPU on a compute node. Each blinking bright rectangles could represent different collectives executing in an order.

![Alt text](tools/dl_comm_logo.gif)

## Installation and running DLCOMM

pip install -r requirements.txt

pip install DLcomm

## Running the benchmark

## YAML configuration file

Workload characteristics for DL COMM are specified by a YAML configuration file. The main configuration file is located at `dl_comm/config/config.yaml`. A sample configuration file is also available in the `examples/config.yaml` for reference.

Below is an example configuration file

```yaml
framework  : pytorch  # tensorflow / jax / titan / monarch
ccl_backend : ccl   # rccl / nccl / xccl (Note: PyTorch 2.7+ users should use 'xccl' instead of 'ccl' for Intel oneCCL)
ccl_debug   : on # on / off - enables CCL debug logging and algorithm selection reporting
use_profiler: unitrace
barrier     : on    # on / off - on: adds MPI barrier before timer printing for accurate timing, off: only rank 0 prints

comm_group:
  mode: combined # within_node/across_node/combined/flatview -> Only one out of four should be used
  
  flatview:
    num_compute_nodes: 2
    num_gpus_per_node: 12
    gpu_ids_per_node: [0,1,2,3,4,5,6,7,8,9,10,11]   
    collective:
      name: allgather   # allgather / reducescatter / broadcast
      op: sum          # max / min / prod / sum
      scale_up_algorithm: topo
      scale_out_algorithm: ring        # rabinseifner 
      iterations: 5
      payload:
        dtype: bfloat16  # float64 / int32 / int64 / bfloat16 / float8 / float32
        count: 1024
        buffer_size: 1KB # 4096  # in Bytes -> float32(4B) x 1024 elements
    verify_correctness: on

  combined:
    within_node: 
      num_compute_nodes: 2
      num_gpus_per_node: 12
      gpu_ids_per_node: [0,1,2,3,4,5,6,7, 8, 9, 10, 11]   
      collective:
        name: allgather   # allgather / reducescatter / broadcast
        op: sum          # max / min / prod / sum
        scale_up_algorithm: ring
        scale_out_algorithm: ring        # rabinseifner 
        iterations: 2
        payload:
          dtype: bfloat16  # float64 / int32 / int64 / bfloat16 / float8 / float32
          count: 1024
          buffer_size: 1KB # 4096  # in Bytes -> float32(4B) x 1024 elements
      verify_correctness: on

    across_node: 
      num_compute_nodes: 2
      num_gpus_per_node: 3
      gpu_ids_per_node: [0,1,3] 
      collective:
        name: alltoall   # allgather / reducescatter / broadcast
        op: sum          # max / min / prod / sum
        scale_up_algorithm: ring
        scale_out_algorithm: ring        # rabinseifner 
        iterations: 4
        payload:
          dtype: bfloat16  # float64 / int32 / int64 / bfloat16 / float8 / float32
          count: 1024
          buffer_size: 1KB # 4096  # in Bytes -> float32(4B) x 1024 elements
      verify_correctness: on
```

### Important Note for PyTorch Users

**Backend Naming**: The `ccl_backend` field naming depends on your PyTorch version:

- **PyTorch < 2.7**: Use `ccl_backend: ccl` for Intel oneCCL
- **PyTorch 2.7+**: Use `ccl_backend: xccl` for Intel oneCCL

Make sure to use the correct backend name for your PyTorch version to avoid initialization errors.

## How to contribute

We welcome contributions from the community to the benchmark code.
If you would like to contribute, please submit an issue to https://github.com/argonne-lcf/DLcomm_benchmark/issues, and contact ALCF DLCOMM team, Kaushik Velusamy at kaushik.v@anl.gov , Musa Cim at mtc5693@psu.edu

## Citation and Reference

## Acknowledgments

This work used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility under Contract DE-AC02-06CH11357 and is supported in part by National Science Foundation under NSF, OCI-1835764 and NSF, CSR-1814872.

## License

Apache 2.0 LICENSE

Copyright (c) 2025, UChicago Argonne, LLC All Rights Reserved

If you have questions about your rights to use or distribute this software, please contact Argonne Intellectual Property Office at partners@anl.gov

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
