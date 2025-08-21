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

Workload characteristics for DL COMM are specified by a YAML configuration file. Multiple example configurations are available in the `examples/` directory, organized in numbered folders (e.g., `examples/1_simple_flat/`, `examples/2_multimode/`). Each folder contains complete configuration files and corresponding job scripts.

Below is an example configuration file

```yaml
framework        : pytorch
ccl_backend      : ccl   # rccl / nccl / xccl (Note: PyTorch 2.7+ users should use 'xccl' instead of 'ccl' for Intel oneCCL)
extended_logging : off
barrier          : on    # on / off - on: adds MPI barrier before timer printing for accurate timing, off: only rank 0 prints
device_type      : gpu
memory_source    : gpu

order_of_run: [simple-allreduce]

simple-allreduce:
  comm_group: flatview
  num_compute_nodes: 2
  num_devices_per_node: 12
  device_ids_per_node: [0,1,2,3,4,5,6,7,8,9,10,11]
  verify_correctness: on
  collective:
    collective_name: allreduce  # allgather / reducescatter / broadcast / allreduce / alltoall
    collective_op: sum          # max / min / prod / sum
    scale_up_algorithm: default
    scale_out_algorithm: default
    iterations: 5
    warmup_iterations: 2
    add_mxm_compute: on
    payload:
      dtype: float32  # float64 / int32 / int64 / bfloat16 / float8 / float32
      count: 
      buffer_size: 100KB
```

### Example 2: Multi-mode Communication

The `examples/2_multimode/` directory demonstrates running multiple collective operations with different communication group modes in a single benchmark run. This example shows:

- **Sequential Execution**: Two different collectives run in order
- **Within-node**: AllGather operation across GPUs within the same node
- **Across-node**: AllReduce operation across GPUs on different nodes
- **Memory Source**: Host memory instead of GPU memory
- **Buffer Size**: 512KB for both operations

```yaml
framework        : pytorch
ccl_backend      : ccl
extended_logging : off
barrier          : on
device_type      : gpu
memory_source    : host

order_of_run: [within-node-allgather, across-node-allreduce]

within-node-allgather:
  comm_group: within_node
  num_compute_nodes: 2
  num_devices_per_node: 12
  device_ids_per_node: [0,1,2,3,4,5,6,7,8,9,10,11]
  verify_correctness: on
  collective:
    collective_name: allgather
    collective_op: sum
    scale_up_algorithm: default
    scale_out_algorithm: default
    iterations: 5
    warmup_iterations: 2
    add_mxm_compute: on
    payload:
      dtype: float32
      count: 
      buffer_size: 512KB

across-node-allreduce:
  comm_group: across_node
  num_compute_nodes: 2
  num_devices_per_node: 12
  device_ids_per_node: [0,1,2,3,4,5,6,7,8,9,10,11]
  verify_correctness: on
  collective:
    collective_name: allreduce
    collective_op: sum
    scale_up_algorithm: default
    scale_out_algorithm: default
    iterations: 5
    warmup_iterations: 2
    add_mxm_compute: on
    payload:
      dtype: float32
      count: 
      buffer_size: 512KB
```

## RCCL and JAX Support

### RCCL with PyTorch

DLcomm supports AMD's ROCm Collective Communications Library (RCCL) for AMD GPU systems. The `examples/8_rccl_pytorch/` directory demonstrates comprehensive collective communication testing with RCCL backend.

**Key Features:**
- **All Collective Operations**: Tests 10 different collective operations (allreduce, allgather, reducescatter, broadcast, reduce, alltoall, alltoallsingle, gather, scatter, barrier)
- **RCCL Backend**: Uses `ccl_backend: nccl` for RCCL integration with PyTorch
- **AMD GPU Optimized**: Configured for AMD GPU systems with ROCm

**Job Script Requirements:**
```bash
# Environment modules
module load miniforge3/23.11.0-0
module load cray-python
module load rocm/6.2.4

# Network configuration for AMD systems
export NCCL_SOCKET_IFNAME=hsn0
export MASTER_ADDR=$(ip -4 addr show dev hsn0 | awk '/inet/{print $2}' | cut -d/ -f1)

# SLURM execution
srun --ntasks=16 --export=ALL --cpu-bind=threads \
  python3 -m dl_comm.dl_comm_main \
  --config-path="$SCRIPT_DIR" \
  --config-name=8_rccl_pytorch
```

### JAX Support (Experimental)

DLcomm provides experimental support for JAX framework with limited collective operations. The `examples/9_rccl_jax/` directory demonstrates JAX integration.

**Current Limitations:**
- **Experimental Status**: JAX support is under active development
- **Limited Collectives**: Only 2 collective operations currently supported (allreduce, allgather)
- **Verification Disabled**: Correctness verification is turned off (`verify_correctness: off`)

**JAX Configuration:**
```yaml
framework        : jax
ccl_backend      : nccl
barrier          : off
order_of_run: [allreduce-jax, allgather-jax]
```

**JAX Job Script Requirements:**
```bash
# JAX-specific environment setup
eval "$(/sw/frontier/miniforge3/23.11.0-0/bin/conda shell.bash hook)"
conda activate jax_env-frontier

# JAX platform configuration
export JAX_PLATFORMS=rocm
export COORDINATOR_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export COORDINATOR_PORT=1234

# SLURM execution with JAX-specific parameters
srun --ntasks=16 --gpus-per-task=1 --cpus-per-task=1 --export=ALL \
  python3 -m dl_comm.dl_comm_main \
  --config-path="$SCRIPT_DIR" \
  --config-name=9_rccl_jax
```

**Note**: JAX support is experimental and may have limitations compared to PyTorch. Only allreduce and allgather operations are currently implemented.

### Important Note for PyTorch Users

**Backend Naming**: The `ccl_backend` field naming depends on your PyTorch version:

- **PyTorch < 2.7**: Use `ccl_backend: ccl` for Intel oneCCL
- **PyTorch 2.7+**: Use `ccl_backend: xccl` for Intel oneCCL

Make sure to use the correct backend name for your PyTorch version to avoid initialization errors.

## Correctness Verification

DLComm includes built-in correctness verification for all collective operations. When `verify_correctness: on` is set in the configuration:

- **Verification Scope**: Correctness is checked on **all iterations** to ensure consistent behavior
- **Failure-Only Reporting**: Correctness results are **only printed when failures occur** to reduce log noise
- **Detailed Diagnostics**: Failed verifications include iteration number and specific rank information
- **Comprehensive Coverage**: All collective operations (AllReduce, AllGather, ReduceScatter, etc.) are validated


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
