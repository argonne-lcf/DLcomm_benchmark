# Limited Native Collective Support in JAX
JAX provides only limited native support for collective operations. Currently, dlcomm includes support for allreduce and allgather with JAX.

# Extending DLComm to Support Additional Frameworks
DLComm can be extended to support additional frameworks like JAX. To add support, search for if framework == "jax" in the codebase to identify relevant integration points.

# Using JAX with RCCL on Frontier
To run JAX with RCCL on the Frontier system, ensure that the appropriate JAX environment is activated. An example activation sequence is included in the provided job script.