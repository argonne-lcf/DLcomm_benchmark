import os
import socket
import jax
import jax.numpy as jnp
import jax.distributed as jdist

# ---------------------------------------------------
# Configuration Parameters
# ---------------------------------------------------
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", 1024))  # Number of elements per device
DTYPE_STR = os.environ.get("DTYPE", "float32")          # e.g., "float32", "float64"

# Map string to JAX dtype
DTYPE_MAP = {
    "float32": jnp.float32,
    "float64": jnp.float64,
    "bfloat16": jnp.bfloat16,
    "int32": jnp.int32,
    "int64": jnp.int64,
}
DTYPE = DTYPE_MAP.get(DTYPE_STR, jnp.float32)

# ---------------------------------------------------
# Rank/World Init
# ---------------------------------------------------
rank = int(os.environ["PMI_RANK"])
world_size = int(os.environ["PMI_SIZE"])
hostname = socket.gethostname()
print(f"Rank {rank} is on host {hostname}")

jdist.initialize(
    coordinator_address=os.environ.get("COORDINATOR_ADDR", "127.0.0.1") + ":1234",
    num_processes=world_size,
    process_id=rank,
)

# ---------------------------------------------------
# Local Buffer Creation
# ---------------------------------------------------
local_device_count = jax.local_device_count()
print(f"[Rank {rank}] Local device count: {local_device_count}")
print(f"[Rank {rank}] Buffer size per device: {BUFFER_SIZE}, dtype: {DTYPE_STR}")

# Shape: [local_device_count, BUFFER_SIZE]
x = jnp.full((local_device_count, BUFFER_SIZE), rank + 1, dtype=DTYPE)

# ---------------------------------------------------
# AllReduce using pmap + psum
# ---------------------------------------------------
def allreduce_sum(x):
    return jax.lax.psum(x, axis_name="i")

y = jax.pmap(allreduce_sum, axis_name="i")(x)

print(f"[Rank {rank}] Input: {x}...")  # Show only first 5 elements for brevity
print(f"[Rank {rank}] AllReduced result (first 5): {y}")

jdist.shutdown()
