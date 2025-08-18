
COLLECTIVES: dict[str, callable] = {}
OPS_NEED_REDUCE: set[str] = set()          

OP_MAP: dict = {}
DTYPES: dict = {}
torch = None
dist = None

def init_framework_constants(framework):
    global OP_MAP, DTYPES, torch, dist

    if framework == 'pytorch':
        import torch as torch_module
        import torch.distributed as dist_module
        torch = torch_module
        dist = dist_module

        
        OP_MAP.clear()
        OP_MAP.update({
            "sum":  dist.ReduceOp.SUM,
            "max":  dist.ReduceOp.MAX,
            "min":  dist.ReduceOp.MIN,
            "prod": dist.ReduceOp.PRODUCT,
        })

        DTYPES.clear()
        DTYPES.update({
            "float16":  (torch.float16, 2),
            "bfloat16": (torch.bfloat16, 2),
            "float32":  (torch.float32, 4),
            "float64":  (torch.float64, 8),
            "int32":    (torch.int32,   4),
            "int64":    (torch.int64,   8),
        })

    elif framework == 'jax':
        import jax
        import jax.numpy as jnp
        from jax import lax
        
        OP_MAP.clear()
        OP_MAP.update({
            "sum": lax.add,
            "max": lax.max,
            "min": lax.min,
            "prod": lax.mul,
        })

        DTYPES.clear()
        DTYPES.update({
            "float16":  (jnp.float16, 2),
            "bfloat16": (jnp.bfloat16, 2),
            "float32":  (jnp.float32, 4),
            "float64":  (jnp.float64, 8),
            "int32":    (jnp.int32,   4),
            "int64":    (jnp.int64,   8),
        })


def register_collective(name: str, needs_op: bool = False):

    name = name.lower()

    def decorator(func):
        COLLECTIVES[name] = func
        if needs_op:
            OPS_NEED_REDUCE.add(name)
        return func

    return decorator

@register_collective("allreduce", needs_op=True)
def _allreduce(tensor, op, group=None, dist=None, log=None, framework="pytorch"):
    if framework == 'pytorch':
        dist.all_reduce(tensor, op=op, group=group)
    elif framework == 'jax':
        from jax import lax
        tensor = lax.all_reduce(tensor, op, axis_name='devices')


@register_collective("reduce", needs_op=True)
def _reduce(tensor, op, group=None, dist=None,log=None, framework="pytorch"):
    if framework == 'pytorch':
        if group is None:
            smallest_rank = 0
        else:
            group_ranks = dist.get_process_group_ranks(group)
            smallest_rank = min(group_ranks)
        dist.reduce(tensor, dst=smallest_rank, op=op, group=group)
    elif framework == 'jax':
        from jax import lax
        # Reduce to rank 0 (or smallest rank in group)
        tensor = lax.reduce_scatter(tensor, op, scatter_dimension=0, axis_name='devices', tiled=True)


@register_collective("broadcast", needs_op=False)      
def _broadcast(tensor, op, group=None, dist=None, log=None, framework="pytorch"):
    if framework == 'pytorch':
        if group is None:
            smallest_rank = 0
        else:
            group_ranks = dist.get_process_group_ranks(group)
            smallest_rank = min(group_ranks)
        dist.broadcast(tensor, src=smallest_rank, group=group)
    elif framework == 'jax':
        from jax import lax
        # Broadcast from rank 0 (or smallest rank in group) 
        tensor = lax.broadcast_in_dim(tensor, tensor.shape, axis_name='devices')
    
 

@register_collective("allgather", needs_op=False)
def _allgather(tensor, op=None, group=None, dist=None, log=None, framework="pytorch"):
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group=group)
        return tensor_list
    elif framework == 'jax':
        from jax import lax
        # All gather along the first dimension
        return lax.all_gather(tensor, axis_name='devices')

@register_collective("gather", needs_op=False)
def _gather(tensor, op=None, group=None, dist=None, log=None, framework="pytorch"):
    if framework == 'pytorch':
        if group is None:
            smallest_rank = 0
        else:
            group_ranks = dist.get_process_group_ranks(group)
            smallest_rank = min(group_ranks)
        world_size = dist.get_world_size(group)
        global_rank = dist.get_rank()
        
        if global_rank == smallest_rank:
            gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.gather(tensor, gather_list, dst=smallest_rank, group=group)
            return gather_list
        else:
            dist.gather(tensor, None, dst=smallest_rank, group=group)
            return None
    elif framework == 'jax':
        from jax import lax
        # Gather to rank 0, return None on other ranks for consistency
        return lax.all_gather(tensor, axis_name='devices')
        





@register_collective("scatter", needs_op=False)
def _scatter(tensor, op=None, group=None, dist=None,log=None,framework="pytorch"):
    if framework == 'pytorch':
        if group is None:
            smallest_rank = 0
        else:
            group_ranks = dist.get_process_group_ranks(group)
            smallest_rank = min(group_ranks)
        world_size = dist.get_world_size(group)
        global_rank = dist.get_rank()
        
        if global_rank == smallest_rank:
            scatter_list = [tensor.clone() for _ in range(world_size)]
            dist.scatter(tensor, scatter_list, src=smallest_rank, group=group)
        else:
            dist.scatter(tensor, None, src=smallest_rank, group=group)
    elif framework == 'jax':
        from jax import lax
        # Scatter from rank 0 across devices
        return lax.dynamic_slice(tensor, (lax.axis_index('devices'),), (1,))




@register_collective("reducescatter", needs_op=True)
def _reduce_scatter(tensor, op, group=None, dist=None,log=None, framework="pytorch"):
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
      
        chunk_size = tensor.numel() // world_size
        input_list = []
        
        for i in range(world_size):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = tensor[start_idx:end_idx].contiguous()
            input_list.append(chunk)
        
     
        output_tensor = torch.empty_like(input_list[0])
        dist.reduce_scatter(output_tensor, input_list, op=op, group=group)
        
        return output_tensor
    elif framework == 'jax':
        from jax import lax
        # Reduce scatter operation
        return lax.reduce_scatter(tensor, op, scatter_dimension=0, axis_name='devices')
  

@register_collective("alltoall", needs_op=False)
def _all_to_all(tensor, op=None, group=None, dist=None,log=None, framework="pytorch"):
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
        
        input_tensor_list = [tensor.clone() for _ in range(world_size)]
        output_tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
        
        return output_tensor_list
    elif framework == 'jax':
        from jax import lax
        # All-to-all communication
        return lax.all_to_all(tensor, axis_name='devices', split_axis=0, concat_axis=0)


@register_collective("alltoallsingle", needs_op=False)
def _all_to_all_single(tensor, op=None, group=None, dist=None, log=None, framework="pytorch"):
    if framework == 'pytorch':
        output_tensor = torch.empty_like(tensor)
        dist.all_to_all_single(output_tensor, tensor, group=group)
        return output_tensor
    elif framework == 'jax':
        from jax import lax
        # Single tensor all-to-all
        return lax.all_to_all(tensor, axis_name='devices', split_axis=0, concat_axis=0)
     

@register_collective("barrier", needs_op=False)
def _barrier(tensor, op=None, group=None, dist=None,log=None, framework="pytorch"):
    if framework == 'pytorch':
        dist.barrier(group=group)
    elif framework == 'jax':
        # JAX doesn't have explicit barriers, but we can do a dummy all_reduce
        from jax import lax
        import jax.numpy as jnp
        dummy = jnp.array(0)
        lax.all_reduce(dummy, lax.add, axis_name='devices')

 
 