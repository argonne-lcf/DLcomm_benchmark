
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
            "sum": lambda x: lax.psum(x, 'i'),
            "max": lambda x: lax.pmax(x, 'i'),
            "min": lambda x: lax.pmin(x, 'i'),
            "mean": lambda x: lax.pmean(x, 'i'),
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
        import jax
        from jax import lax

        reducer_map = {
            "sum":  lax.psum,
            "max":  lax.pmax,
            "min":  lax.pmin,
            "mean": lax.pmean,
        }

        reducer = reducer_map[op]

        
        def allreduce_fn(x):
            return reducer(x, axis_name="i")

        return jax.pmap(allreduce_fn, axis_name="i")(tensor)


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
        pass

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
        pass
    
@register_collective("alltoall", needs_op=False)
def _all_to_all(tensor, op=None, group=None, dist=None,log=None, framework="pytorch"):
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
        
        input_tensor_list = [tensor.clone() for _ in range(world_size)]
        output_tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
        
        return output_tensor_list
    elif framework == 'jax':
        import jax
        from jax import lax

        def alltoall_fn(x):
            return lax.all_to_all(
                x,
                axis_name="i",
                split_axis=1,    
                concat_axis=1
            )


        return jax.pmap(alltoall_fn, axis_name="i")(tensor)
 

@register_collective("allgather", needs_op=False)
def _allgather(tensor, op=None, group=None, dist=None, log=None, framework="pytorch"):
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group=group)
        return tensor_list
    elif framework == 'jax':
        import jax
        from jax import lax
        
        def gather_fn(x):
            return lax.all_gather(x, axis_name='i')

        return jax.pmap(gather_fn, axis_name='i')(tensor)

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
        pass





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
        pass


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
        pass
  



@register_collective("alltoallsingle", needs_op=False)
def _all_to_all_single(tensor, op=None, group=None, dist=None, log=None, framework="pytorch"):
    if framework == 'pytorch':
        output_tensor = torch.empty_like(tensor)
        dist.all_to_all_single(output_tensor, tensor, group=group)
        return output_tensor
    elif framework == 'jax':
        pass

@register_collective("barrier", needs_op=False)
def _barrier(tensor, op=None, group=None, dist=None,log=None, framework="pytorch"):
    if framework == 'pytorch':
        dist.barrier(group=group)
    elif framework == 'jax':
        pass

 
 