
import torch
import torch.distributed as dist


COLLECTIVES: dict[str, callable] = {}
OPS_NEED_REDUCE: set[str] = set()          

OP_MAP: dict[str, dist.ReduceOp] = {
    "sum":  dist.ReduceOp.SUM,
    "max":  dist.ReduceOp.MAX,
    "min":  dist.ReduceOp.MIN,
    "prod": dist.ReduceOp.PRODUCT,
}


DTYPES = {
    "float16": (torch.float16, 2),
    "bfloat16": (torch.bfloat16, 2),
    "float32": (torch.float32, 4),
    "float64": (torch.float64, 8),
    "int32":   (torch.int32,   4),
    "int64":   (torch.int64,   8),
    "float8": (torch.float32, 4), 
}


def register_collective(name: str, needs_op: bool = False):

    name = name.lower()

    def decorator(func):
        COLLECTIVES[name] = func
        if needs_op:
            OPS_NEED_REDUCE.add(name)
        return func

    return decorator

@register_collective("allreduce", needs_op=True)
def _allreduce(tensor, op, group=None, dist=None, log=None):
    dist.all_reduce(tensor, op=op, group=group)


@register_collective("reduce", needs_op=True)
def _reduce(tensor, op, group=None, dist=None,log=None):
    if group is None:
        smallest_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        smallest_rank = min(group_ranks)
    dist.reduce(tensor, dst=smallest_rank, op=op, group=group)


@register_collective("broadcast", needs_op=False)      
def _broadcast(tensor, op, group=None, dist=None, log=None):
    if group is None:
        smallest_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        smallest_rank = min(group_ranks)
    dist.broadcast(tensor, src=smallest_rank, group=group)
    
 

@register_collective("allgather", needs_op=False)
def _allgather(tensor, op=None, group=None, dist=None, log=None):
    world_size = dist.get_world_size(group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)
    return tensor_list

@register_collective("gather", needs_op=False)
def _gather(tensor, op=None, group=None, dist=None, log=None):
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
        

@register_collective("scatter", needs_op=False)
def _scatter(tensor, op=None, group=None, dist=None,log=None):
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



@register_collective("reducescatter", needs_op=True)
def _reduce_scatter(tensor, op, group=None, dist=None,log=None):
    world_size = dist.get_world_size(group)
    global_rank = dist.get_rank()   

    input_list = []
    for i in range(world_size):
        chunk = tensor.clone()  
        input_list.append(chunk)
    
    dist.reduce_scatter(tensor, input_list, op=op, group=group)
  

@register_collective("alltoall", needs_op=False)
def _all_to_all(tensor, op=None, group=None, dist=None,log=None):
    world_size = dist.get_world_size(group)
    
    input_tensor_list = [tensor.clone() for _ in range(world_size)]
    output_tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    
    dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
    return output_tensor_list


@register_collective("alltoallsingle", needs_op=False)
def _all_to_all_single(tensor, op=None, group=None, dist=None, log=None):

  
    output_tensor = torch.empty_like(tensor)
    dist.all_to_all_single(output_tensor, tensor, group=group)
    return output_tensor
     

@register_collective("barrier", needs_op=False)
def _barrier(tensor, op=None, group=None, dist=None,log=None):
    dist.barrier(group=group)

 