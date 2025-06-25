
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
def _allreduce(tensor, op, group=None, dist=None):
    dist.all_reduce(tensor, op=op, group=group)


@register_collective("reduce", needs_op=True)
def _reduce(tensor, op, group=None, dist=None):
    # Find the smallest global rank in the group to use as destination
    group_ranks = dist.get_process_group_ranks(group)
    smallest_rank = min(group_ranks)
    dist.reduce(tensor, dst=smallest_rank, op=op, group=group)


@register_collective("broadcast")      
def _broadcast(tensor, group=None, dist=None):
    # Find the smallest global rank in the group to use as source
    group_ranks = dist.get_process_group_ranks(group)
    smallest_rank = min(group_ranks)
    dist.broadcast(tensor, src=smallest_rank, group=group)


@register_collective("allgather", needs_op=False)
def _allgather(tensor, op=None, group=None, dist=None):
    # Create tensor_list internally for benchmark usage
    world_size = dist.get_world_size(group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)
    return tensor_list  # Return pure list - no concatenation


@register_collective("reducescatter", needs_op=True)
def _reduce_scatter(tensor, op, group=None, dist=None):
    # Simple ReduceScatter testing with ones
    world_size = dist.get_world_size(group)
    global_rank = dist.get_rank()   
    group_rank = dist.get_rank(group)   
    
     
    input_list = []
    for i in range(world_size):
        chunk = tensor.clone()  
        input_list.append(chunk)
    
    dist.reduce_scatter(tensor, input_list, op=op, group=group)
    
    # Return diagnostic info for validation
    return {
        'global_rank': global_rank,
        'group_rank': group_rank,
        'my_chunk_index': group_rank,
        'expected_value': float(world_size)  # Each element should equal world_size after SUM
    }


@register_collective("gather", needs_op=False)
def _gather(tensor, op=None, group=None, dist=None):
    # Find the smallest global rank in the group to use as destination
    group_ranks = dist.get_process_group_ranks(group)
    smallest_rank = min(group_ranks)
    world_size = dist.get_world_size(group)
    global_rank = dist.get_rank()  # My global rank
    
    if global_rank == smallest_rank:
        # I am the destination rank - create gather_list
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list, dst=smallest_rank, group=group)
        return gather_list
    else:
        # I am not the destination - gather_list must be None
        dist.gather(tensor, None, dst=smallest_rank, group=group)
        return None


@register_collective("scatter", needs_op=False)
def _scatter(tensor, op=None, group=None, dist=None):
    # Find the smallest global rank in the group to use as source
    group_ranks = dist.get_process_group_ranks(group)
    smallest_rank = min(group_ranks)
    global_rank = dist.get_rank()  # My global rank
    world_size = dist.get_world_size(group)
    group_rank = dist.get_rank(group)
    
    if global_rank == smallest_rank:
        # I am the source rank - create scatter_list with unique data for each rank
        scatter_list = []
        for i in range(world_size):
            # Create unique tensor for each rank (filled with group rank number)
            unique_tensor = torch.full_like(tensor, float(i))
            scatter_list.append(unique_tensor)
        
        dist.scatter(tensor, scatter_list, src=smallest_rank, group=group)
        
        # Return info showing what was scattered
        return {
            'type': 'source',
            'source_global_rank': global_rank,
            'scattered_data': [{'to_group_rank': i, 'value': float(i), 'tensor_sum': float(i * tensor.numel())} 
                             for i in range(world_size)]
        }
    else:
        # I am not the source - scatter_list must be None
        dist.scatter(tensor, None, src=smallest_rank, group=group)
        
        # Return info showing what I received
        return {
            'type': 'receiver',
            'receiver_global_rank': global_rank,
            'receiver_group_rank': group_rank,
            'expected_value': float(group_rank),
            'received_tensor_sum': float(tensor.sum()),
            'is_correct': float(tensor.sum()) == float(group_rank * tensor.numel())
        }


@register_collective("alltoall", needs_op=False)
def _all_to_all(tensor, op=None, group=None, dist=None):
    world_size = dist.get_world_size(group)
    global_rank = dist.get_rank()
    group_rank = dist.get_rank(group)
    
    # Create meaningful input data: each chunk I send has a unique pattern
    # Chunk for rank i will be filled with value: (my_group_rank * 100 + i)
    input_tensor_list = []
    chunk_size = tensor.numel() // world_size
    
    for i in range(world_size):
        chunk = torch.full((chunk_size,), float(group_rank * 100 + i), 
                          dtype=tensor.dtype, device=tensor.device)
        input_tensor_list.append(chunk)
    
    # Prepare output buffers
    output_tensor_list = [torch.empty_like(chunk) for chunk in input_tensor_list]
    
    # Perform AllToAll exchange
    dist.all_to_all(output_tensor_list, input_tensor_list, group=group)
    
    # Prepare validation information
    sent_data = []
    received_data = []
    
    for i in range(world_size):
        # What I sent to rank i
        sent_value = float(group_rank * 100 + i)
        sent_data.append({
            'to_group_rank': i,
            'sent_value': sent_value,
            'chunk_sum': sent_value * chunk_size
        })
        
        # What I received from rank i  
        received_chunk = output_tensor_list[i]
        received_sum = float(received_chunk.sum())
        expected_value = float(i * 100 + group_rank)  # What rank i should have sent me
        expected_sum = expected_value * chunk_size
        
        received_data.append({
            'from_group_rank': i,
            'expected_value': expected_value,
            'expected_sum': expected_sum,
            'received_sum': received_sum,
            'is_correct': abs(received_sum - expected_sum) < max(abs(expected_sum) * 0.01, 50.0)  # 1% tolerance or 50.0, whichever is larger
        })
    
    return {
        'global_rank': global_rank,
        'group_rank': group_rank,
        'sent_data': sent_data,
        'received_data': received_data,
        'concatenated_result': torch.cat(output_tensor_list)
    }


@register_collective("barrier", needs_op=False)
def _barrier(tensor, op=None, group=None, dist=None):
    dist.barrier(group=group)

 