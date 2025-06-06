
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
    "float32": (torch.float32, 4),
    "float64": (torch.float64, 8),
    "int32":   (torch.int32,   4),
    "int64":   (torch.int64,   8),
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
def _allreduce(tensor, op, group=None):
    dist.all_reduce(tensor, op=op, group=group)


@register_collective("reduce", needs_op=True)
def _reduce(tensor, op, group=None):
    dist.reduce(tensor, dst=0, op=op, group=group)


@register_collective("broadcast")      
def _broadcast(tensor, op=None, group=None):
    dist.broadcast(tensor, src=0, group=group)


@register_collective("allgather")
def _allgather(tensor, op=None, group=None):
    world = dist.get_world_size(group)
    out   = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(out, tensor, group=group)


@register_collective("reducescatter", needs_op=True)
def _reduce_scatter(tensor, op, group=None):
    world  = dist.get_world_size(group)
    inbufs = [tensor.clone() for _ in range(world)]
    dist.reduce_scatter(tensor, inbufs, op=op, group=group)
