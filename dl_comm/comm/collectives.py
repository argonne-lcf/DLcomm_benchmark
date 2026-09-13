
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
            # Each destination must receive a DISTINCT buffer, otherwise a
            # scatter that delivered the wrong slice (or delivered nothing)
            # cannot be detected. See docs/fixes/01-rank-dependent-verification.md
            from dl_comm.verify import scatter_source, choose_moduli
            rank_mod, pos_mod = choose_moduli(tensor.dtype, world_size, None)
            scatter_list = [
                scatter_source(torch, tensor.numel(), tensor.dtype, i,
                               world_size, rank_mod, pos_mod, device=tensor.device)
                for i in range(world_size)
            ]
            dist.scatter(tensor, scatter_list, src=smallest_rank, group=group)
        else:
            dist.scatter(tensor, None, src=smallest_rank, group=group)
        return tensor
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


# ---------------------------------------------------------------------------
# Vector and point-to-point operations
#
# See docs/fixes/10-p2p-and-vector-collectives.md
#
# alltoallv exercises the uneven-split path that alltoall/alltoallsingle
# cannot: real workloads (MoE routing, unbalanced sharding) send a different
# element count to every peer, and that path has different performance and
# different failure modes from the equal-split case.
#
# sendrecv is not a collective but is the standard pairwise
# latency/bandwidth measurement (the osu_latency / osu_bw equivalent), which
# the tool previously had no way to express.
# ---------------------------------------------------------------------------


def _uneven_splits(total_elems, world_size, group_rank):
    """Deterministic, rank-dependent, non-uniform split of `total_elems`.

    Every rank computes the identical split table, so the send and receive
    sides agree without extra communication. The distribution is deliberately
    skewed: rank i's share grows with i, so an implementation that silently
    assumes equal chunks produces wrong sizes rather than passing by luck.
    """
    if world_size == 1:
        return [total_elems]
    weights = [i + 1 for i in range(world_size)]
    wsum = sum(weights)
    splits = [max(1, (total_elems * w) // wsum) for w in weights]
    # absorb the rounding remainder into the last entry
    splits[-1] += total_elems - sum(splits)
    if splits[-1] < 1:
        # fall back to an equal split when the buffer is too small to skew
        base = total_elems // world_size
        splits = [base] * world_size
        splits[-1] += total_elems - base * world_size
    return splits


@register_collective("alltoallv", needs_op=False)
def _all_to_all_v(tensor, op=None, group=None, dist=None, log=None,
                  framework="pytorch"):
    """all_to_all_single with uneven input/output split sizes."""
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
        group_rank = dist.get_rank(group)

        total = tensor.numel()
        # what this rank sends to each peer
        out_splits = _uneven_splits(total, world_size, group_rank)
        # what this rank receives is peer j's share destined for us; because
        # every rank uses the same table, rank i receives out_splits[group_rank]
        # elements from each peer j
        in_splits = [out_splits[group_rank]] * world_size

        send = tensor[:sum(out_splits)].contiguous()
        recv = torch.empty(sum(in_splits), dtype=tensor.dtype,
                           device=tensor.device)

        dist.all_to_all_single(recv, send,
                               output_split_sizes=in_splits,
                               input_split_sizes=out_splits,
                               group=group)
        return recv
    elif framework == 'jax':
        pass


@register_collective("sendrecv", needs_op=False)
def _send_recv(tensor, op=None, group=None, dist=None, log=None,
               framework="pytorch"):
    """Pairwise point-to-point exchange.

    Ranks are paired (0,1), (2,3), ... within the group. The even member sends
    then receives; the odd member receives then sends. This ordering avoids the
    deadlock that symmetric blocking send-first would cause. A group with an
    odd rank count leaves the final rank idle, which is reported rather than
    silently ignored.
    """
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
        group_rank = dist.get_rank(group)

        if world_size < 2:
            if log is not None:
                log.warning("[SENDRECV] group has fewer than 2 ranks; nothing to do")
            return None

        # translate group-local rank to the global rank dist.send/recv expect
        if group is None:
            global_of = lambda r: r          # noqa: E731
        else:
            ranks = dist.get_process_group_ranks(group)
            global_of = lambda r: ranks[r]   # noqa: E731

        if world_size % 2 and group_rank == world_size - 1:
            if log is not None and group_rank == world_size - 1:
                log.warning(
                    f"[SENDRECV] odd group size {world_size}; rank {group_rank} idle")
            return None

        partner_local = group_rank + 1 if group_rank % 2 == 0 else group_rank - 1
        partner = global_of(partner_local)
        recv = torch.empty_like(tensor)

        if group_rank % 2 == 0:
            dist.send(tensor, dst=partner, group=group)
            dist.recv(recv, src=partner, group=group)
        else:
            dist.recv(recv, src=partner, group=group)
            dist.send(tensor, dst=partner, group=group)

        return recv
    elif framework == 'jax':
        pass


@register_collective("sendrecv_async", needs_op=False)
def _send_recv_async(tensor, op=None, group=None, dist=None, log=None,
                     framework="pytorch"):
    """Non-blocking pairwise exchange via isend/irecv.

    Measures the same pairing as `sendrecv` but with both directions in flight
    simultaneously, which is the bidirectional-bandwidth case.
    """
    if framework == 'pytorch':
        world_size = dist.get_world_size(group)
        group_rank = dist.get_rank(group)

        if world_size < 2:
            if log is not None:
                log.warning("[SENDRECV_ASYNC] group has fewer than 2 ranks")
            return None

        if group is None:
            global_of = lambda r: r          # noqa: E731
        else:
            ranks = dist.get_process_group_ranks(group)
            global_of = lambda r: ranks[r]   # noqa: E731

        if world_size % 2 and group_rank == world_size - 1:
            return None

        partner_local = group_rank + 1 if group_rank % 2 == 0 else group_rank - 1
        partner = global_of(partner_local)
        recv = torch.empty_like(tensor)

        # Order by rank parity, exactly as the blocking _send_recv does.
        #
        # XCCL's isend/irecv are not async at enqueue: the first call blocks
        # until its peer posts the matching operation. Two jobs proved this --
        # 8824532 had every rank call isend first and all 22 participants hung
        # in isend; 8824561 swapped to irecv first and all 22 hung in irecv.
        # The failing property is not which call comes first, it is that BOTH
        # peers issue the SAME call first, so nothing can ever match.
        #
        # Parity ordering pairs an isend on one peer with an irecv on the
        # other. Both requests are still outstanding before either wait(), so
        # the two directions remain in flight together and this is still the
        # bidirectional case rather than a blocking exchange.
        if group_rank % 2 == 0:
            reqs = [dist.isend(tensor, dst=partner, group=group),
                    dist.irecv(recv, src=partner, group=group)]
        else:
            reqs = [dist.irecv(recv, src=partner, group=group),
                    dist.isend(tensor, dst=partner, group=group)]
        for r in reqs:
            r.wait()

        return recv
    elif framework == 'jax':
        pass

 
 