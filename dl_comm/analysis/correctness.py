"""Correctness verification for DLcomm collectives.

Every checker below derives its expected result from the *rank-dependent*
payload produced by :mod:`dl_comm.verify.payload`, so that a collective which
transferred nothing, transferred the wrong chunk, or shuffled the rank order is
detected. See ``docs/fixes/01-rank-dependent-verification.md``.

All checkers:
  * compute a local verdict,
  * reduce the verdict across the group so a failure on any rank is reported,
  * record the outcome in :mod:`dl_comm.verify.failures` so the run can exit
    nonzero (``docs/fixes/04-fail-loudly.md``).
"""

from dl_comm.verify import (
    build_payload,
    scatter_source,
    position_term,
    rank_signature,
    expected_reduction,
    choose_moduli,
    tolerance_for,
    failures,
)

# Reverse lookup from a torch ReduceOp back to its config name.
_OP_NAMES = ("sum", "max", "min", "prod")


def _op_to_name(op, dist):
    if op is None:
        return None
    mapping = {
        dist.ReduceOp.SUM: "sum",
        dist.ReduceOp.MAX: "max",
        dist.ReduceOp.MIN: "min",
        dist.ReduceOp.PRODUCT: "prod",
    }
    for key, name in mapping.items():
        if op == key:
            return name
    return None


def _group_info(dist, group):
    """Return ``(group_ranks, world_size, root_rank, my_index)``.

    ``group`` is normally a ``torch.distributed`` ProcessGroup, but the
    torchcomms backend passes a ``TorchCommsGroup``, which is a plain object
    holding a communicator and its rank list. It is not registered with
    ``torch.distributed``, so ``dist.get_world_size`` reaches
    ``group.size()`` and raises AttributeError. Read the rank list directly
    when it is present.
    """
    ranks_attr = getattr(group, "ranks", None)
    if ranks_attr is not None:
        group_ranks = list(ranks_attr)
        world_size = len(group_ranks)
    else:
        world_size = dist.get_world_size(group)
        if group is None:
            group_ranks = list(range(world_size))
        else:
            group_ranks = list(dist.get_process_group_ranks(group))
    root = min(group_ranks)
    my_rank = dist.get_rank()
    my_index = group_ranks.index(my_rank) if my_rank in group_ranks else None
    return group_ranks, world_size, root, my_index


def _reduce_verdict(context, dist, torch, tensor_like, group, group_ranks, root,
                    local_ok, label, extra=""):
    """All-reduce the local boolean so any rank's failure fails the whole group.

    Uses ``all_reduce`` with MIN rather than a root-only ``gather`` so that the
    verdict is consistent on every rank and no rank can skip the collective.
    """
    log = context["log"]
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
    if hasattr(tensor_like, "device"):
        flag = flag.to(tensor_like.device)

    # Works for both backends: the torchcomms adapter exposes ReduceOp and
    # unwraps a TorchCommsGroup itself.
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    group_ok = bool(flag.item() == 1)

    iteration = context.get("iteration", "?")
    if not group_ok:
        detail = f"{label} iteration {iteration} [FAILED]"
        if extra:
            detail += f" - {extra}"
        # Only the group root writes the log line, but every rank records the
        # failure so the MPI-wide reduction at end of run sees it.
        if dist.get_rank() == root:
            log.output(f"[CORRECTNESS]{detail}")
        failures.record_failure(detail)
    else:
        failures.record_pass()
    return group_ok


def _skip(context, label, reason):
    iteration = context.get("iteration", "?")
    detail = f"{label} iteration {iteration} - {reason}"
    context["log"].output(f"[CORRECTNESS]{detail}")
    failures.record_skip(detail)


def check_collective_correctness(context, tensor_after, collective_name, op=None,
                                 group=None, result_data=None, group_type=None,
                                 group_id=None):
    framework = context["cfg"].framework.lower()
    if framework != "pytorch":
        return
    if collective_name == "barrier":
        return

    import torch
    import torch.distributed as dist

    # Under ccl_backend: torchcomms the collectives run through the adapter,
    # not torch.distributed, and the groups they produce are TorchCommsGroup
    # objects that torch.distributed cannot introspect or reduce over. Use the
    # same adapter the run used, so the checker talks to the communicator that
    # actually carried the data.
    if getattr(group, "comm", None) is not None:
        from dl_comm.comm import torchcomms_backend as _tcb
        active = _tcb.active_dist()
        if active is not None:
            dist = active

    handler = _HANDLERS.get(collective_name)
    if handler is None:
        return

    label = f"[{group_type}-Group-{group_id}] {collective_name}"
    handler(context, tensor_after, op, group, group_id, result_data, dist, torch, label)


# ---------------------------------------------------------------------------
# Reductions: every rank ends with the reduction of all rank signatures.
# ---------------------------------------------------------------------------

def _expected_reduced(torch, tensor, op_name, world_size, rank_mod, pos_mod):
    """Reduction of ``pos + sig`` across the group, elementwise."""
    n = tensor.numel()
    pos = position_term(torch, n, pos_mod)
    if op_name == "prod":
        # positional term is disabled for prod (pos_mod == 1)
        value = expected_reduction("prod", world_size, rank_mod)
        return torch.full_like(tensor, value)

    sigs = [rank_signature(i, world_size, rank_mod, op_name) for i in range(world_size)]
    if op_name == "sum":
        # sum over ranks of (pos + sig) = world_size*pos + sum(sig)
        expected = pos * world_size + sum(sigs)
    elif op_name == "max":
        expected = pos + max(sigs)
    elif op_name == "min":
        expected = pos + min(sigs)
    else:
        raise ValueError(f"unsupported op '{op_name}'")
    return expected.to(tensor.dtype).to(tensor.device)


def _check_reduction(context, tensor_after, op, group, group_id, result_data,
                     dist, torch, label, root_only):
    op_name = _op_to_name(op, dist)
    if op_name is None:
        _skip(context, label, "no reduction op supplied")
        return

    group_ranks, world_size, root, my_index = _group_info(dist, group)
    rank_mod, pos_mod = choose_moduli(tensor_after.dtype, world_size, op_name)
    rtol, atol = tolerance_for(tensor_after.dtype)

    expected = _expected_reduced(torch, tensor_after, op_name, world_size,
                                 rank_mod, pos_mod)

    if root_only and dist.get_rank() != root:
        # Non-root ranks of `reduce` hold undefined data; they must still take
        # part in the verdict all_reduce.
        local_ok = True
    else:
        local_ok = bool(torch.allclose(tensor_after.to(expected.dtype), expected,
                                       rtol=rtol, atol=atol))

    _reduce_verdict(context, dist, torch, tensor_after, group, group_ranks, root,
                    local_ok, label,
                    extra=f"op={op_name}, group_size={world_size}")


def _check_allreduce(context, t, op, group, gid, result, dist, torch, label):
    _check_reduction(context, t, op, group, gid, result, dist, torch, label, False)


def _check_reduce(context, t, op, group, gid, result, dist, torch, label):
    _check_reduction(context, t, op, group, gid, result, dist, torch, label, True)


def _check_reducescatter(context, t, op, group, gid, result, dist, torch, label):
    op_name = _op_to_name(op, dist)
    if op_name is None:
        _skip(context, label, "no reduction op supplied")
        return
    if result is None:
        _skip(context, label, "collective returned no result data")
        return

    group_ranks, world_size, root, my_index = _group_info(dist, group)
    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, op_name)
    rtol, atol = tolerance_for(t.dtype)

    # This rank receives chunk `my_index` of the reduced buffer. The positional
    # term therefore starts at my_index * chunk_size, which is what makes a
    # wrong-chunk bug detectable.
    chunk = result.numel()
    offset = (my_index if my_index is not None else 0) * chunk
    pos = position_term(torch, chunk, pos_mod, offset=offset)

    sigs = [rank_signature(i, world_size, rank_mod, op_name) for i in range(world_size)]
    if op_name == "sum":
        expected = pos * world_size + sum(sigs)
    elif op_name == "max":
        expected = pos + max(sigs)
    elif op_name == "min":
        expected = pos + min(sigs)
    else:
        expected = torch.full_like(pos, expected_reduction("prod", world_size, rank_mod))
    expected = expected.to(result.dtype).to(result.device)

    local_ok = bool(torch.allclose(result.to(expected.dtype), expected,
                                   rtol=rtol, atol=atol))
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=f"op={op_name}, chunk={chunk}, index={my_index}")


# ---------------------------------------------------------------------------
# Data movement: position and source rank both matter.
# ---------------------------------------------------------------------------

def _check_broadcast(context, t, op, group, gid, result, dist, torch, label):
    group_ranks, world_size, root, my_index = _group_info(dist, group)
    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)

    # Everyone must end up holding the ROOT's payload, i.e. group index 0.
    expected = build_payload(torch, t.numel(), t.dtype, 0, world_size, None,
                             device=t.device, rank_modulus=rank_mod,
                             position_modulus=pos_mod)
    local_ok = bool(torch.allclose(t, expected, rtol=rtol, atol=atol))
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=f"expected root(index 0) payload")


def _check_allgather(context, t, op, group, gid, result, dist, torch, label):
    group_ranks, world_size, root, my_index = _group_info(dist, group)
    if result is None:
        _skip(context, label, "collective returned no result data")
        return
    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)

    bad = []
    for i, got in enumerate(result):
        expected = build_payload(torch, got.numel(), got.dtype, i, world_size, None,
                                 device=got.device, rank_modulus=rank_mod,
                                 position_modulus=pos_mod)
        if not torch.allclose(got, expected, rtol=rtol, atol=atol):
            bad.append(i)

    local_ok = not bad and len(result) == world_size
    extra = f"wrong/misordered slots {bad}" if bad else f"got {len(result)} of {world_size} slots"
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=extra)


def _check_gather(context, t, op, group, gid, result, dist, torch, label):
    group_ranks, world_size, root, my_index = _group_info(dist, group)
    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)
    is_root = dist.get_rank() == root

    if is_root:
        if result is None:
            _skip(context, label, "root received no gather list")
            return
        bad = []
        for i, got in enumerate(result):
            expected = build_payload(torch, got.numel(), got.dtype, i, world_size,
                                     None, device=got.device,
                                     rank_modulus=rank_mod, position_modulus=pos_mod)
            if not torch.allclose(got, expected, rtol=rtol, atol=atol):
                bad.append(i)
        local_ok = not bad and len(result) == world_size
        extra = f"wrong/misordered slots {bad}" if bad else ""
    else:
        local_ok = True
        extra = ""

    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=extra)


def _check_scatter(context, t, op, group, gid, result, dist, torch, label):
    group_ranks, world_size, root, my_index = _group_info(dist, group)
    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)

    # This rank must receive exactly the slice the root addressed to it.
    expected = scatter_source(torch, t.numel(), t.dtype,
                              my_index if my_index is not None else 0,
                              world_size, rank_mod, pos_mod, device=t.device)
    local_ok = bool(torch.allclose(t, expected, rtol=rtol, atol=atol))
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=f"expected slice for index {my_index}")


def _check_alltoall(context, t, op, group, gid, result, dist, torch, label):
    group_ranks, world_size, root, my_index = _group_info(dist, group)
    if result is None:
        _skip(context, label, "collective returned no result data")
        return
    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)

    # Slot i of the output came from group member i, who sent its own payload.
    bad = []
    for i, got in enumerate(result):
        expected = build_payload(torch, got.numel(), got.dtype, i, world_size, None,
                                 device=got.device, rank_modulus=rank_mod,
                                 position_modulus=pos_mod)
        if not torch.allclose(got, expected, rtol=rtol, atol=atol):
            bad.append(i)

    local_ok = not bad and len(result) == world_size
    extra = f"wrong/misordered slots {bad}" if bad else ""
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=extra)


def _check_alltoallsingle(context, t, op, group, gid, result, dist, torch, label):
    group_ranks, world_size, root, my_index = _group_info(dist, group)
    if result is None:
        _skip(context, label, "collective returned no result data")
        return
    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)

    # all_to_all_single splits the buffer into world_size contiguous chunks;
    # chunk i of the output is chunk `my_index` of member i's input.
    n = result.numel()
    chunk = n // world_size if world_size else n
    parts = []
    for i in range(world_size):
        sig = rank_signature(i, world_size, rank_mod, None)
        offset = (my_index if my_index is not None else 0) * chunk
        parts.append(position_term(torch, chunk, pos_mod, offset=offset) + sig)
    expected = torch.cat(parts).to(result.dtype).to(result.device) if parts else None

    if expected is None or expected.numel() != n:
        _skip(context, label, f"cannot model expected layout (n={n}, ws={world_size})")
        return

    local_ok = bool(torch.allclose(result.to(expected.dtype), expected,
                                   rtol=rtol, atol=atol))
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=f"chunk={chunk}, index={my_index}")


def _check_alltoallv(context, t, op, group, gid, result, dist, torch, label):
    """Verify the uneven-split exchange.

    Rank r receives, from every peer j, the slice of j's payload that j
    assigned to r. Because all ranks compute the same split table, the
    expected content of each received chunk is fully determined.
    """
    from dl_comm.comm.collectives import _uneven_splits

    group_ranks, world_size, root, my_index = _group_info(dist, group)
    if result is None:
        _skip(context, label, "collective returned no result data")
        return
    if my_index is None:
        _skip(context, label, "rank index within group is unknown")
        return

    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)

    splits = _uneven_splits(t.numel(), world_size, my_index)
    my_share = splits[my_index]

    if result.numel() != my_share * world_size:
        _reduce_verdict(context, dist, torch, t, group, group_ranks, root,
                        False, label,
                        extra=(f"received {result.numel()} elems, expected "
                               f"{my_share * world_size} "
                               f"({world_size} peers x {my_share})"))
        return

    # chunk j of the output is peer j's elements [offset : offset+my_share],
    # where offset is the sum of the shares j assigned to ranks before us
    offset = sum(splits[:my_index])
    parts = []
    for j in range(world_size):
        sig = rank_signature(j, world_size, rank_mod, None)
        parts.append(position_term(torch, my_share, pos_mod, offset=offset) + sig)

    expected = torch.cat(parts).to(result.dtype).to(result.device)
    local_ok = bool(torch.allclose(result.to(expected.dtype), expected,
                                   rtol=rtol, atol=atol))
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=f"uneven splits={splits}, my_share={my_share}")


def _check_sendrecv(context, t, op, group, gid, result, dist, torch, label):
    """Verify the pairwise exchange: each rank must hold its partner's payload.

    A rank left idle by an odd group size records a skip rather than a pass,
    so an all-idle configuration cannot look like a clean verification.
    """
    group_ranks, world_size, root, my_index = _group_info(dist, group)
    if my_index is None:
        _skip(context, label, "rank index within group is unknown")
        return

    if world_size < 2:
        _skip(context, label, f"group of {world_size} has no pairs")
        return

    if world_size % 2 and my_index == world_size - 1:
        _skip(context, label, f"odd group size {world_size}; this rank is idle")
        return

    if result is None:
        _skip(context, label, "collective returned no result data")
        return

    rank_mod, pos_mod = choose_moduli(t.dtype, world_size, None)
    rtol, atol = tolerance_for(t.dtype)

    partner = my_index + 1 if my_index % 2 == 0 else my_index - 1
    expected = build_payload(torch, result.numel(), result.dtype, partner,
                             world_size, None, device=result.device,
                             rank_modulus=rank_mod, position_modulus=pos_mod)

    local_ok = bool(torch.allclose(result, expected, rtol=rtol, atol=atol))
    _reduce_verdict(context, dist, torch, t, group, group_ranks, root, local_ok,
                    label, extra=f"partner={partner}")


_HANDLERS = {
    "allreduce": _check_allreduce,
    "reduce": _check_reduce,
    "reducescatter": _check_reducescatter,
    "broadcast": _check_broadcast,
    "allgather": _check_allgather,
    "gather": _check_gather,
    "scatter": _check_scatter,
    "alltoall": _check_alltoall,
    "alltoallsingle": _check_alltoallsingle,
    "alltoallv": _check_alltoallv,
    "sendrecv": _check_sendrecv,
    "sendrecv_async": _check_sendrecv,
}
