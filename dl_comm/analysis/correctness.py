import torch
import torch.distributed as dist


def check_collective_correctness(context, tensor_after, collective_name, op=None, group=None, result_data=None, group_type=None, group_id=None):
    if context['iteration'] != 0:
        return
        
    if collective_name == "allreduce":
        _check_allreduce(context, tensor_after, op, group, group_type, group_id)
    elif collective_name == "reduce":
        _check_reduce(context, tensor_after, op, group, group_type, group_id)
    elif collective_name == "broadcast":
        _check_broadcast(context, tensor_after, op, group, group_type, group_id)
    elif collective_name == "gather":
        _check_gather(context, tensor_after, op, group, group_type, group_id, result_data)
    elif collective_name == "scatter":
        _check_scatter(context, tensor_after, op, group, group_type, group_id)
    elif collective_name == "reducescatter":
        _check_reducescatter(context, tensor_after, op, group, group_type, group_id)
    elif collective_name == "alltoall":
        _check_alltoall(context, tensor_after, op, group, group_type, group_id, result_data)
    elif collective_name == "alltoallsingle":
        _check_alltoallsingle(context, tensor_after, op, group, group_type, group_id, result_data)
    elif collective_name == "allgather":
        _check_allgather(context, tensor_after, op, group, group_type, group_id, result_data)


def _check_allreduce(context, tensor_after, op, group, group_type, group_id):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        group_ranks = list(range(world_size))
        dst_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        dst_rank = min(group_ranks)
    
    if op == dist.ReduceOp.SUM:
        expected_value = world_size
    elif op == dist.ReduceOp.MAX:
        expected_value = 1
    elif op == dist.ReduceOp.MIN:
        expected_value = 1
    elif op == dist.ReduceOp.PRODUCT:
        expected_value = 1
    
    expected_tensor = torch.full_like(tensor_after, expected_value)
    is_correct = torch.allclose(tensor_after, expected_tensor, rtol=1e-6)
    
    correct_tensor = torch.tensor([1 if is_correct else 0], dtype=torch.int32).to(tensor_after.device)
    
    my_rank = dist.get_rank()

    if my_rank== dst_rank:
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
        dist.gather(correct_tensor, gathered_results, dst=dst_rank, group=group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == world_size:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllReduce verification [PASSED] - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllReduce verification [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=dst_rank, group=group)


def _check_reduce(context, tensor_after, op, group, group_type, group_id):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        group_ranks = list(range(world_size))
        dst_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        dst_rank = min(group_ranks)
    
    my_rank = dist.get_rank()
    
    if my_rank == dst_rank:
        if op == dist.ReduceOp.SUM:
            expected_value = world_size
        elif op == dist.ReduceOp.MAX:
            expected_value = 1
        elif op == dist.ReduceOp.MIN:
            expected_value = 1
        elif op == dist.ReduceOp.PRODUCT:
            expected_value = 1
        
        expected_tensor = torch.full_like(tensor_after, expected_value)
        is_correct = torch.allclose(tensor_after, expected_tensor, rtol=1e-6)
        
        if is_correct:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Reduce verification [PASSED] - Root rank received correct value")
        else:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Reduce verification [FAILED] - Root rank received incorrect value")


def _check_broadcast(context, tensor_after, op, group, group_type, group_id):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        group_ranks = list(range(world_size))
        src_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        src_rank = min(group_ranks)
    
    expected_tensor = torch.ones_like(tensor_after)
    is_correct = torch.allclose(tensor_after, expected_tensor, rtol=1e-6)
    
    correct_tensor = torch.tensor([1 if is_correct else 0], dtype=torch.int32).to(tensor_after.device)
    
    my_rank = dist.get_rank()
    
    if my_rank == src_rank:
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
        dist.gather(correct_tensor, gathered_results, dst=src_rank, group=group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == world_size:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Broadcast verification [PASSED] - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Broadcast verification [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=src_rank, group=group)


def _check_gather(context, tensor_after, op, group, group_type, group_id, result_data):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        group_ranks = list(range(world_size))
        dst_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        dst_rank = min(group_ranks)
    
    my_rank = dist.get_rank()
    
    if my_rank == dst_rank:
        
        if result_data is None:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Gather verification [FAILED] - No result data available")
            return
        
         
        expected_tensor = torch.ones_like(tensor_after)
        all_correct = True
        failed_ranks = []
        
        for rank_idx, gathered_tensor in enumerate(result_data):
            if not torch.allclose(gathered_tensor, expected_tensor, rtol=1e-6):
                all_correct = False
                failed_ranks.append(rank_idx)
        
        if all_correct:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Gather verification [PASSED] - Root rank received correct values from all {world_size} ranks")
        else:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Gather verification [FAILED] - Incorrect values received from ranks {failed_ranks}")


def _check_scatter(context, tensor_after, op, group, group_type, group_id):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        src_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        src_rank = min(group_ranks)
    
    expected_tensor = torch.ones_like(tensor_after)
    is_correct = torch.allclose(tensor_after, expected_tensor, rtol=1e-6)
    
    correct_tensor = torch.tensor([1 if is_correct else 0], dtype=torch.int32).to(tensor_after.device)
    
    my_rank = dist.get_rank()
    
    if my_rank == src_rank:
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
        dist.gather(correct_tensor, gathered_results, dst=src_rank, group=group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == world_size:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Scatter verification [PASSED] - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] Scatter verification [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=src_rank, group=group)


def _check_reducescatter(context, tensor_after, op, group, group_type, group_id):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        dst_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        dst_rank = min(group_ranks)
    
    if op == dist.ReduceOp.SUM:
        expected_value = world_size
    elif op == dist.ReduceOp.MAX:
        expected_value = 1
    elif op == dist.ReduceOp.MIN:
        expected_value = 1
    elif op == dist.ReduceOp.PRODUCT:
        expected_value = 1
    
    expected_tensor = torch.full_like(tensor_after, expected_value)
    is_correct = torch.allclose(tensor_after, expected_tensor, rtol=1e-6)
    
    correct_tensor = torch.tensor([1 if is_correct else 0], dtype=torch.int32).to(tensor_after.device)
    
    my_rank = dist.get_rank()

    if my_rank == dst_rank:
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
        dist.gather(correct_tensor, gathered_results, dst=dst_rank, group=group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == world_size:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] ReduceScatter verification [PASSED] - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] ReduceScatter verification [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=dst_rank, group=group)


def _check_alltoall(context, tensor_after, op, group, group_type, group_id, result_data):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        dst_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        dst_rank = min(group_ranks)
    
    my_rank = dist.get_rank()
    
    if result_data is None:
        if my_rank == dst_rank:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllToAll verification [FAILED] - No result data available")
        return
    
    expected_tensor = torch.ones_like(tensor_after)
    all_correct = True
    failed_tensor_indices = []
    
    for tensor_idx, received_tensor in enumerate(result_data):
        if not torch.allclose(received_tensor, expected_tensor, rtol=1e-6):
            all_correct = False
            failed_tensor_indices.append(tensor_idx)
    
    correct_tensor = torch.tensor([1 if all_correct else 0], dtype=torch.int32).to(tensor_after.device)
    
    if my_rank == dst_rank:
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
        dist.gather(correct_tensor, gathered_results, dst=dst_rank, group=group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == world_size:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllToAll verification [PASSED] - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllToAll verification [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=dst_rank, group=group)


def _check_alltoallsingle(context, tensor_after, op, group, group_type, group_id, result_data):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        dst_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        dst_rank = min(group_ranks)
    
    my_rank = dist.get_rank()
    
    if result_data is None:
        if my_rank == dst_rank:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllToAllSingle verification [FAILED] - No result data available")
        return
    
    # For alltoallsingle, result_data should be a single tensor, not a list
    expected_tensor = torch.ones_like(tensor_after)
    all_correct = torch.allclose(result_data, expected_tensor, rtol=1e-6)
    
    correct_tensor = torch.tensor([1 if all_correct else 0], dtype=torch.int32).to(tensor_after.device)
    
    if my_rank == dst_rank:
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
        dist.gather(correct_tensor, gathered_results, dst=dst_rank, group=group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == world_size:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllToAllSingle verification [PASSED] - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllToAllSingle verification [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=dst_rank, group=group)


def _check_allgather(context, tensor_after, op, group, group_type, group_id, result_data):
    log = context['log']
    world_size = dist.get_world_size(group)
    
    if group is None:
        dst_rank = 0
    else:
        group_ranks = dist.get_process_group_ranks(group)
        dst_rank = min(group_ranks)
    
    my_rank = dist.get_rank()
    
    if result_data is None:
        if my_rank == dst_rank:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllGather verification [FAILED] - No result data available")
        return
    
    expected_tensor = torch.ones_like(tensor_after)
    all_correct = True
    failed_tensor_indices = []
    
    for tensor_idx, gathered_tensor in enumerate(result_data):
        if not torch.allclose(gathered_tensor, expected_tensor, rtol=1e-6):
            all_correct = False
            failed_tensor_indices.append(tensor_idx)
    
    correct_tensor = torch.tensor([1 if all_correct else 0], dtype=torch.int32).to(tensor_after.device)
    
    if my_rank == dst_rank:
        gathered_results = [torch.zeros_like(correct_tensor) for _ in range(world_size)]
        dist.gather(correct_tensor, gathered_results, dst=dst_rank, group=group)
        
        total_correct = sum(result.item() for result in gathered_results)
        if total_correct == world_size:
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllGather verification [PASSED] - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllGather verification [FAILED] - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=dst_rank, group=group)

