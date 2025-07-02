import torch
import torch.distributed as dist


def check_collective_correctness(context, tensor_after, collective_name, op=None, group=None, result_data=None, group_type=None, group_id=None):
    if context['iteration'] != 0:
        return
        
    if collective_name == "allreduce":
        _check_allreduce(context, tensor_after, op, group, group_type, group_id)


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
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllReduce verification PASSED - All {world_size} ranks received correct values")
        else:
            failed_ranks = [i for i, result in enumerate(gathered_results) if result.item() == 0]
            log.output(f"[CORRECTNESS][{group_type}-Group-{group_id}] AllReduce verification FAILED - Ranks {failed_ranks} received incorrect values")
    else:
        dist.gather(correct_tensor, None, dst=dst_rank, group=group)