 

from dl_comm.timer.timer import TIMES


def bytes_per_rank(coll_name, buf_bytes):
     
    if coll_name == "allreduce":
        return 2 * buf_bytes
    if coll_name == "reduce":
        return buf_bytes
 
    return buf_bytes


def bytes_per_coll(coll_name, buf_bytes):
   
    if coll_name == "allreduce":
        return 2 * buf_bytes
    if coll_name == "reduce":
        return buf_bytes
    if coll_name == "broadcast":
        try:
            import torch.distributed as dist
            return buf_bytes if dist.get_rank() == 0 else 0
        except:
            return buf_bytes
    return buf_bytes


def print_all_bandwidths(logger, buf_bytes, coll_name):
     
    title = "[BANDWIDTH]"
    logger.output(f"{title} -------------------------------------------")
    for label, vals in TIMES.items():
        if label == "init time" or label == "import time" or label.startswith("Group Creation"):
            continue
        avg = sum(vals) / len(vals)
        bw = bytes_per_coll(coll_name, buf_bytes) / avg
        logger.output(f"{title} {label:<22}= {bw:,.1f} bytes/sec")
    logger.output(f"{title} -------------------------------------------\n")