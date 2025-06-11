 
from contextlib import contextmanager
from time import perf_counter
from collections import defaultdict
from dl_comm.utils.utility import DLCOMMLogger


TIMES = defaultdict(list)



@contextmanager
def timer(label: str):
    start = perf_counter()
    yield
    TIMES[label].append(perf_counter() - start)


def print_all_times(logger, title="[TIMERS]"):
    logger.output(f"{title} -------------------------------------------")
    for label, vals in TIMES.items():
        if len(vals) == 1:
            logger.output(f"{title} {label:<15}= {vals[0]:.6f} s")
             
        else:
            joined = ", ".join(f"{v:.6f}" for v in vals)
            logger.output(f"{title} {label:<15}= [{joined}] s")
            
    logger.output(f"{title} -------------------------------------------\n")


def bytes_per_rank(coll_name, buf_bytes):
    if coll_name == "allreduce":
        return 2 * buf_bytes
    if coll_name == "reduce":
        return buf_bytes
    if coll_name == "broadcast":
        return buf_bytes if dist.get_rank(group=dist.group.WORLD) == 0 else 0
    return buf_bytes


def bytes_per_coll(coll_name, buf_bytes):
    if coll_name == "allreduce":
        return 2 * buf_bytes
    if coll_name == "reduce":
        return buf_bytes
    if coll_name == "broadcast":
        return buf_bytes if dist.get_rank() == 0 else 0
    return buf_bytes


def print_all_bandwidths(logger, buf_bytes, coll_name):
    title = "[BANDWIDTH]"
    logger.output(f"{title} -------------------------------------------")
    for label, vals in TIMES.items():
        if not label.startswith("Latencies"):
            continue
        avg = sum(vals) / len(vals)
        bw = bytes_per_coll(coll_name, buf_bytes) / avg
        logger.output(f"{title} {label:<22}= {bw:,.1f} bytes/sec")
    logger.output(f"{title} -------------------------------------------\n")