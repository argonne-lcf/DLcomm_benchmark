 
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


