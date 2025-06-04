 
from contextlib import contextmanager
from time import perf_counter
from collections import defaultdict
 
_TIMES = defaultdict(list)

@contextmanager
def timer(label: str): 
    start = perf_counter()
    yield
    _TIMES[label].append(perf_counter() - start)

def print_all_times(title: str = "[TIMERS]"):
 
    print(title, "-------------------------------------------")
    for label, vals in _TIMES.items():
        if len(vals) == 1:
            print(f"{title} {label:<15}= {vals[0]:.6f} s")
        else:
            for i, v in enumerate(vals):
                print(f"{title} {label} [{i}]   = {v:.6f} s")
    print(title, "-------------------------------------------\n")