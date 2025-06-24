 
from contextlib import contextmanager
from time import perf_counter_ns
from collections import defaultdict
from mpi4py import MPI

TIMES = defaultdict(list)    

@contextmanager
def timer(label: str):
    start = perf_counter_ns()
    yield
    TIMES[label].append(perf_counter_ns() - start)

def gather_and_print_all_times(logger, ranks_responsible_for_logging, title="[TIMERS]"):
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    
    my_data = None
    if mpi_rank in ranks_responsible_for_logging:
        my_data = {
            'rank': mpi_rank,
            'timers': dict(TIMES)
        }
    
    all_data = MPI.COMM_WORLD.gather(my_data, root=0)
    
    if mpi_rank == 0:
        logger.output(f"{title} -------------------------------------------")
        
        group_timers = {}
        
        for data in all_data:
            if data is not None:
                rank = data['rank']
                timers = data['timers']
                
                for label, vals in timers.items():
                    if "import" in label.lower():
                        group_key = "import"
                    elif "init" in label.lower():
                        group_key = "init"
                    elif "group creation (within)" in label.lower():
                        group_key = "within_creation"
                    elif "group creation (across)" in label.lower():
                        group_key = "across_creation"
                    elif "within-group-" in label.lower():
                        import re
                        match = re.search(r'within-group-(\d+)', label.lower())
                        if match:
                            group_key = f"within-{match.group(1)}"
                        else:
                            group_key = "within-unknown"
                    elif "across-group-" in label.lower():
                        import re
                        match = re.search(r'across-group-(\d+)', label.lower())
                        if match:
                            group_key = f"across-{match.group(1)}"
                        else:
                            group_key = "across-unknown"
                    elif "flatview" in label.lower():
                        group_key = "flatview"
                    elif "total" in label.lower():
                        group_key = "combined"
                    else:
                        group_key = "other"
                    
                    if group_key not in group_timers:
                        group_timers[group_key] = {}
                    
                    if label not in group_timers[group_key] or rank < group_timers[group_key][label]['rank']:
                        group_timers[group_key][label] = {
                            'vals': vals,
                            'rank': rank
                        }
        
        group_order = []
        
        within_groups = [k for k in group_timers.keys() if k.startswith("within-")]
        within_groups.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 999)
        group_order.extend(within_groups)
        
        across_groups = [k for k in group_timers.keys() if k.startswith("across-")]
        across_groups.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 999)
        group_order.extend(across_groups)
        
        other_groups = [k for k in group_timers.keys() if not k.startswith(("within-", "across-"))]
        group_order.extend(other_groups)
        
        setup_order = ["import", "init", "within_creation", "across_creation", "other"]
        for group_key in setup_order:
            if group_key in group_timers:
                for label, timer_data in group_timers[group_key].items():
                    vals = timer_data['vals']
                    rank = timer_data['rank']
                    if len(vals) == 1:
                        logger.output(f"[TIMERS][LOGGING RANK - {rank}] {label:<25}= {vals[0]} ns")
                    else:
                        joined = ", ".join(f"{v}" for v in vals)
                        logger.output(f"[TIMERS][LOGGING RANK - {rank}] {label:<25}= [{joined}] ns")
        
        iteration_data = {}
        non_iteration_data = {}
        
        for group_key in group_order:
            if group_key in group_timers and group_key not in setup_order:
                for label, timer_data in group_timers[group_key].items():
                    vals = timer_data['vals']
                    rank = timer_data['rank']
                    
                    if len(vals) > 1:
                        iteration_data[label] = {'vals': vals, 'rank': rank}
                    else:
                        non_iteration_data[label] = {'vals': vals, 'rank': rank}
        
        for label, timer_data in non_iteration_data.items():
            vals = timer_data['vals']
            rank = timer_data['rank']
            logger.output(f"[TIMERS][LOGGING RANK - {rank}] {label:<25}= {vals[0]} ns")
        
        if iteration_data:
            logger.output("")
            logger.output("[TIMERS] ITERATION TABLE:")
            
            headers = list(iteration_data.keys())
            max_iterations = max(len(data['vals']) for data in iteration_data.values())
            
            col_width = 20
            
            header_line1 = f"{'Iteration':<12}"
            for label in headers:
                header_line1 += f"{label:^{col_width}}"
            logger.output(header_line1)
            
            header_line2 = f"{'':12}"
            for label in headers:
                rank = iteration_data[label]['rank']
                rank_str = f"LOGGING RANK - {rank}"
                header_line2 += f"{rank_str:^{col_width}}"
            logger.output(header_line2)
            
            separator = "-" * len(header_line1)
            logger.output(separator)
            
            for i in range(max_iterations):
                row = f"{i:<12}"
                for label in headers:
                    vals = iteration_data[label]['vals']
                    if i < len(vals):
                        row += f"{vals[i]:^{col_width}}"
                    else:
                        row += f"{'-':^{col_width}}"
                logger.output(row)
            
            logger.output(separator)
            logger.output("")
        
        logger.output(f"{title} -------------------------------------------\n")


def print_all_times(logger, title="[TIMERS]"):
    logger.output("")
    logger.output(f"{title} -------------------------------------------")
    
    for label, vals in TIMES.items():
        if len(vals) == 1:
            logger.output(f"{title} {label:<25}= {vals[0]} ns")
        else:
            joined = ", ".join(f"{v}" for v in vals)
            logger.output(f"{title} {label:<25}= [{joined}] ns")

    logger.output(f"{title} -------------------------------------------\n")
