 
from contextlib import contextmanager
from time import perf_counter
from collections import defaultdict
from mpi4py import MPI
import re
TIMES = defaultdict(list)    

def reset_times():
    TIMES.clear()

@contextmanager
def timer(label: str):
    start = perf_counter()
    yield
    TIMES[label].append(perf_counter() - start)



def gather_and_print_all_times(logger, ranks_responsible_for_logging, barrier_enabled, title="[TIMERS]", phase_filter=None, collective_name=None):
    # ========================================================================
    # GATHER TIMER DATA FROM ALL RANKS
    # ========================================================================
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    
    my_data = None
    if mpi_rank in ranks_responsible_for_logging:
        my_data = {
            'rank': mpi_rank,
            'timers': dict(TIMES)
        }
    
    all_data = MPI.COMM_WORLD.gather(my_data, root=0)
    if mpi_rank == 0:
        logger.output("")
        logger.output(f"{title} -------------------------------------------")
        
        # ========================================================================
        # CATEGORIZE TIMER DATA BY GROUP TYPE
        # ========================================================================
        group_timers = {}
        
        for data in all_data:
            if data is not None:
                rank = data['rank']
                timers = data['timers']
                
                for label, vals in timers.items():
                    if "import time" == label:
                        group_key = "import"
                    elif "init time" == label:
                        group_key = "init"
                    elif "Group Creation (Within)" == label:
                        group_key = "group_creation_within"
                    elif "Group Creation (Across)" == label:
                        group_key = "group_creation_across"
                    elif "(flatview)" == label.lower():
                        group_key = "flatview"
                    elif "(within-group-" in label.lower():
                        match = re.search(r'\(within-group-(\d+)\)', label.lower())
                        if match:
                            group_key = f"within-{match.group(1)}"
                        else:
                            group_key = "within-unknown"
                    elif "(across-group-" in label.lower():
                        match = re.search(r'\(across-group-(\d+)\)', label.lower())
                        if match:
                            group_key = f"across-{match.group(1)}"
                        else:
                            group_key = "across-unknown"
                    else:
                        continue
                    
                    if group_key not in group_timers:
                        group_timers[group_key] = {}
                    
                    if label not in group_timers[group_key] or rank < group_timers[group_key][label]['rank']:
                        group_timers[group_key][label] = {
                            'vals': vals,
                            'rank': rank
                        }
        
        # ========================================================================
        # APPLY PHASE FILTER AND ORGANIZE GROUPS FOR ORDERED OUTPUT
        # ========================================================================
        group_order = []
        
        if phase_filter == "setup":
            pass
        elif phase_filter == "within":
            within_groups = [k for k in group_timers.keys() if k.startswith("within-")]
            within_groups.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 999)
            group_order.extend(within_groups)
        elif phase_filter == "across":
            across_groups = [k for k in group_timers.keys() if k.startswith("across-")]
            across_groups.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 999)
            group_order.extend(across_groups)
        else:
            within_groups = [k for k in group_timers.keys() if k.startswith("within-")]
            within_groups.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 999)
            group_order.extend(within_groups)
            
            across_groups = [k for k in group_timers.keys() if k.startswith("across-")]
            across_groups.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 999)
            group_order.extend(across_groups)
            
            flatview_groups = [k for k in group_timers.keys() if k == "flatview"]
            group_order.extend(flatview_groups)

        
        # ========================================================================
        # PRINT SETUP TIMERS (import, init, group creation)
        # ========================================================================
        setup_order = []
        
        if phase_filter == "setup":
            available_group_creation = [k for k in group_timers.keys() if k.startswith("group_creation_")]
            setup_order = ["import", "init"] + available_group_creation
        elif phase_filter == "within":
            setup_order = []
        elif phase_filter == "across":
            setup_order = []
        else:
            setup_order = []
        
        for group_key in setup_order:
            if group_key in group_timers:
                for label, timer_data in group_timers[group_key].items():
                    vals = timer_data['vals']
                    rank = timer_data['rank']
                    if len(vals) == 1:
                        logger.output(f"[TIMERS][LOGGING RANK - {rank}] {label:<25}= {vals[0]:.6f} s")
                    else:
                        joined = ", ".join(f"{v:.6f}" for v in vals)
                        logger.output(f"[TIMERS][LOGGING RANK - {rank}] {label:<25}= [{joined}] s")
        
        # ========================================================================
        # PRINT COMMUNICATION TIMERS
        # ========================================================================
        iteration_data = {}
        
        for group_key in group_order:
            if group_key in group_timers:
                for label, timer_data in group_timers[group_key].items():
                    vals = timer_data['vals']
                    rank = timer_data['rank']
                    
                    if len(vals) > 1:
                        iteration_data[label] = {'vals': vals, 'rank': rank}
                    else:
                        logger.output(f"[TIMERS][LOGGING RANK - {rank}] {label:<25}= {vals[0]:.6f} s")
        
        # ========================================================================
        # PRINT BARRIER STATUS
        # ========================================================================
        if phase_filter != "setup":
            logger.output("")
            if barrier_enabled:
                logger.info(f"  {title} [BARRIER ENABLED] Timing measurements used MPI barriers for synchronization")
            else:
                logger.info(f"  {title} [BARRIER DISABLED] Warning: Timing without barriers - other collectives may still be in process")
        
        # ========================================================================
        # PRINT ITERATION TABLE FOR MULTI-RUN TIMERS
        # ========================================================================
        if iteration_data:
            logger.output("")
            if collective_name:
                logger.output(f"[TIMERS] ITERATION TABLE FOR {collective_name.upper()}:")
            else:
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
                        row += f"{vals[i]:.6f}".center(col_width)
                    else:
                        row += f"{'-':^{col_width}}"
                logger.output(row)
            
            logger.output(separator)
            logger.output("")
        
        # ========================================================================
        # END TIMER REPORT
        # ========================================================================
        logger.output(f"{title} -------------------------------------------")


def print_all_times(logger, title="[TIMERS]"):
    logger.output("")
    logger.output(f"{title} -------------------------------------------")
    
    for label, vals in TIMES.items():
        if len(vals) == 1:
            logger.output(f"{title} {label:<25}= {vals[0]:.6f} s")
        else:
            joined = ", ".join(f"{v:.6f}" for v in vals)
            logger.output(f"{title} {label:<25}= [{joined}] s")

    logger.output(f"{title} -------------------------------------------\n")
