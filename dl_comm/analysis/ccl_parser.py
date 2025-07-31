 
import re


def parse_ccl_selection(log_path: str, algo_name: str):
    sel = {}
    start_re = re.compile(rf"\b{re.escape(algo_name)}\b selection", re.IGNORECASE)
    table_re = re.compile(r'^\s*([a-z ]+table)\s*$', re.IGNORECASE)
    choice_re = re.compile(r'^\s*\[.*?\]\s*:\s*(\S+)\s*$', re.IGNORECASE)
    with open(log_path) as f:
        lines = f.readlines()
    for idx, L in enumerate(lines):
        if start_re.search(L):
            break
    else:
        return sel
    current_table = None
    for L in lines[idx+1:]:
        if re.match(r'^\d{4}:\d{2}.*\|CCL_', L):
            break
        m_table = table_re.match(L)
        m_choice = choice_re.match(L)
        if m_table:
            current_table = m_table.group(1).strip()
        elif m_choice and current_table:
            sel[current_table] = m_choice.group(1).strip()
    return sel


def get_ccl_table_name(collective_name: str) -> str:
    """Map collective names to their CCL table selection names"""
    mapping = {
        'allreduce': 'allreduce',
        'reduce': 'reduce', 
        'broadcast': 'broadcast',
        'allgather': 'allgather',
        'gather': 'gather',
        'scatter': 'scatter',
        'reducescatter': 'reduce_scatter',  # reducescatter uses reduce_scatter table
        'alltoall': 'alltoall',
        'alltoallsingle': 'alltoall',  # alltoallsingle uses alltoall table
        'barrier': 'barrier'
    }
    return mapping.get(collective_name.lower(), collective_name.lower())


def get_readable_table_name(table_name: str) -> str:
    """Map CCL internal table names to more readable descriptions"""
    mapping = {
        'main table': 'scale_up table',
        'fallback table': 'scale_up fallback',
        'scaleout table': 'scale_out table'
    }
    return mapping.get(table_name.lower(), table_name)


def report_ccl_selection(log_path: str, algo_name: str, logger, scale_up_config=None, scale_out_config=None):
    # Map the algorithm name to the correct table name for CCL selection
    table_name = get_ccl_table_name(algo_name)
    
    selection = parse_ccl_selection(log_path, table_name)
    if not selection:
        logger.info(f"No '{table_name} selection' block found in {log_path}")
    else:
        logger.info(f"[SELECTION] {table_name} table selection:")
        for tbl, impl in selection.items():
            readable_name = get_readable_table_name(tbl)
            
            # Add user config info in parentheses
            user_config = ""
            if tbl.lower() == 'main table':
                if not scale_up_config or scale_up_config.strip() == '':
                    user_config = " (user's selection: N/A (default))"
                elif scale_up_config.lower() == 'default':
                    user_config = " (user's selection: default)"
                else:
                    user_config = f" (user's selection: {scale_up_config})"
            elif tbl.lower() == 'scaleout table':
                if not scale_out_config or scale_out_config.strip() == '':
                    user_config = " (user's selection: N/A (default))"
                elif scale_out_config.lower() == 'default':
                    user_config = " (user's selection: default)"
                else:
                    user_config = f" (user's selection: {scale_out_config})"
            
            logger.info(f"[SELECTION] {readable_name:17s} → {impl}{user_config}")
        
        # Add note if user selection doesn't match CCL selection
        if scale_up_config and scale_up_config != 'default':
            main_selection = selection.get('main table', '')
            if main_selection and main_selection.lower() != scale_up_config.lower():
                logger.info(f"[SELECTION] NOTE: CCL overrode user scale_up algorithm '{scale_up_config}' with '{main_selection}'. Check oneCCL documentation for available algorithms and hardware constraints.")
        
        if scale_out_config and scale_out_config != 'default':
            scaleout_selection = selection.get('scaleout table', '')
            if scaleout_selection and scaleout_selection.lower() != scale_out_config.lower():
                logger.info(f"[SELECTION] NOTE: CCL overrode user scale_out algorithm '{scale_out_config}' with '{scaleout_selection}'. Check oneCCL documentation for available algorithms and hardware constraints.")