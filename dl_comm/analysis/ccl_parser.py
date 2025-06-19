 
import re


def parse_ccl_selection(log_path: str, algo_name: str):
    sel = {}
    start_re = re.compile(rf"{re.escape(algo_name)} selection", re.IGNORECASE)
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


def report_ccl_selection(log_path: str, algo_name: str, logger):
   
    selection = parse_ccl_selection(log_path, algo_name)
    if not selection:
        logger.info(f"No '{algo_name} selection' block found in {log_path}")
    else:
        logger.info(f"[SELECTION] {algo_name} table selection:")
        for tbl, impl in selection.items():
            logger.info(f"[SELECTION] {tbl:15s} → {impl}")