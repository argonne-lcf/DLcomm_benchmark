import re
import subprocess
import os
import shutil
from datetime import datetime
import glob
import time

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

def filter_logs_post_run(log):
    pbs_jobid = os.environ.get('PBS_JOBID', 'local_run')
    workspace_dir = os.environ.get('PBS_O_WORKDIR', os.getcwd())
    temp_log = os.environ.get('TEMP_LOG_FILE')
    if not temp_log:
        temp_log = f"/tmp/temp_combined_{pbs_jobid}.log"
    if not os.path.exists(temp_log):
        temp_logs = glob.glob("/tmp/temp_combined_*.log")
        return None, None, None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(workspace_dir, "logs", f"run_{timestamp}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        return None, None, None
    dlcomm_file = os.path.join(output_dir, "terminal_output.log")
    debug_file = os.path.join(output_dir, "debug_output.log")
    dlcomm_count = 0
    debug_count = 0
    try:
        with open(temp_log, 'r') as infile, \
             open(dlcomm_file, 'w') as dlcomm_out, \
             open(debug_file, 'w') as debug_out:
            for line in infile:
                if 'DL_COMM' in line:
                    dlcomm_out.write(line)
                    dlcomm_count += 1
                else:
                    debug_out.write(line)
                    debug_count += 1
        combined_dest = os.path.join(output_dir, "combined_output.log")
        shutil.copy2(temp_log, combined_dest)
        job_info_file = os.path.join(output_dir, "job_info.txt")
        with open(job_info_file, 'w') as f:
            f.write(f"Job ID: {pbs_jobid}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"DL_COMM lines: {dlcomm_count}\n")
            f.write(f"Debug lines: {debug_count}\n")
            f.write(f"Working directory: {workspace_dir}\n")
        latest_link = os.path.join(workspace_dir, "logs", "latest")
        try:
            if os.path.exists(latest_link) or os.path.islink(latest_link):
                os.remove(latest_link)
            os.symlink(output_dir, latest_link)
        except Exception:
            pass
        return debug_file, dlcomm_file, output_dir
    except Exception:
        return None, None, None

def finalize_logs(output_dir, log):
    if not output_dir:
        return
    pbs_jobid = os.environ.get('PBS_JOBID', 'local_run')
    temp_log = os.environ.get('TEMP_LOG_FILE')
    if not temp_log:
        temp_log = f"/tmp/temp_combined_{pbs_jobid}.log"
    dlcomm_file = os.path.join(output_dir, "terminal_output.log")
    debug_file = os.path.join(output_dir, "debug_output.log")
    combined_file = os.path.join(output_dir, "combined_output.log")
    time.sleep(1.0)
    try:
        if os.path.exists(temp_log):
            dlcomm_count = 0
            debug_count = 0
            with open(temp_log, 'r') as infile, \
                 open(dlcomm_file, 'w') as dlcomm_out, \
                 open(debug_file, 'w') as debug_out:
                for line in infile:
                    if 'DL_COMM' in line:
                        dlcomm_out.write(line)
                        dlcomm_count += 1
                    else:
                        debug_out.write(line)
                        debug_count += 1
            shutil.move(temp_log, combined_file)
        else:
            pass
    except Exception:
        pass
