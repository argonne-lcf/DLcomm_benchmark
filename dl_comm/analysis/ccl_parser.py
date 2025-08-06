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
    mapping = {
        'allreduce': 'allreduce',
        'reduce': 'reduce', 
        'broadcast': 'broadcast',
        'allgather': 'allgather',
        'gather': 'gather',
        'scatter': 'scatter',
        'reducescatter': 'reduce_scatter',
        'alltoall': 'alltoall',
        'alltoallsingle': 'alltoall',
        'barrier': 'barrier'
    }
    return mapping.get(collective_name.lower(), collective_name.lower())


def get_readable_table_name(table_name: str) -> str:
    mapping = {
        'main table': 'scale_up table',
        'fallback table': 'scale_up fallback',
        'scaleout table': 'scale_out table'
    }
    return mapping.get(table_name.lower(), table_name)


def parse_nccl_selection(log_path: str, collective_name: str):
    implementations = {}
    current_impl = None
    last_impl = None
    
    algo_map = {
        '0': 'Tree',
        '1': 'Ring', 
        '2': 'CollNetDirect',
        '3': 'CollNetChain',
        '4': 'NVLS',
        '5': 'NVLSTree'
    }
    
    proto_map = {
        '0': 'LL',
        '1': 'LL128', 
        '2': 'Simple'
    }
    
    impl_pattern = re.compile(r'.*\[CONFIG\]\s+Implementation\s+:\s+(.+)')
    job_complete_pattern = re.compile(r'.*\[MPI\]\s+Job\s+complete')
    collective_pattern = re.compile(r'.*NCCL INFO\s+(AllReduce|Reduce|AllGather|ReduceScatter|Broadcast|Scatter|Gather|AllToAll):', re.IGNORECASE)
    selection_pattern = re.compile(r'.*NCCL INFO\s+(\d+)\s+Bytes\s+->\s+Algo\s+(\d+)\s+proto\s+(\d+)')
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            impl_match = impl_pattern.search(line)
            if impl_match:
                current_impl = impl_match.group(1).strip()
                if current_impl not in implementations:
                    implementations[current_impl] = {}
                i += 1
                continue
            
            if job_complete_pattern.search(line) and current_impl:
                last_impl = current_impl
            
            collective_match = collective_pattern.search(line)
            if current_impl and collective_match:
                collective_op = collective_match.group(1)
                for j in range(i + 1, min(i + 15, len(lines))):
                    selection_match = selection_pattern.search(lines[j])
                    if selection_match:
                        buffer_size = int(selection_match.group(1))
                        algo_id = selection_match.group(2)
                        proto_id = selection_match.group(3)
                        
                        algo_name = algo_map.get(algo_id, f'Algorithm{algo_id}')
                        proto_name = proto_map.get(proto_id, f'Protocol{proto_id}')
                        
                        if collective_op not in implementations[current_impl]:
                            implementations[current_impl][collective_op] = {}
                        
                        key = f'{buffer_size} bytes'
                        implementations[current_impl][collective_op][key] = f'{algo_name} (protocol: {proto_name})'
                        break
                    if 'NCCL INFO' in lines[j] and any(op in lines[j] for op in ['AllReduce:', 'Reduce:', 'AllGather:', 'ReduceScatter:', 'Broadcast:']):
                        break
            
            i += 1
        
        if last_impl and last_impl in implementations:
            return {last_impl: implementations[last_impl]}
        else:
            return implementations
                    
    except Exception as e:
        return {}
    
    return implementations


def report_nccl_selection(log_path: str, collective_name: str, logger, scale_up_config=None, scale_out_config=None):
    selections = parse_nccl_selection(log_path, collective_name)
    
    if not selections:
        logger.info(f"[SELECTION] No NCCL algorithm selections found for {collective_name}")
        return
    
    logger.info(f"[SELECTION] NCCL algorithm selections for {collective_name}:")
    
    collective_mapping = {
        'reduce': 'Reduce',
        'allreduce': 'AllReduce', 
        'allgather': 'AllGather',
        'reducescatter': 'ReduceScatter',
        'broadcast': 'Broadcast',
        'scatter': 'Scatter',
        'gather': 'Gather',
        'alltoall': 'AllToAll'
    }
    
    target_collective = collective_mapping.get(collective_name.lower(), collective_name)
    
    found_any = False
    for impl_name, impl_data in selections.items():
        if impl_data and target_collective in impl_data:
            if not found_any:
                found_any = True
            op_selections = impl_data[target_collective]
            filtered_selections = []
            for buffer_size, algorithm in op_selections.items():
                size_bytes = int(buffer_size.split()[0])
                if size_bytes > 1:
                    filtered_selections.append(algorithm)
            
            unique_algorithms = list(set(filtered_selections))
            for algorithm in unique_algorithms:
                user_config = ""
                if not scale_up_config or scale_up_config.strip() == '':
                    user_config = " (user's selection: N/A (default))"
                elif scale_up_config.lower() == 'default':
                    user_config = " (user's selection: default)"
                else:
                    user_config = f" (user's selection: {scale_up_config})"
                
                logger.info(f"[SELECTION] Algorithm: {algorithm}{user_config}")
        elif impl_data:
            logger.info(f"[SELECTION] Parsing the algorithm info failed")
        else:
            logger.info(f"[SELECTION] No selections found")
    
    if not found_any:
        logger.info(f"[SELECTION] No implementations found using {collective_name} collective")


def report_ccl_selection(log_path: str, algo_name: str, logger, scale_up_config=None, scale_out_config=None):
    table_name = get_ccl_table_name(algo_name)
    
    selection = parse_ccl_selection(log_path, table_name)
    if not selection:
        logger.info(f"No '{table_name} selection' block found in {log_path}")
    else:
        logger.info(f"[SELECTION] {table_name} table selection:")
        for tbl, impl in selection.items():
            readable_name = get_readable_table_name(tbl)
            
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
        
        if scale_up_config and scale_up_config != 'default':
            main_selection = selection.get('main table', '')
            if main_selection and main_selection.lower() != scale_up_config.lower():
                logger.info(f"[SELECTION] NOTE: CCL overrode user scale_up algorithm '{scale_up_config}' with '{main_selection}'. Check oneCCL documentation for available algorithms and hardware constraints.")
        
        if scale_out_config and scale_out_config != 'default':
            scaleout_selection = selection.get('scaleout table', '')
            if scaleout_selection and scaleout_selection.lower() != scale_out_config.lower():
                logger.info(f"[SELECTION] NOTE: CCL overrode user scale_out algorithm '{scale_out_config}' with '{scaleout_selection}'. Check oneCCL documentation for available algorithms and hardware constraints.")