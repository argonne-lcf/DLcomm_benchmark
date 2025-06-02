
import os
import subprocess
from pathlib import Path

EXT_ENV = ["--env", "FI_CXI_DEFAULT_CQ_SIZE=1048576"]

def run_mpiexec_and_unitrace(
    python_module: str,
    buf_size_bytes: int,
    trace_dir: Path,
    env_vars: dict,
    np: int,
    ppn: int,
    cpu_bind: str = None
):

 
    trace_dir.mkdir(parents=True, exist_ok=True)


    # building python command
    cmd = ["mpiexec"]
    cmd += EXT_ENV
    cmd += ["--np", str(np), "-ppn", str(ppn), "--cpu-bind", cpu_bind]
    
    output_prefix = str(trace_dir / "trace_rank")
    cmd += [
        "unitrace",
        "--chrome-sycl-logging",
        "--chrome-ccl-logging",
        "--chrome-kernel-logging",
        "--chrome-call-logging",
        "--output", output_prefix,
        "--output-dir-path", str(trace_dir),
    ]
    cmd += [
        "python3",
        "-m", python_module,
        str(buf_size_bytes)
    ]


    #print( f"command is done {cmd}  ") 

    # run the command and write to log file:
    log_file = trace_dir / "run_output.txt"
    with log_file.open("w") as lf:
        subprocess.run(
            cmd,
            env={**os.environ, **env_vars},
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=True
        )
