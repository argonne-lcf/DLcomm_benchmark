import os
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum

 
LOG_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

 
OUTPUT_LEVEL = 35
logging.addLevelName(OUTPUT_LEVEL, "OUTPUT")

def output(self, message, *args, **kwargs):
    if self.isEnabledFor(OUTPUT_LEVEL):
        self._log(OUTPUT_LEVEL, message, args, **kwargs)

logging.Logger.output = output


class DLCOMMLogger:
     
    __instance = None

    def __init__(self, log_file="dlcomm.log", log_dir=None):
        if DLCOMMLogger.__instance is not None:
            raise Exception(f"Class {self.__class__.__name__} is a singleton!")
        
      
        if log_dir is None:
            log_dir = os.getcwd()
        
        log_path = Path(log_dir) / log_file
      
        self.logger = logging.getLogger("DL_COMM")
        self.logger.setLevel(logging.DEBUG)
        
        
        self.logger.handlers = []
        
         
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
 

        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        
         
        self.logger.addHandler(file_handler)
        
 
        
        DLCOMMLogger.__instance = self
    
    @staticmethod
    def get_instance(log_file="dlcomm.log", log_dir=None):
        if DLCOMMLogger.__instance is None:
            DLCOMMLogger(log_file=log_file, log_dir=log_dir)
        return DLCOMMLogger.__instance.logger
    
    @staticmethod
    def reset():
        if DLCOMMLogger.__instance is not None:
             
            for handler in DLCOMMLogger.__instance.logger.handlers:
                handler.close()
            DLCOMMLogger.__instance = None
    
    @staticmethod
    def flush():
        """Force flush all handlers"""
        if DLCOMMLogger.__instance is not None:
            for handler in DLCOMMLogger.__instance.logger.handlers:
                handler.flush()


class Profile:
    """Dummy profiler class for compatibility"""
    def __init__(self, cat, name=None, epoch=None, step=None, image_idx=None, image_size=None):
        return 
    
    def log(self, func):
        return func
    
    def log_init(self, func):
        return func
    
    def iter(self, func, iter_name="step"):
        return func
    
    def __enter__(self):
        return
    
    def __exit__(self, type, value, traceback):
        return
    
    def update(self, epoch=None, step=None, image_idx=None, image_size=None, args={}):
        return
    
    def flush(self):
        return
    
    def reset(self):
        return
    
    def log_static(self, func):
        return func


# Helper functions that might be used elsewhere in your code
def utcnow(format=LOG_TS_FORMAT):
    """Get current UTC timestamp as formatted string"""
    return datetime.now().strftime(format)


def dummy_mxm_compute(device, dtype=None, size=None, framework="pytorch"):
    if framework == 'pytorch':
        import torch
        import time
        
        if device.type == 'cuda':
            device_name = torch.cuda.get_device_name(device.index)
            device_id = device.index
        elif device.type == 'xpu':
            device_name = f"XPU Device {device.index}"
            device_id = device.index
        else:
            device_name = "CPU"
            device_id = 0
        
        A = torch.randn(size, size, dtype=dtype, device=device)
        B = torch.randn(size, size, dtype=dtype, device=device)
        
        start_time = time.perf_counter()
        C = torch.matmul(A, B)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'xpu':
            torch.xpu.synchronize()
        end_time = time.perf_counter()
        
        time_ms = (end_time - start_time) * 1000
        total_flops = 2 * (size ** 3)
        gflops = total_flops / 1e9
        tflops_throughput = (total_flops / 1e12) / ((end_time - start_time))
        
        del A, B, C
        
        return {
            'time_ms': time_ms,
            'gflops': gflops,
            'tflops_throughput': tflops_throughput,
            'size': size,
            'device_name': device_name,
            'device_id': device_id,
            'device_type': device.type
        }
    elif framework == 'jax':
        return None