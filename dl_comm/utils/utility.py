 
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