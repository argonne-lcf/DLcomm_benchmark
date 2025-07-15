"""
Configuration package for DL-COMM benchmarking.
"""

from .validation import ConfigValidator, parse_buffer_size, adjust_buffer_size_for_group_divisibility
from .system_info import print_system_info

__all__ = ['ConfigValidator', 'parse_buffer_size', 'adjust_buffer_size_for_group_divisibility', 'print_system_info']