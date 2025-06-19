"""
Configuration package for DL-COMM benchmarking.
"""

from .validation import ConfigValidator, parse_buffer_size

__all__ = ['ConfigValidator', 'parse_buffer_size']