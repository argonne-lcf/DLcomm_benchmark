"""
Communication utilities for DL-COMM benchmarking.
"""

from .comm_setup import setup_communication_groups
from .collectives import COLLECTIVES, OPS_NEED_REDUCE, OP_MAP, DTYPES

__all__ = ['setup_communication_groups', 'COLLECTIVES', 'OPS_NEED_REDUCE', 'OP_MAP', 'DTYPES']