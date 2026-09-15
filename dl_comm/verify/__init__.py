"""Verification support for DLcomm: rank-dependent payloads and failure accounting."""

from .payload import (
    build_payload,
    scatter_source,
    position_term,
    rank_signature,
    expected_reduction,
    choose_moduli,
    tolerance_for,
    is_exact_dtype,
    DEFAULT_RANK_MODULUS,
    DEFAULT_POSITION_MODULUS,
    PROD_CONTRIBUTORS,
    SCATTER_OFFSET,
)
from . import failures

__all__ = [
    "build_payload",
    "scatter_source",
    "position_term",
    "rank_signature",
    "expected_reduction",
    "choose_moduli",
    "tolerance_for",
    "is_exact_dtype",
    "DEFAULT_RANK_MODULUS",
    "DEFAULT_POSITION_MODULUS",
    "PROD_CONTRIBUTORS",
    "SCATTER_OFFSET",
    "failures",
]
