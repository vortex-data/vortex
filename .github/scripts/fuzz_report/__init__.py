"""Fuzzer crash reporting utilities."""

from .extract import extract_crash_info
from .dedup import check_duplicate, DedupResult
from .template import render_template

__all__ = [
    "extract_crash_info",
    "check_duplicate",
    "DedupResult",
    "render_template",
]
