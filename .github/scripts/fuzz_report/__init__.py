"""Fuzzer crash reporting utilities."""

from .dedup import DedupResult, check_duplicate
from .extract import extract_crash_info
from .template import render_template

__all__ = [
    "extract_crash_info",
    "check_duplicate",
    "DedupResult",
    "render_template",
]
