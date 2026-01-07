# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Comparison module for analyzing benchmark results."""

from .analyzer import (
    PivotComparison,
    TargetRef,
    compare,
    compare_runs,
    compare_within_run,
    extract_target_fields,
    find_improvements,
    find_regressions,
    geometric_mean,
    summary_stats,
)
from .reporter import pivot_comparison_table

__all__ = [
    "PivotComparison",
    "TargetRef",
    "compare",
    "compare_runs",
    "compare_within_run",
    "extract_target_fields",
    "find_improvements",
    "find_regressions",
    "geometric_mean",
    "pivot_comparison_table",
    "summary_stats",
]
