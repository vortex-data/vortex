# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark comparison and analysis."""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TargetRef:
    """Parsed reference to a specific benchmark target."""

    engine: str | None  # None means wildcard
    format: str | None  # None means wildcard
    run: str  # run_id, label, or "latest"

    @classmethod
    def parse(cls, ref: str) -> "TargetRef":
        """
        Parse a reference string.

        Format: engine:format@run
        Examples:
            - duckdb:parquet@latest
            - *:vortex@baseline
            - datafusion:*@2025-01-15T10-30-00_tpch
        """
        # Split on @ to get run reference
        if "@" in ref:
            target_part, run = ref.rsplit("@", 1)
        else:
            target_part = ref
            run = "latest"

        # Split on : to get engine and format
        if ":" in target_part:
            engine, fmt = target_part.split(":", 1)
        else:
            # Could be just engine or just format
            # Default to treating it as a run reference
            return cls(engine=None, format=None, run=ref)

        return cls(
            engine=None if engine == "*" else engine,
            format=None if fmt == "*" else fmt,
            run=run,
        )


def extract_target_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Extract engine and format from target column if present."""
    df = df.copy()
    if "target" in df.columns and len(df) > 0:
        # Handle both dict and string representations
        first_target = df["target"].iloc[0]
        if isinstance(first_target, dict):
            df["engine"] = df["target"].apply(lambda t: t.get("engine", "") if isinstance(t, dict) else "")
            df["format"] = df["target"].apply(lambda t: t.get("format", "") if isinstance(t, dict) else "")

    # Extract query number from name if present
    if "name" in df.columns:
        # Pattern: dataset_qNN/engine:format
        pattern = r"_q(\d+)/"
        df["query"] = df["name"].apply(lambda n: int(m.group(1)) if (m := re.search(pattern, str(n))) else None)

    return df


def geometric_mean(values: pd.Series) -> float:
    """Calculate geometric mean of positive values."""
    valid = values[values > 0].dropna()
    if len(valid) == 0:
        return float("nan")
    return float(np.exp(np.log(valid).mean()))


def filter_by_ref(df: pd.DataFrame, ref: TargetRef) -> pd.DataFrame:
    """Filter DataFrame by a target reference."""
    df = df.copy()

    if ref.engine is not None and "engine" in df.columns:
        df = df[df["engine"] == ref.engine]
    if ref.format is not None and "format" in df.columns:
        df = df[df["format"] == ref.format]

    return df


def compare(
    base_df: pd.DataFrame,
    target_df: pd.DataFrame,
    join_on: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare two DataFrames, computing ratios.

    Returns DataFrame with base_value, target_value, ratio columns.
    """
    base_df = extract_target_fields(base_df)
    target_df = extract_target_fields(target_df)

    if join_on is None:
        join_on = ["query"]

    # Ensure join columns exist
    join_on = [c for c in join_on if c in base_df.columns and c in target_df.columns]

    merged = pd.merge(
        base_df,
        target_df,
        on=join_on,
        how="outer",
        suffixes=("_base", "_target"),
    )

    # Compute ratio (target / base, so < 1 means target is faster)
    if "value_base" in merged.columns and "value_target" in merged.columns:
        merged["ratio"] = merged["value_target"] / merged["value_base"]

    return merged


def summary_stats(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics from a comparison DataFrame.

    Returns dict with geomean, improvement count, regression count, etc.
    """
    ratios = comparison_df["ratio"].dropna()

    improvement_threshold = 0.9  # 10% faster
    regression_threshold = 1.1  # 10% slower

    improvements = ratios[ratios < improvement_threshold]
    regressions = ratios[ratios > regression_threshold]
    neutral = ratios[(ratios >= improvement_threshold) & (ratios <= regression_threshold)]

    # Find best and worst
    best_idx = ratios.idxmin() if len(ratios) > 0 else None
    worst_idx = ratios.idxmax() if len(ratios) > 0 else None

    return {
        "geomean": geometric_mean(ratios),
        "count": len(ratios),
        "improvements": len(improvements),
        "regressions": len(regressions),
        "neutral": len(neutral),
        "best_ratio": ratios.min() if len(ratios) > 0 else float("nan"),
        "worst_ratio": ratios.max() if len(ratios) > 0 else float("nan"),
        "best_name": comparison_df.loc[best_idx, "query"] if best_idx is not None else None,
        "worst_name": comparison_df.loc[worst_idx, "query"] if worst_idx is not None else None,
    }


def find_regressions(comparison_df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Find queries that regressed beyond threshold."""
    return comparison_df[comparison_df["ratio"] > (1.0 + threshold)]


def find_improvements(comparison_df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
    """Find queries that improved beyond threshold."""
    return comparison_df[comparison_df["ratio"] < (1.0 - threshold)]


@dataclass
class PivotComparison:
    """Result of a pivot comparison."""

    df: pd.DataFrame  # Pivoted DataFrame
    baseline: str  # Baseline label (engine:format or run label)
    columns: list[str]  # All column labels (including baseline)


def compare_within_run(
    df: pd.DataFrame,
    baseline_engine: str | None = None,
    baseline_format: str | None = None,
    filter_engine: str | None = None,
    filter_format: str | None = None,
) -> PivotComparison:
    """
    Compare different engine:format combinations within a single run.

    Returns a PivotComparison with one row per query and columns for each engine:format.
    Each cell contains (value, ratio) where ratio is relative to baseline.
    """
    df = extract_target_fields(df)

    if filter_engine is not None and "engine" in df.columns:
        df = df[df["engine"] == filter_engine]
    if filter_format is not None and "format" in df.columns:
        df = df[df["format"] == filter_format]

    # Find unique engine:format combinations
    combos_df = df.groupby(["engine", "format"]).size().reset_index()[["engine", "format"]]
    columns = [f"{row['engine']}:{row['format']}" for _, row in combos_df.iterrows()]

    if len(columns) < 2:
        raise ValueError("Need at least 2 engine:format combinations to compare")

    # Determine baseline
    if baseline_engine is None or baseline_format is None:
        baseline_engine = combos_df.iloc[0]["engine"]
        baseline_format = combos_df.iloc[0]["format"]

    baseline_key = f"{baseline_engine}:{baseline_format}"

    # Create engine:format column
    df["combo"] = df["engine"] + ":" + df["format"]

    # Pivot to get queries as rows, combos as columns
    pivot = df.pivot_table(index="query", columns="combo", values="value", aggfunc="mean")

    # Compute ratios relative to baseline
    if baseline_key in pivot.columns:
        baseline_values = pivot[baseline_key]
        ratio_df = pivot.div(baseline_values, axis=0)
        ratio_df.columns = [f"{c}_ratio" for c in ratio_df.columns]

        # Combine value and ratio columns
        result = pivot.join(ratio_df)
    else:
        result = pivot

    return PivotComparison(df=result.reset_index(), baseline=baseline_key, columns=columns)


def compare_runs(
    run_data: list[tuple[str, pd.DataFrame]],
    baseline_label: str | None = None,
    filter_engine: str | None = None,
    filter_format: str | None = None,
) -> PivotComparison:
    """
    Compare multiple runs.

    Args:
        run_data: List of (label, DataFrame) tuples for each run
        baseline_label: Label of the baseline run (defaults to first)

    Returns a PivotComparison with one row per (query, engine, format) and columns for each run.
    """
    if len(run_data) < 2:
        raise ValueError("Need at least 2 runs to compare")

    labels = [label for label, _ in run_data]
    if baseline_label is None:
        baseline_label = labels[0]

    # Process each run and add run label
    processed: list[pd.DataFrame] = []
    for label, df in run_data:
        df = extract_target_fields(df.copy())
        if filter_engine is not None and "engine" in df.columns:
            df = df[df["engine"] == filter_engine]
        if filter_format is not None and "format" in df.columns:
            df = df[df["format"] == filter_format]
        df["run"] = label

        if not df.empty:
            processed.append(df)

    # Combine all runs
    combined: pd.DataFrame = pd.concat(processed, ignore_index=True)

    # Pivot to get (query, engine, format) as rows, runs as columns
    pivot = combined.pivot_table(index=["query", "engine", "format"], columns="run", values="value", aggfunc="mean")

    # Deduplicate labels while preserving order (two runs can share a label).
    unique_labels = list(dict.fromkeys(labels))

    # Reorder columns to match input order
    pivot = pivot[[label for label in unique_labels if label in pivot.columns]]

    # Compute ratios relative to baseline
    if baseline_label in pivot.columns:
        baseline_values = pivot[baseline_label]
        ratio_df = pivot.div(baseline_values, axis=0)
        ratio_df.columns = [f"{c}_ratio" for c in ratio_df.columns]
        result = pivot.join(ratio_df)
    else:
        result = pivot

    return PivotComparison(df=result.reset_index(), baseline=baseline_label, columns=unique_labels)
