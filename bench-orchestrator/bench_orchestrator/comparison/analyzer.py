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


class BenchmarkAnalyzer:
    """Analyzes and compares benchmark results."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with a DataFrame of results.

        Expected columns: name, value, target (with engine/format), storage, dataset
        """
        self.df = df
        self._extract_target_fields()

    def _extract_target_fields(self) -> None:
        """Extract engine and format from target column if present."""
        if "target" in self.df.columns and len(self.df) > 0:
            # Handle both dict and string representations
            first_target = self.df["target"].iloc[0]
            if isinstance(first_target, dict):
                self.df["engine"] = self.df["target"].apply(
                    lambda t: t.get("engine", "") if isinstance(t, dict) else ""
                )
                self.df["format"] = self.df["target"].apply(
                    lambda t: t.get("format", "") if isinstance(t, dict) else ""
                )
            elif isinstance(first_target, str):
                # Try to parse engine:format from name
                pass

        # Extract query number from name if present
        if "name" in self.df.columns:
            # Pattern: dataset_qNN/engine:format
            pattern = r"_q(\d+)/"
            self.df["query"] = self.df["name"].apply(
                lambda n: int(m.group(1)) if (m := re.search(pattern, str(n))) else None
            )

    @staticmethod
    def geometric_mean(values: pd.Series) -> float:
        """Calculate geometric mean of positive values."""
        valid = values[values > 0].dropna()
        if len(valid) == 0:
            return float("nan")
        return float(np.exp(np.log(valid).mean()))

    def filter_by_ref(self, ref: TargetRef) -> pd.DataFrame:
        """Filter DataFrame by a target reference."""
        df = self.df.copy()

        if ref.engine is not None and "engine" in df.columns:
            df = df[df["engine"] == ref.engine]
        if ref.format is not None and "format" in df.columns:
            df = df[df["format"] == ref.format]

        return df

    def compare(
        self,
        base_df: pd.DataFrame,
        target_df: pd.DataFrame,
        join_on: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare two DataFrames, computing ratios.

        Returns DataFrame with base_value, target_value, ratio columns.
        """
        if join_on is None:
            join_on = ["query"]

        # print(base_df[["name"]])
        # print(target_df[["name"]])

        # Ensure join columns exist
        join_on = [c for c in join_on if c in base_df.columns and c in target_df.columns]

        merged = pd.merge(
            base_df,
            target_df,
            on=join_on,
            how="outer",
            suffixes=("_base", "_target"),
        )

        # print(merged)

        # Compute ratio (target / base, so < 1 means target is faster)
        if "value_base" in merged.columns and "value_target" in merged.columns:
            merged["ratio"] = merged["value_target"] / merged["value_base"]

        return merged

    def compare_runs(
        self,
        base_run_df: pd.DataFrame,
        target_run_df: pd.DataFrame,
        group_by: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compare two runs.

        Args:
            base_run_df: DataFrame from base run
            target_run_df: DataFrame from target run
            group_by: Columns to include in the comparison grouping
        """
        join_on = ["query"]
        if group_by:
            join_on.extend([c for c in group_by if c not in join_on])

        return self.compare(base_run_df, target_run_df, join_on=join_on)

    def summary_stats(self, comparison_df: pd.DataFrame) -> dict[str, Any]:
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
            "geomean": self.geometric_mean(ratios),
            "count": len(ratios),
            "improvements": len(improvements),
            "regressions": len(regressions),
            "neutral": len(neutral),
            "best_ratio": ratios.min() if len(ratios) > 0 else float("nan"),
            "worst_ratio": ratios.max() if len(ratios) > 0 else float("nan"),
            "best_name": comparison_df.loc[best_idx, "query"] if best_idx is not None else None,
            "worst_name": comparison_df.loc[worst_idx, "query"] if worst_idx is not None else None,
        }

    def find_regressions(self, comparison_df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
        """Find queries that regressed beyond threshold."""
        return comparison_df[comparison_df["ratio"] > (1.0 + threshold)]

    def find_improvements(self, comparison_df: pd.DataFrame, threshold: float = 0.10) -> pd.DataFrame:
        """Find queries that improved beyond threshold."""
        return comparison_df[comparison_df["ratio"] < (1.0 - threshold)]
