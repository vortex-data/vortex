# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import re
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

# Analysis overview:
# - Join base and PR benchmark rows on benchmark identity.
# - Use log-ratios because benchmark slowdowns/speedups are multiplicative.
# - Treat parquet rows as controls to estimate systemic drift beta(q).
# - Attribute the remaining change to the PR as alpha(q, c).
# - Call a row significant only when alpha clears a conservative noise floor.
# - Collapse those row-level results into a short verdict for the PR comment.
#
# Concretely:
# - raw ratio = median_runtime_pr / median_runtime_base
# - log_ratio = log(raw ratio)
# - beta(q) = mean(log_ratio) across parquet control rows for query q
# - alpha(q, c) = log_ratio(q, c) - beta(q)
# - attributed impact = geometric mean of alpha ratios across non-control rows

# Benchmarks are noisier than textbook measurement data, so use a conservative
# cutoff that is closer to a 99% two-sided interval before calling a change real.
Z_SCORE_99 = 2.5758293035489004
CONTROL_FORMAT = "parquet"
FILE_SIZE_METRIC = "file_size"


@dataclass
class MedianPolishResult:
    """Robust additive decomposition for the query x config log-ratio matrix."""

    overall: float
    row_effects: pd.Series
    column_effects: pd.Series
    residuals: pd.DataFrame
    converged: bool


def extract_dataset_key(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataset metadata into a stable join key."""

    if "dataset" not in df.columns:
        df["dataset_key"] = pd.NA
    else:
        df["dataset_key"] = df["dataset"].apply(
            lambda x: str(sorted(x.items())) if pd.notna(x) and isinstance(x, dict) else pd.NA
        )
    return df


def split_file_size_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split shared-stream file-size rows from benchmark timing rows."""

    if df.empty:
        return df.copy(), df.copy()

    metric = df["metric"] if "metric" in df.columns else pd.Series(pd.NA, index=df.index)
    file_size = df["file_size"] if "file_size" in df.columns else pd.Series(pd.NA, index=df.index)
    mask = metric.eq(FILE_SIZE_METRIC) | file_size.notna()
    return df[mask].copy(), df[~mask].copy()


def extract_target_fields(name: str) -> pd.Series:
    """Parse query, engine, and format from the benchmark name."""

    if not isinstance(name, str):
        return pd.Series({"engine": "unknown", "file_format": "unknown", "query": pd.NA})

    match = re.search(r"_q(\d+)/([^:]+):(.+)$", name)
    if match is None:
        return pd.Series({"engine": "unknown", "file_format": "unknown", "query": pd.NA})

    return pd.Series(
        {
            "engine": match.group(2),
            "file_format": match.group(3),
            "query": int(match.group(1)),
        }
    )


def positive_samples(values: Any) -> np.ndarray:
    """Keep only finite, strictly positive runtime samples."""

    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        return np.array([], dtype=float)
    samples = np.asarray(values, dtype=float)
    return samples[np.isfinite(samples) & (samples > 0)]


def log_runtime_stats(values: Any) -> dict[str, float]:
    """Summarize repeated runtimes on the log scale."""

    samples = positive_samples(values)
    if samples.size == 0:
        return {
            "sample_count": 0,
            "log_mean": float("nan"),
            "log_std": float("nan"),
            "log_se": float("nan"),
        }

    logs = np.log(samples)
    log_std = float(np.std(logs, ddof=1)) if logs.size > 1 else 0.0
    return {
        "sample_count": int(logs.size),
        "log_mean": float(logs.mean()),
        "log_std": log_std,
        "log_se": float(log_std / np.sqrt(logs.size)),
    }


def ratio_stats(
    base_values: Any,
    pr_values: Any,
    base_median: float,
    pr_median: float,
) -> dict[str, float]:
    """Compute the PR/base effect and its sampling error for one matched row."""

    base_stats = log_runtime_stats(base_values)
    pr_stats = log_runtime_stats(pr_values)

    if not np.isfinite(base_median) or not np.isfinite(pr_median) or base_median <= 0 or pr_median <= 0:
        return {
            "ratio": float("nan"),
            "log_ratio": float("nan"),
            "log_ratio_se": float("nan"),
            **{f"base_{key}": value for key, value in base_stats.items()},
            **{f"pr_{key}": value for key, value in pr_stats.items()},
        }

    ratio = pr_median / base_median
    return {
        "ratio": ratio,
        "log_ratio": float(np.log(ratio)),
        "log_ratio_se": float(np.hypot(base_stats["log_se"], pr_stats["log_se"])),
        **{f"base_{key}": value for key, value in base_stats.items()},
        **{f"pr_{key}": value for key, value in pr_stats.items()},
    }


def median_polish(table: pd.DataFrame, max_iterations: int = 10, tolerance: float = 1e-8) -> MedianPolishResult | None:
    """Estimate row and column effects for the log-ratio matrix."""

    working = table.copy().astype(float)
    working = working.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if working.shape[0] < 2 or working.shape[1] < 2:
        return None

    row_effects = pd.Series(0.0, index=working.index, dtype=float)
    column_effects = pd.Series(0.0, index=working.columns, dtype=float)
    overall = 0.0
    converged = False

    for _ in range(max_iterations):
        row_medians = working.median(axis=1)
        working = working.sub(row_medians, axis=0)
        row_effects = row_effects.add(row_medians, fill_value=0.0)
        row_shift = float(row_effects.median())
        row_effects -= row_shift
        overall += row_shift

        column_medians = working.median(axis=0)
        working = working.sub(column_medians, axis=1)
        column_effects = column_effects.add(column_medians, fill_value=0.0)
        column_shift = float(column_effects.median())
        column_effects -= column_shift
        overall += column_shift

        largest_adjustment = max(
            float(row_medians.abs().max()) if not row_medians.empty else 0.0,
            float(column_medians.abs().max()) if not column_medians.empty else 0.0,
        )
        if largest_adjustment <= tolerance:
            converged = True
            break

    return MedianPolishResult(
        overall=float(overall),
        row_effects=row_effects,
        column_effects=column_effects,
        residuals=working,
        converged=converged,
    )


def mean_with_standard_error(group: pd.DataFrame, value_column: str, se_column: str) -> float:
    """Approximate the standard error of a group mean from row-level errors."""

    valid = group[[value_column, se_column]].dropna()
    if valid.empty:
        return float("nan")
    return float(np.sqrt(np.square(valid[se_column]).sum()) / len(valid))


def classify_signal(alpha_log_ratio: float, alpha_log_se: float, control_noise_log_std: float, threshold: float) -> str:
    """Label an attributed change as real or noise using a conservative floor."""

    if np.isnan(alpha_log_ratio):
        return "N/A"

    effect_floor = np.log1p(threshold)
    sample_noise = Z_SCORE_99 * alpha_log_se if np.isfinite(alpha_log_se) else 0.0
    systemic_noise = Z_SCORE_99 * control_noise_log_std if np.isfinite(control_noise_log_std) else 0.0
    noise_floor = max(effect_floor, sample_noise, systemic_noise)

    if abs(alpha_log_ratio) < noise_floor:
        return "noise"
    return "regression" if alpha_log_ratio > 0 else "improvement"


def build_statistical_analysis(df: pd.DataFrame, threshold_pct: int) -> dict[str, Any] | None:
    """Build the full alpha/beta attribution model for the markdown report."""

    matched = df[
        df["query"].notna()
        & df["engine"].notna()
        & df["file_format"].notna()
        & df["value_base"].notna()
        & df["value_pr"].notna()
    ].copy()

    if matched.empty:
        return None

    # One row here is one query/config benchmark matched between base and PR.
    rows: list[dict[str, Any]] = []
    for _, row in matched.iterrows():
        stats = ratio_stats(
            row.get("all_runtimes_base"),
            row.get("all_runtimes_pr"),
            float(row["value_base"]),
            float(row["value_pr"]),
        )
        rows.append(
            {
                "name": row["name"],
                "query": int(row["query"]),
                "engine": row["engine"],
                "file_format": row["file_format"],
                "combo": f"{row['engine']}:{row['file_format']}",
                "is_control": row["file_format"] == CONTROL_FORMAT,
                **stats,
            }
        )

    detail_df = pd.DataFrame(rows).sort_values(["query", "engine", "file_format"]).reset_index(drop=True)
    controls = detail_df[detail_df["is_control"] & detail_df["log_ratio"].notna()]
    if controls.empty:
        return None

    # beta(q): systemic drift inferred from parquet controls for query q.
    query_rows: list[dict[str, Any]] = []
    for query, group in controls.groupby("query", sort=True):
        beta_log_ratio = float(group["log_ratio"].mean())
        query_rows.append(
            {
                "query": int(query),
                "beta_log_ratio": beta_log_ratio,
                "beta_ratio": float(np.exp(beta_log_ratio)),
                "beta_log_se": mean_with_standard_error(group, "log_ratio", "log_ratio_se"),
                "beta_log_std": float(group["log_ratio"].std(ddof=1)) if len(group) > 1 else 0.0,
                "control_count": int(len(group)),
            }
        )

    query_stats = pd.DataFrame(query_rows)
    detail_df = detail_df.merge(query_stats, on="query", how="left")

    systemic_shift_log_ratio = float(query_stats["beta_log_ratio"].mean())
    systemic_shift_std = float(query_stats["beta_log_ratio"].std(ddof=1)) if len(query_stats) > 1 else 0.0
    # alpha(q, c): PR-attributable effect after subtracting the control drift.
    detail_df["alpha_log_ratio"] = detail_df["log_ratio"] - detail_df["beta_log_ratio"]
    detail_df["alpha_ratio"] = np.exp(detail_df["alpha_log_ratio"])
    detail_df["alpha_log_se"] = np.hypot(detail_df["log_ratio_se"], detail_df["beta_log_se"])
    # Noise floor = max(user threshold, sampling error, control drift variability).
    detail_df["noise_floor_log"] = np.maximum.reduce(
        [
            np.full(len(detail_df), np.log1p(threshold_pct / 100.0)),
            Z_SCORE_99 * np.nan_to_num(detail_df["alpha_log_se"], nan=0.0),
            np.full(len(detail_df), Z_SCORE_99 * systemic_shift_std),
        ]
    )
    detail_df["noise_floor_ratio"] = np.exp(detail_df["noise_floor_log"])
    detail_df["signal"] = detail_df.apply(
        lambda row: classify_signal(
            row["alpha_log_ratio"],
            row["alpha_log_se"],
            systemic_shift_std,
            threshold_pct / 100.0,
        ),
        axis=1,
    )

    # Median polish gives a robust overall shift estimate.
    log_ratio_table = detail_df.pivot(index="query", columns="combo", values="log_ratio")
    polish = median_polish(log_ratio_table)

    return {
        "detail_df": detail_df,
        "query_stats": query_stats,
        "systemic_shift_ratio": float(np.exp(systemic_shift_log_ratio)),
        "systemic_shift_std": systemic_shift_std,
        "median_polish": polish,
    }


def calculate_geo_mean(df: pd.DataFrame) -> float:
    """Geometric mean of positive ratios from a DataFrame ratio column."""

    valid_ratios = [r for r in df["ratio"] if r > 0 and not pd.isna(r)]
    if len(valid_ratios) > 0:
        return math.exp(sum(math.log(r) for r in valid_ratios) / len(valid_ratios))
    return float("nan")


def geometric_mean_from_values(values: pd.Series) -> float:
    """Geometric mean of a ratio series."""

    valid_values = values[(values > 0) & values.notna()]
    if len(valid_values) == 0:
        return float("nan")
    return float(np.exp(np.log(valid_values).mean()))


def format_ratio_change(ratio: float) -> str:
    """Render a ratio as a signed percent delta."""

    if pd.isna(ratio) or ratio <= 0:
        return "N/A"
    return f"{(ratio - 1.0) * 100:+.1f}%"


def format_performance(
    ratio: float, improvement_threshold: float, regression_threshold: float, target_name: str
) -> str:
    """Render a geomean ratio with a coarse emoji summary."""

    if pd.isna(ratio):
        return f"no {target_name.lower()} data"

    if improvement_threshold <= ratio <= regression_threshold:
        emoji = "➖"
    elif ratio < 1:
        emoji = "✅"
    else:
        emoji = "❌"
    return f"{ratio:.3f}x {emoji}"


def format_integer_value(value: float) -> str:
    """Render numeric timing values for markdown tables."""

    if pd.isna(value):
        return ""
    return str(int(value))


def format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable size."""

    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    if size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def format_size_change(change_bytes: int) -> str:
    """Format a byte change with a sign."""

    sign = "+" if change_bytes > 0 else ""
    return f"{sign}{format_size(abs(change_bytes))}"


def format_pct_change(pct: float) -> str:
    """Format a percentage change with a sign."""

    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def extract_file_size_data(df: pd.DataFrame) -> dict[tuple[str, str, str, str], int]:
    """Extract file-size rows keyed by benchmark, scale factor, format, and file."""

    data = {}
    if df.empty:
        return data

    for _, row in df.iterrows():
        metadata = row.get("file_size")
        if not isinstance(metadata, dict):
            continue

        key = (
            str(metadata.get("benchmark", "")),
            str(metadata.get("scale_factor", "1.0")),
            str(metadata.get("format", "")),
            str(metadata.get("file", "")),
        )
        value = row.get("value")
        if pd.isna(value):
            continue
        data[key] = int(value)

    return data


def format_file_size_report(base_rows: pd.DataFrame, pr_rows: pd.DataFrame) -> str:
    """Render a shared-comment file-size comparison report."""

    pr_data = extract_file_size_data(pr_rows)
    if not pr_data:
        return ""

    base_data = extract_file_size_data(base_rows)
    pr_scopes = {(benchmark, scale_factor) for benchmark, scale_factor, _file_format, _file_name in pr_data}
    base_data = {key: value for key, value in base_data.items() if key[:2] in pr_scopes}
    if not base_data:
        return "_No baseline file sizes found for base commit._"

    comparisons = []
    format_totals: dict[str, dict[str, int]] = {}

    for key in sorted(set(base_data) | set(pr_data)):
        _benchmark, scale_factor, file_format, file_name = key
        base_size = base_data.get(key, 0)
        pr_size = pr_data.get(key, 0)

        totals = format_totals.setdefault(file_format, {"base": 0, "pr": 0})
        totals["base"] += base_size
        totals["pr"] += pr_size

        change = pr_size - base_size
        if change == 0:
            continue

        if base_size > 0:
            pct_change = (pr_size / base_size - 1) * 100
        elif pr_size > 0:
            pct_change = float("inf")
        else:
            pct_change = 0.0

        comparisons.append(
            {
                "file": file_name,
                "scale_factor": scale_factor,
                "format": file_format,
                "base_size": base_size,
                "pr_size": pr_size,
                "change": change,
                "pct_change": pct_change,
            }
        )

    if not comparisons:
        return "_No file size changes detected._"

    comparisons.sort(key=lambda comparison: comparison["pct_change"], reverse=True)

    total_base = sum(totals["base"] for totals in format_totals.values())
    total_pr = sum(totals["pr"] for totals in format_totals.values())
    overall_pct_str = "new" if total_base == 0 else format_pct_change((total_pr / total_base - 1) * 100)
    increases = sum(1 for comparison in comparisons if comparison["change"] > 0)
    decreases = sum(1 for comparison in comparisons if comparison["change"] < 0)

    output = StringIO()
    print("<details>", file=output)
    print(
        f"<summary>File Size Changes ({len(comparisons)} files changed, "
        f"{overall_pct_str} overall, {increases}↑ {decreases}↓)</summary>",
        file=output,
    )
    print("", file=output)
    print("<br>", file=output)
    print("", file=output)
    print("| File | Scale | Format | Base | HEAD | Change | % |", file=output)
    print("|------|-------|--------|------|------|--------|---|", file=output)

    for comparison in comparisons:
        pct_str = "new" if comparison["pct_change"] == float("inf") else format_pct_change(comparison["pct_change"])
        base_str = format_size(comparison["base_size"]) if comparison["base_size"] > 0 else "-"
        print(
            f"| {comparison['file']} | {comparison['scale_factor']} | {comparison['format']} | {base_str} | "
            f"{format_size(comparison['pr_size'])} | {format_size_change(comparison['change'])} | {pct_str} |",
            file=output,
        )

    print("", file=output)
    print("**Totals:**", file=output)
    for file_format in sorted(format_totals):
        totals = format_totals[file_format]
        base_total = totals["base"]
        pr_total = totals["pr"]
        pct_str = "" if base_total == 0 else f" ({format_pct_change((pr_total / base_total - 1) * 100)})"
        print(f"- {file_format}: {format_size(base_total)} → {format_size(pr_total)}{pct_str}", file=output)

    print("", file=output)
    print("</details>", file=output)
    return output.getvalue().rstrip()


def format_name_with_highlight(
    name: str, ratio: float, improvement_threshold: float, regression_threshold: float
) -> str:
    """Highlight clearly large raw changes in the detailed per-config tables."""

    if pd.isna(ratio):
        return name
    if ratio <= improvement_threshold:
        return f"{name} 🚀"
    if ratio >= regression_threshold:
        return f"{name} 🚨"
    return name


def format_signal(signal: str) -> str:
    """Render the attributed-change label for markdown output."""

    if signal == "improvement":
        return "✅ faster"
    if signal == "regression":
        return "🚨 regression"
    if signal == "noise":
        return "➖ noise"
    return "N/A"


def build_verdict(statistical_analysis: dict[str, Any]) -> dict[str, str] | None:
    """Collapse row-level attribution into a short PR-comment headline."""

    alpha_rows = statistical_analysis["detail_df"][~statistical_analysis["detail_df"]["is_control"]].copy()
    if alpha_rows.empty:
        return None

    # Attributed impact is the geometric mean of non-control alpha ratios.
    attributed_impact_ratio = geometric_mean_from_values(alpha_rows["alpha_ratio"])
    if pd.isna(attributed_impact_ratio):
        return None

    # Confidence depends on directional consistency, share above the noise floor,
    # and whether the controls themselves look unusually noisy.
    signs = alpha_rows["alpha_log_ratio"].dropna()
    consistent_sign_share = 0.0
    if not signs.empty:
        positive_share = float((signs > 0).mean())
        negative_share = float((signs < 0).mean())
        consistent_sign_share = max(positive_share, negative_share)

    significant_share = float((alpha_rows["signal"] != "noise").mean())
    evidence_share = min(consistent_sign_share, significant_share)

    control_sigma = float(np.exp(statistical_analysis["systemic_shift_std"]))
    aggregate_noise_floor = max(
        1.0 + 1e-9,
        control_sigma,
        geometric_mean_from_values(alpha_rows["noise_floor_ratio"]),
    )

    if (
        not np.isfinite(aggregate_noise_floor)
        or attributed_impact_ratio < aggregate_noise_floor
        and (1.0 / attributed_impact_ratio if attributed_impact_ratio > 0 else float("inf")) < aggregate_noise_floor
    ):
        status = "No clear signal"
    elif attributed_impact_ratio > 1.0:
        status = "Likely regression"
    else:
        status = "Likely improvement"

    if control_sigma > 1.05:
        confidence = "environment too noisy"
    elif evidence_share >= 0.7:
        confidence = "high"
    elif evidence_share >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "status": status,
        "impact": format_ratio_change(attributed_impact_ratio),
        "confidence": confidence,
        "environment_shift": format_ratio_change(statistical_analysis["systemic_shift_ratio"]),
    }


def build_within_engine_statistical_analyses(df: pd.DataFrame, threshold_pct: int) -> dict[str, dict[str, Any]]:
    """Build an attribution model per engine, using that engine's own parquet rows as controls."""

    analyses = {}
    matched = df[df["engine"].notna() & (df["engine"] != "unknown")]
    for engine, engine_df in matched.groupby("engine", sort=False):
        if engine_df["file_format"].eq(CONTROL_FORMAT).sum() == 0:
            continue
        if (~engine_df["file_format"].eq(CONTROL_FORMAT)).sum() == 0:
            continue
        analysis = build_statistical_analysis(engine_df.copy(), threshold_pct)
        if analysis is not None:
            analyses[str(engine)] = analysis
    return analyses


def format_within_engine_summary(analyses: dict[str, dict[str, Any]]) -> str | None:
    """Render a compact summary of per-engine attributed changes."""

    summaries = []
    for engine in sorted(analyses, key=lambda value: (ENGINE_ORDER.get(value, len(ENGINE_ORDER)), value)):
        verdict = build_verdict(analyses[engine])
        if verdict is None:
            continue
        display_name = {
            "datafusion": "DataFusion",
            "duckdb": "DuckDB",
        }.get(engine, engine)
        summaries.append(
            f"{display_name} {verdict['status']} ({verdict['impact']}, {verdict['confidence']} confidence)"
        )

    if not summaries:
        return None
    return " · ".join(summaries)


def format_report_help() -> str:
    """Render explanatory markdown for the benchmark report headline fields."""

    return "\n".join(
        [
            "<details>",
            "<summary>How to read Verdict and Engines</summary>",
            "",
            "<br>",
            "",
            "- **Verdict**: Overall PR-level signal after subtracting baseline drift "
            "estimated from Parquet control rows. It can be `Likely improvement`, "
            "`Likely regression`, or `No clear signal`.",
            "- **Engines**: Per-engine attribution. DataFusion is compared against "
            "DataFusion/Parquet controls; DuckDB is compared against DuckDB/Parquet "
            "controls. This answers whether each engine improved or regressed independently.",
            "- **Confidence**: Based on directional consistency, share of rows above "
            "the noise floor, and control-run noise.",
            "",
            "</details>",
        ]
    )


ENGINE_ORDER = {
    "vortex": 0,
    "datafusion": 1,
    "duckdb": 2,
    "lance": 3,
    "arrow": 4,
}

FILE_FORMAT_ORDER = {
    "vortex-file-compressed": 0,
    "vortex-compact": 1,
    "parquet": 2,
    "lance": 3,
    "duckdb": 4,
    "arrow": 5,
}


def group_sort_key(group_key: tuple[str, str]) -> tuple[int, int, str, str]:
    """Keep output ordering stable and grouped by likely reader interest."""

    engine, file_format = group_key
    return (
        ENGINE_ORDER.get(engine, len(ENGINE_ORDER)),
        FILE_FORMAT_ORDER.get(file_format, len(FILE_FORMAT_ORDER)),
        engine,
        file_format,
    )


def main() -> None:
    """Render the benchmark comparison markdown used in CI PR comments."""

    benchmark_name = sys.argv[3] if len(sys.argv) > 3 else ""

    base = pd.read_json(sys.argv[1], lines=True)
    pr = pd.read_json(sys.argv[2], lines=True)

    base_commit_id = set(base["commit_id"].unique())
    pr_commit_id = set(pr["commit_id"].unique())
    assert len(base_commit_id) == 1, base_commit_id
    assert len(pr_commit_id) == 1, pr_commit_id
    base_commit_id = next(iter(base_commit_id))
    pr_commit_id = next(iter(pr_commit_id))

    base_file_sizes, base = split_file_size_rows(base)
    pr_file_sizes, pr = split_file_size_rows(pr)

    if "storage" not in base:
        base["storage"] = pd.NA
    if "storage" not in pr:
        pr["storage"] = pd.NA

    base = extract_dataset_key(base)
    pr = extract_dataset_key(pr)

    df3 = pd.merge(base, pr, on=["name", "storage", "dataset_key"], how="right", suffixes=("_base", "_pr"))
    df3["ratio"] = df3["value_pr"] / df3["value_base"]
    df3[["engine", "file_format", "query"]] = df3["name"].apply(extract_target_fields)

    is_s3_benchmark = "s3" in benchmark_name.lower()
    threshold_pct = 30 if is_s3_benchmark else 10
    improvement_threshold = 1.0 - (threshold_pct / 100.0)
    regression_threshold = 1.0 + (threshold_pct / 100.0)

    vortex_df = df3[df3["name"].str.contains("vortex", case=False, na=False)]
    parquet_df = df3[df3["name"].str.contains("parquet", case=False, na=False)]

    vortex_geo_mean_ratio = calculate_geo_mean(vortex_df)
    parquet_geo_mean_ratio = calculate_geo_mean(parquet_df)

    statistical_analysis = build_statistical_analysis(df3, threshold_pct)
    verdict = build_verdict(statistical_analysis) if statistical_analysis is not None else None
    engine_analyses = build_within_engine_statistical_analyses(df3, threshold_pct)
    engine_summary = format_within_engine_summary(engine_analyses)

    summary_fields: list[str] = []

    if verdict is not None:
        summary_fields.append(f"**Verdict**: {verdict['status']} ({verdict['confidence']} confidence)")
        summary_fields.append(f"**Attributed Vortex impact**: {verdict['impact']}")
    if engine_summary is not None:
        summary_fields.append(f"**Engines**: {engine_summary}")

    if len(vortex_df) > 0:
        vortex_performance = format_performance(
            vortex_geo_mean_ratio,
            improvement_threshold,
            regression_threshold,
            "vortex",
        )
        summary_fields.append(f"**Vortex (geomean)**: {vortex_performance}")
    if len(parquet_df) > 0:
        parquet_performance = format_performance(
            parquet_geo_mean_ratio,
            improvement_threshold,
            regression_threshold,
            "parquet",
        )
        summary_fields.append(f"**Parquet (geomean)**: {parquet_performance}")

    if verdict is not None:
        shifts = f"Parquet (control) {verdict['environment_shift']}"
        if statistical_analysis is not None:
            polish = statistical_analysis["median_polish"]
            if polish is not None:
                shifts += f" · Median polish {format_ratio_change(float(np.exp(polish.overall)))}"
        summary_fields.append(f"**Shifts**: {shifts}")

    print("<br>".join(summary_fields))
    print("")
    print(format_report_help())
    print("")
    print("---")
    print("")

    if statistical_analysis is not None:
        alpha_rows = statistical_analysis["detail_df"][~statistical_analysis["detail_df"]["is_control"]].copy()
        if not alpha_rows.empty:
            alpha_rows = alpha_rows.sort_values(["query", "engine", "file_format"])
            alpha_table = pd.DataFrame(
                {
                    "Query": alpha_rows["query"].astype(int),
                    "Config": alpha_rows["combo"],
                    "Raw Δ": alpha_rows["ratio"].map(format_ratio_change),
                    "Control Δ": alpha_rows["beta_ratio"].map(format_ratio_change),
                    "Attributed α": alpha_rows["alpha_ratio"].map(format_ratio_change),
                    "Noise floor": alpha_rows["noise_floor_ratio"].map(format_ratio_change),
                    "Significant?": alpha_rows["signal"].map(format_signal),
                }
            )

    grouped_tables = df3.groupby(["engine", "file_format"], dropna=False, sort=False)
    for engine, file_format in sorted(grouped_tables.groups.keys(), key=group_sort_key):
        group_df = grouped_tables.get_group((engine, file_format)).sort_values("name")
        group_performance = format_performance(
            calculate_geo_mean(group_df),
            improvement_threshold,
            regression_threshold,
            "group",
        )
        significant_improvements = (group_df["ratio"] < improvement_threshold).sum()
        significant_regressions = (group_df["ratio"] > regression_threshold).sum()
        unit = group_df["unit_base"].dropna().iloc[0] if group_df["unit_base"].notna().any() else "unit"
        display_df = pd.DataFrame(
            {
                "name": [
                    format_name_with_highlight(name, ratio, improvement_threshold, regression_threshold)
                    for name, ratio in zip(group_df["name"], group_df["ratio"])
                ],
                f"PR {pr_commit_id[:8]} ({unit})": group_df["value_pr"].map(format_integer_value),
                f"base {base_commit_id[:8]} ({unit})": group_df["value_base"].map(format_integer_value),
                "ratio (PR/base)": group_df["ratio"],
            }
        )
        print("<details>")
        summary_text = (
            f"{engine} / {file_format} ({group_performance}, {significant_improvements}↑ {significant_regressions}↓)"
        )
        print(f"<summary>{summary_text}</summary>")
        print("")
        print("<br>")
        print("")
        print(
            display_df.to_markdown(
                index=False,
                tablefmt="github",
                floatfmt=".2f",
            )
        )
        print("")
        print("</details>")

    file_size_report = format_file_size_report(base_file_sizes, pr_file_sizes)
    if file_size_report:
        print("")
        print("---")
        print("")
        print(file_size_report)

    if statistical_analysis is not None and not alpha_rows.empty:
        print("<details>")
        print("<summary>Full attributed analysis</summary>")
        print("")
        print("<br>")
        print("")
        print(
            alpha_table.to_markdown(
                index=False,
                tablefmt="github",
            )
        )
        print("")
        print("</details>")


if __name__ == "__main__":
    main()
