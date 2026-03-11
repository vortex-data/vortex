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
from typing import Any

import numpy as np
import pandas as pd

Z_SCORE_95 = 1.959963984540054
CONTROL_FORMAT = "parquet"


@dataclass
class MedianPolishResult:
    overall: float
    row_effects: pd.Series
    column_effects: pd.Series
    residuals: pd.DataFrame
    converged: bool


def extract_dataset_key(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in df.columns:
        df["dataset_key"] = pd.NA
    else:
        df["dataset_key"] = df["dataset"].apply(
            lambda x: str(sorted(x.items())) if pd.notna(x) and isinstance(x, dict) else pd.NA
        )
    return df


def extract_target_fields(name: str) -> pd.Series:
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
    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        return np.array([], dtype=float)
    samples = np.asarray(values, dtype=float)
    return samples[np.isfinite(samples) & (samples > 0)]


def log_runtime_stats(values: Any) -> dict[str, float]:
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


def robust_scale(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")

    median = np.median(array)
    mad = np.median(np.abs(array - median))
    return float(1.4826 * mad)


def median_polish(table: pd.DataFrame, max_iterations: int = 10, tolerance: float = 1e-8) -> MedianPolishResult | None:
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
    valid = group[[value_column, se_column]].dropna()
    if valid.empty:
        return float("nan")
    return float(np.sqrt(np.square(valid[se_column]).sum()) / len(valid))


def classify_signal(alpha_log_ratio: float, alpha_log_se: float, control_noise_log_std: float, threshold: float) -> str:
    if np.isnan(alpha_log_ratio):
        return "N/A"

    effect_floor = np.log1p(threshold)
    sample_noise = Z_SCORE_95 * alpha_log_se if np.isfinite(alpha_log_se) else 0.0
    systemic_noise = Z_SCORE_95 * control_noise_log_std if np.isfinite(control_noise_log_std) else 0.0
    noise_floor = max(effect_floor, sample_noise, systemic_noise)

    if abs(alpha_log_ratio) < noise_floor:
        return "noise"
    return "regression" if alpha_log_ratio > 0 else "improvement"


def build_statistical_analysis(df: pd.DataFrame, threshold_pct: int) -> dict[str, Any] | None:
    matched = df[
        df["query"].notna()
        & df["engine"].notna()
        & df["file_format"].notna()
        & df["value_base"].notna()
        & df["value_pr"].notna()
    ].copy()

    if matched.empty:
        return None

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
    detail_df["alpha_log_ratio"] = detail_df["log_ratio"] - detail_df["beta_log_ratio"]
    detail_df["alpha_ratio"] = np.exp(detail_df["alpha_log_ratio"])
    detail_df["alpha_log_se"] = np.hypot(detail_df["log_ratio_se"], detail_df["beta_log_se"])
    detail_df["noise_floor_log"] = np.maximum.reduce(
        [
            np.full(len(detail_df), np.log1p(threshold_pct / 100.0)),
            Z_SCORE_95 * np.nan_to_num(detail_df["alpha_log_se"], nan=0.0),
            np.full(len(detail_df), Z_SCORE_95 * systemic_shift_std),
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

    log_ratio_table = detail_df.pivot(index="query", columns="combo", values="log_ratio")
    polish = median_polish(log_ratio_table)
    residual_noise_log_scale = robust_scale(polish.residuals.to_numpy().ravel()) if polish is not None else float("nan")

    return {
        "detail_df": detail_df,
        "query_stats": query_stats,
        "systemic_shift_ratio": float(np.exp(systemic_shift_log_ratio)),
        "systemic_shift_std": systemic_shift_std,
        "median_polish": polish,
        "residual_noise_ratio": float(np.exp(residual_noise_log_scale))
        if np.isfinite(residual_noise_log_scale)
        else float("nan"),
    }


def calculate_geo_mean(df: pd.DataFrame) -> float:
    valid_ratios = [r for r in df["ratio"] if r > 0 and not pd.isna(r)]
    if len(valid_ratios) > 0:
        return math.exp(sum(math.log(r) for r in valid_ratios) / len(valid_ratios))
    return float("nan")


def format_ratio_change(ratio: float) -> str:
    if pd.isna(ratio) or ratio <= 0:
        return "N/A"
    return f"{(ratio - 1.0) * 100:+.1f}%"


def format_performance(
    ratio: float, improvement_threshold: float, regression_threshold: float, target_name: str
) -> str:
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
    if pd.isna(value):
        return ""
    return str(int(value))


def format_name_with_highlight(
    name: str, ratio: float, improvement_threshold: float, regression_threshold: float
) -> str:
    if pd.isna(ratio):
        return name
    if ratio <= improvement_threshold:
        return f"🚀 {name}"
    if ratio >= regression_threshold:
        return f"🚨 {name}"
    return name


def format_signal(signal: str) -> str:
    if signal == "improvement":
        return "✅ faster"
    if signal == "regression":
        return "🚨 regression"
    if signal == "noise":
        return "➖ noise"
    return "N/A"


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
    engine, file_format = group_key
    return (
        ENGINE_ORDER.get(engine, len(ENGINE_ORDER)),
        FILE_FORMAT_ORDER.get(file_format, len(FILE_FORMAT_ORDER)),
        engine,
        file_format,
    )


def main() -> None:
    benchmark_name = sys.argv[3] if len(sys.argv) > 3 else ""

    base = pd.read_json(sys.argv[1], lines=True)
    pr = pd.read_json(sys.argv[2], lines=True)

    base_commit_id = set(base["commit_id"].unique())
    pr_commit_id = set(pr["commit_id"].unique())
    assert len(base_commit_id) == 1, base_commit_id
    assert len(pr_commit_id) == 1, pr_commit_id
    base_commit_id = next(iter(base_commit_id))
    pr_commit_id = next(iter(pr_commit_id))

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

    geo_mean_ratio = calculate_geo_mean(df3)
    vortex_geo_mean_ratio = calculate_geo_mean(vortex_df)
    parquet_geo_mean_ratio = calculate_geo_mean(parquet_df)
    overall_performance = (
        "no data"
        if pd.isna(geo_mean_ratio)
        else format_performance(geo_mean_ratio, improvement_threshold, regression_threshold, "overall")
    )

    summary_lines = [
        "## Summary",
        "",
        f"- **Overall**: {overall_performance}",
    ]
    if len(vortex_df) > 0:
        vortex_performance = format_performance(
            vortex_geo_mean_ratio,
            improvement_threshold,
            regression_threshold,
            "vortex",
        )
        summary_lines.append(
            f"- **Vortex**: {vortex_performance}"
        )
    if len(parquet_df) > 0:
        parquet_performance = format_performance(
            parquet_geo_mean_ratio,
            improvement_threshold,
            regression_threshold,
            "parquet",
        )
        summary_lines.append(
            f"- **Parquet**: {parquet_performance}"
        )

    statistical_analysis = build_statistical_analysis(df3, threshold_pct)
    if statistical_analysis is not None:
        systemic_shift = format_ratio_change(statistical_analysis["systemic_shift_ratio"])
        control_sigma = format_ratio_change(float(np.exp(statistical_analysis["systemic_shift_std"])))
        residual_noise = format_ratio_change(statistical_analysis["residual_noise_ratio"])
        summary_lines.extend(
            [
                "",
                "## Statistical Summary",
                "",
                f"- **Systemic shift ({CONTROL_FORMAT} controls)**: {systemic_shift}",
                f"- **Control sigma**: {control_sigma}",
                f"- **Residual noise**: {residual_noise}",
            ]
        )

        polish = statistical_analysis["median_polish"]
        if polish is not None:
            summary_lines.append(f"- **Median polish overall**: {format_ratio_change(float(np.exp(polish.overall)))}")

    print("\n".join(summary_lines))
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
            print("## Attributed Change")
            print("")
            print(
                alpha_table.to_markdown(
                    index=False,
                    tablefmt="github",
                )
            )
            print("")

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


if __name__ == "__main__":
    main()
