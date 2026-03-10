# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import math
import sys

import pandas as pd

S3_THRESHOLD_PCT = 30
DEFAULT_THRESHOLD_PCT = 10
NOISE_LOW_MAX_CV_PCT = 5.0
NOISE_MEDIUM_MAX_CV_PCT = 15.0
DRIFT_NOTICEABLE_THRESHOLD_PCT = 5.0
DRIFT_CONSISTENCY_THRESHOLD_PCT = 80.0
DRIFT_RESIDUAL_SPREAD_THRESHOLD_PCT = 5.0

# Check if benchmark name argument is provided (will be added from workflow)
benchmark_name = sys.argv[3] if len(sys.argv) > 3 else ""

base = pd.read_json(sys.argv[1], lines=True)
pr = pd.read_json(sys.argv[2], lines=True)

base_commit_id = set(base["commit_id"].unique())
pr_commit_id = set(pr["commit_id"].unique())

assert len(base_commit_id) == 1, base_commit_id
base_commit_id = next(iter(base_commit_id))
assert len(pr_commit_id) == 1, pr_commit_id
pr_commit_id = next(iter(pr_commit_id))

# Handle missing storage field
if "storage" not in base:
    base["storage"] = pd.NA
if "storage" not in pr:
    pr["storage"] = pd.NA


# Handle missing dataset field and create a dataset key for joining
def extract_dataset_key(df):
    if "dataset" not in df.columns:
        df["dataset_key"] = pd.NA
    else:
        # Convert dataset dict to a string representation for joining
        df["dataset_key"] = df["dataset"].apply(
            lambda x: str(sorted(x.items())) if pd.notna(x) and isinstance(x, dict) else pd.NA
        )
    return df


base = extract_dataset_key(base)
pr = extract_dataset_key(pr)

# Join on name, storage, and dataset_key
# NB: `pd.merge` considers two null key values to be equal, so benchmarks without these keys will match.
df3 = pd.merge(base, pr, on=["name", "storage", "dataset_key"], how="right", suffixes=("_base", "_pr"))

# assert df3["unit_base"].equals(df3["unit_pr"]), (df3["unit_base"], df3["unit_pr"])

# Determine threshold based on benchmark name
# Use 30% threshold for S3 benchmarks, 10% for others
is_s3_benchmark = "s3" in benchmark_name.lower()
threshold_pct = S3_THRESHOLD_PCT if is_s3_benchmark else DEFAULT_THRESHOLD_PCT
improvement_threshold = 1.0 - (threshold_pct / 100.0)  # e.g., 0.7 for 30%, 0.9 for 10%
regression_threshold = 1.0 + (threshold_pct / 100.0)  # e.g., 1.3 for 30%, 1.1 for 10%


def compute_cv_pct(runtimes):
    """Compute coefficient of variation (std_dev / mean * 100) as a percentage."""
    if not isinstance(runtimes, list) or len(runtimes) < 2:
        return float("nan")
    n = len(runtimes)
    mean = sum(runtimes) / n
    if mean == 0:
        return float("nan")
    variance = sum((x - mean) ** 2 for x in runtimes) / (n - 1)
    return (variance**0.5 / mean) * 100


# Compute CV% from all_runtimes when available
has_runtimes_pr = "all_runtimes_pr" in df3.columns
has_runtimes_base = "all_runtimes_base" in df3.columns
if has_runtimes_pr:
    df3["cv_pct_pr"] = df3["all_runtimes_pr"].apply(compute_cv_pct)
if has_runtimes_base:
    df3["cv_pct_base"] = df3["all_runtimes_base"].apply(compute_cv_pct)
if has_runtimes_pr or has_runtimes_base:
    cv_columns = [column for column in ["cv_pct_pr", "cv_pct_base"] if column in df3.columns]
    df3["cv_pct_max"] = df3[cv_columns].max(axis=1, skipna=True)


def describe_noise(cv_pct):
    """Bucket runtime noise into labels that are easy to scan in GitHub tables."""
    if pd.isna(cv_pct):
        return "unknown"
    if cv_pct < NOISE_LOW_MAX_CV_PCT:
        return "low"
    if cv_pct < NOISE_MEDIUM_MAX_CV_PCT:
        return "medium"
    return "high"


if "cv_pct_max" in df3.columns:
    df3["noise"] = df3["cv_pct_max"].apply(describe_noise)

# Generate summary statistics
df3["ratio"] = df3["value_pr"] / df3["value_base"]
df3["remark"] = pd.Series([""] * len(df3))
df3["remark"] = df3["remark"].case_when(
    [
        (df3["ratio"] >= regression_threshold, "🚨"),
        (df3["ratio"] <= improvement_threshold, "🚀"),
    ]
)

# Filter for different target combinations for summary statistics
vortex_df = df3[df3["name"].str.contains("vortex", case=False, na=False)]
duckdb_vortex_df = df3[df3["name"].str.contains("duckdb.*vortex", case=False, na=False, regex=True)]
datafusion_vortex_df = df3[df3["name"].str.contains("datafusion.*vortex", case=False, na=False, regex=True)]
parquet_df = df3[df3["name"].str.contains("parquet", case=False, na=False)]


# Overall performance (all results)
valid_positive_ratios = [r for r in df3["ratio"] if r > 0 and not pd.isna(r)]
if len(valid_positive_ratios) > 0:
    geo_mean_ratio = math.exp(sum(math.log(r) for r in valid_positive_ratios) / len(valid_positive_ratios))
else:
    geo_mean_ratio = float("nan")


# Performance for different target combinations
def calculate_geo_mean(df):
    valid_ratios = [r for r in df["ratio"] if r > 0 and not pd.isna(r)]
    if len(valid_ratios) > 0:
        return math.exp(sum(math.log(r) for r in valid_ratios) / len(valid_ratios))
    else:
        return float("nan")


def calculate_run_drift_metrics(df):
    """Summarize common-mode movement across the whole benchmark run.

    We work in log-ratio space because ratios compose multiplicatively:
    a 10% slowdown (1.10x) and a 10% speedup (0.90x) are roughly symmetric
    once transformed with log(). That makes the median log-ratio a robust
    estimate of "the whole run was faster/slower than usual".
    """
    valid_ratios = [r for r in df["ratio"] if r > 0 and not pd.isna(r)]
    if not valid_ratios:
        return {
            "drift_ratio": float("nan"),
            "same_direction_pct": float("nan"),
            "residual_mad_pct": float("nan"),
            "is_baseline_suspect": False,
            "drift_level": "unknown",
        }

    log_ratios = pd.Series([math.log(r) for r in valid_ratios])
    median_log_ratio = float(log_ratios.median())
    drift_ratio = math.exp(median_log_ratio)

    # Count how often benchmarks move in the same direction as the run-wide drift.
    # This distinguishes "everything got faster/slower together" from a mixed run
    # with a similar central tendency.
    if median_log_ratio < 0:
        same_direction_pct = float((log_ratios < 0).mean() * 100)
    elif median_log_ratio > 0:
        same_direction_pct = float((log_ratios > 0).mean() * 100)
    else:
        same_direction_pct = float((log_ratios == 0).mean() * 100)

    # Residual MAD measures how tightly benchmarks cluster around the run-wide
    # drift. Small residual spread plus broad agreement is a strong indicator
    # that the baseline itself is shifted rather than the PR changing specific
    # benchmarks independently.
    residual_logs = log_ratios - median_log_ratio
    residual_mad = float(residual_logs.abs().median())
    residual_mad_pct = (math.exp(residual_mad) - 1.0) * 100

    is_baseline_suspect = (
        abs(drift_ratio - 1.0) * 100 >= DRIFT_NOTICEABLE_THRESHOLD_PCT
        and same_direction_pct >= DRIFT_CONSISTENCY_THRESHOLD_PCT
        and residual_mad_pct <= DRIFT_RESIDUAL_SPREAD_THRESHOLD_PCT
    )
    if is_baseline_suspect:
        drift_level = "large"
    elif abs(drift_ratio - 1.0) * 100 >= DRIFT_NOTICEABLE_THRESHOLD_PCT:
        drift_level = "noticeable"
    else:
        drift_level = "small"

    return {
        "drift_ratio": drift_ratio,
        "same_direction_pct": same_direction_pct,
        "residual_mad_pct": residual_mad_pct,
        "is_baseline_suspect": is_baseline_suspect,
        "drift_level": drift_level,
    }


def format_performance(ratio, target_name):
    if pd.isna(ratio):
        return f"no {target_name.lower()} data"
    else:
        # Use neutral emoji if within threshold
        if improvement_threshold <= ratio <= regression_threshold:
            emoji = "➖"
        elif ratio < 1:
            emoji = "✅"
        else:
            emoji = "❌"
        return f"{ratio:.3f}x {emoji}"


def build_summary_lines(
    overall_ratio,
    vortex_df,
    parquet_df,
    duckdb_vortex_df,
    datafusion_vortex_df,
    threshold_pct,
    run_drift_metrics,
):
    """Build markdown summary lines from precomputed benchmark metrics."""
    vortex_geo_mean_ratio = calculate_geo_mean(vortex_df)
    duckdb_vortex_geo_mean_ratio = calculate_geo_mean(duckdb_vortex_df)
    datafusion_vortex_geo_mean_ratio = calculate_geo_mean(datafusion_vortex_df)
    parquet_geo_mean_ratio = calculate_geo_mean(parquet_df)

    overall_performance = "no data" if pd.isna(overall_ratio) else format_performance(overall_ratio, "overall")
    vortex_performance = format_performance(vortex_geo_mean_ratio, "vortex")
    duckdb_vortex_performance = format_performance(duckdb_vortex_geo_mean_ratio, "duckdb:vortex")
    datafusion_vortex_performance = format_performance(datafusion_vortex_geo_mean_ratio, "datafusion:vortex")
    parquet_performance = format_performance(parquet_geo_mean_ratio, "parquet")

    summary_lines = [
        "## Summary",
        "",
        f"- **Overall**: {overall_performance}",
        (
            f"- **Run drift**: {run_drift_metrics['drift_ratio']:.3f}x, "
            f"{run_drift_metrics['drift_level']} whole-run shift "
            f"({run_drift_metrics['same_direction_pct']:.0f}% aligned, "
            f"residual MAD {run_drift_metrics['residual_mad_pct']:.1f}%)"
            if not pd.isna(run_drift_metrics["drift_ratio"])
            else "- **Run drift**: no data"
        ),
    ]

    if len(vortex_df) > 0:
        summary_lines.append(f"- **Vortex**: {vortex_performance}")

    if len(parquet_df) > 0:
        summary_lines.append(f"- **Parquet**: {parquet_performance}")

    if len(duckdb_vortex_df) > 0:
        summary_lines.append(f"- **duckdb:vortex**: {duckdb_vortex_performance}")

    if len(datafusion_vortex_df) > 0:
        summary_lines.append(f"- **datafusion:vortex**: {datafusion_vortex_performance}")

    if len(vortex_df) > 0:
        vortex_valid_ratios = vortex_df["ratio"].dropna()
        if len(vortex_valid_ratios) > 0:
            improvements = vortex_valid_ratios[vortex_valid_ratios < 1.0]
            if len(improvements) > 0:
                best_idx = improvements.idxmin()
                best_improvement = f"{vortex_df.loc[best_idx, 'name']} ({vortex_df.loc[best_idx, 'ratio']:.3f}x)"
            else:
                best_improvement = "No improvements"

            regressions = vortex_valid_ratios[vortex_valid_ratios > 1.0]
            if len(regressions) > 0:
                worst_idx = regressions.idxmax()
                worst_regression = f"{vortex_df.loc[worst_idx, 'name']} ({vortex_df.loc[worst_idx, 'ratio']:.3f}x)"
            else:
                worst_regression = "No regressions"
        else:
            best_improvement = "No valid vortex comparisons"
            worst_regression = "No valid vortex comparisons"

        significant_improvements = (vortex_df["ratio"] < improvement_threshold).sum()
        significant_regressions = (vortex_df["ratio"] > regression_threshold).sum()
        summary_lines.extend(
            [
                f"- **Best**: {best_improvement}",
                f"- **Worst**: {worst_regression}",
                f"- **Significant (>{threshold_pct}%)**: {significant_improvements}↑ {significant_regressions}↓",
            ]
        )

    if run_drift_metrics["is_baseline_suspect"]:
        summary_lines.append("- **Baseline**: likely unreliable for this run; most benchmarks shifted together")

    return summary_lines


def build_display_table(df, pr_commit_id, base_commit_id):
    """Build the markdown table with display-friendly units and columns.

    This keeps presentation-only concerns out of the metric computation above.
    """
    display_df = df.copy()

    # Convert rendered timing values from ns to ms to keep the GitHub table narrower.
    # This affects display only; all comparison math above stays in the original units.
    display_df["value_pr_display"] = display_df["value_pr"].astype(float)
    display_df["value_base_display"] = display_df["value_base"].astype(float)
    display_df["unit_display"] = display_df["unit_base"].copy()
    ns_mask = display_df["unit_display"] == "ns"
    display_df.loc[ns_mask, "value_pr_display"] = display_df.loc[ns_mask, "value_pr_display"] / 1_000_000
    display_df.loc[ns_mask, "value_base_display"] = display_df.loc[ns_mask, "value_base_display"] / 1_000_000
    display_df.loc[ns_mask, "unit_display"] = "ms"

    table_dict = {
        "name": display_df["name"],
        f"PR {pr_commit_id[:8]}": display_df["value_pr_display"],
        f"base {base_commit_id[:8]}": display_df["value_base_display"],
        "ratio (PR/base)": display_df["ratio"],
        "unit": display_df["unit_display"],
    }

    if "noise" in display_df.columns:
        table_dict["noise"] = display_df["noise"]

    table_dict["remark"] = display_df["remark"]
    return pd.DataFrame(table_dict)


run_drift_metrics = calculate_run_drift_metrics(df3)
summary_lines = build_summary_lines(
    geo_mean_ratio,
    vortex_df,
    parquet_df,
    duckdb_vortex_df,
    datafusion_vortex_df,
    threshold_pct,
    run_drift_metrics,
)
table_df = build_display_table(df3, pr_commit_id, base_commit_id)

# Output complete formatted markdown
print("\n".join(summary_lines))
print("")
print("<details>")
print("<summary>Detailed Results Table</summary>")
print("")
print(table_df.to_markdown(index=False, tablefmt="github", floatfmt=".2f"))
print("</details>")
