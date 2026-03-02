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
threshold_pct = 30 if is_s3_benchmark else 10
improvement_threshold = 1.0 - (threshold_pct / 100.0)  # e.g., 0.7 for 30%, 0.9 for 10%
regression_threshold = 1.0 + (threshold_pct / 100.0)  # e.g., 1.3 for 30%, 1.1 for 10%


def compute_abs_z_score(runtimes):
    """Compute |median - mean| / stddev from a list of runtimes."""
    if not isinstance(runtimes, list) or len(runtimes) < 2:
        return float("nan")
    n = len(runtimes)
    mean = sum(runtimes) / n
    variance = sum((x - mean) ** 2 for x in runtimes) / (n - 1)
    stddev = math.sqrt(variance)
    if stddev == 0:
        return 0.0
    sorted_rt = sorted(runtimes)
    if n % 2 == 1:
        median = sorted_rt[n // 2]
    else:
        median = (sorted_rt[n // 2 - 1] + sorted_rt[n // 2]) / 2
    return abs((median - mean) / stddev)


# Compute |z-score| from all_runtimes when available
has_z_pr = "all_runtimes_pr" in df3.columns
has_z_base = "all_runtimes_base" in df3.columns
if has_z_pr:
    df3["abs_z_score_pr"] = df3["all_runtimes_pr"].apply(compute_abs_z_score)
if has_z_base:
    df3["abs_z_score_base"] = df3["all_runtimes_base"].apply(compute_abs_z_score)

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


vortex_geo_mean_ratio = calculate_geo_mean(vortex_df)
duckdb_vortex_geo_mean_ratio = calculate_geo_mean(duckdb_vortex_df)
datafusion_vortex_geo_mean_ratio = calculate_geo_mean(datafusion_vortex_df)
parquet_geo_mean_ratio = calculate_geo_mean(parquet_df)

# Find best and worst changes for vortex-only results
vortex_valid_ratios = vortex_df["ratio"].dropna()
if len(vortex_valid_ratios) > 0:
    # Best improvement: smallest ratio (< 1.0, fastest performance)
    improvements = vortex_valid_ratios[vortex_valid_ratios < 1.0]
    if len(improvements) > 0:
        best_idx = improvements.idxmin()
        best_improvement = f"{vortex_df.loc[best_idx, 'name']} ({vortex_df.loc[best_idx, 'ratio']:.3f}x)"
    else:
        best_improvement = "No improvements"

    # Worst regression: largest ratio (> 1.0, slowest performance)
    regressions = vortex_valid_ratios[vortex_valid_ratios > 1.0]
    if len(regressions) > 0:
        worst_idx = regressions.idxmax()
        worst_regression = f"{vortex_df.loc[worst_idx, 'name']} ({vortex_df.loc[worst_idx, 'ratio']:.3f}x)"
    else:
        worst_regression = "No regressions"
else:
    best_improvement = "No valid vortex comparisons"
    worst_regression = "No valid vortex comparisons"

# Count significant changes for vortex-only results
significant_improvements = (vortex_df["ratio"] < improvement_threshold).sum()
significant_regressions = (vortex_df["ratio"] > regression_threshold).sum()


# Build summary
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


overall_performance = "no data" if pd.isna(geo_mean_ratio) else format_performance(geo_mean_ratio, "overall")
vortex_performance = format_performance(vortex_geo_mean_ratio, "vortex")
duckdb_vortex_performance = format_performance(duckdb_vortex_geo_mean_ratio, "duckdb:vortex")
datafusion_vortex_performance = format_performance(datafusion_vortex_geo_mean_ratio, "datafusion:vortex")
parquet_performance = format_performance(parquet_geo_mean_ratio, "parquet")


summary_lines = [
    "## Summary",
    "",
    f"- **Overall**: {overall_performance}",
]

# Only add vortex-specific sections if we have vortex data
if len(vortex_df) > 0:
    summary_lines.extend([f"- **Vortex**: {vortex_performance}"])

if len(parquet_df) > 0:
    summary_lines.extend([f"- **Parquet**: {parquet_performance}"])

# Only add duckdb:vortex section if we have that data
if len(duckdb_vortex_df) > 0:
    summary_lines.append(f"- **duckdb:vortex**: {duckdb_vortex_performance}")

# Only add datafusion:vortex section if we have that data
if len(datafusion_vortex_df) > 0:
    summary_lines.append(f"- **datafusion:vortex**: {datafusion_vortex_performance}")

# Only add best/worst if we have vortex data
if len(vortex_df) > 0:
    summary_lines.extend(
        [
            f"- **Best**: {best_improvement}",
            f"- **Worst**: {worst_regression}",
            f"- **Significant (>{threshold_pct}%)**: {significant_improvements}↑ {significant_regressions}↓",
        ]
    )

# Build table
table_dict = {
    "name": df3["name"],
    f"PR {pr_commit_id[:8]}": df3["value_pr"],
    f"base {base_commit_id[:8]}": df3["value_base"],
    "ratio (PR/base)": df3["ratio"],
    "unit": df3["unit_base"],
}

if has_z_pr:
    table_dict["abs(z-score) PR"] = df3["abs_z_score_pr"]
if has_z_base:
    table_dict["abs(z-score) base"] = df3["abs_z_score_base"]

table_dict["remark"] = df3["remark"]
table_df = pd.DataFrame(table_dict)

# Output complete formatted markdown
print("\n".join(summary_lines))
print("")
print("<details>")
print("<summary>Detailed Results Table</summary>")
print("")
print(table_df.to_markdown(index=False, tablefmt="github", floatfmt=".2f"))
print("</details>")
