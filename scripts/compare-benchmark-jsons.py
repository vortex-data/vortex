# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

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

# Generate summary statistics
df3["ratio"] = df3["value_pr"] / df3["value_base"]

# Calculate geometric mean of ratios (better for performance ratios)
import math
valid_positive_ratios = [r for r in df3["ratio"] if r > 0 and not pd.isna(r)]
if len(valid_positive_ratios) > 0:
    geo_mean_ratio = math.exp(sum(math.log(r) for r in valid_positive_ratios) / len(valid_positive_ratios))
else:
    geo_mean_ratio = float('nan')

# Find best and worst changes (handle case where all ratios are NaN)
valid_ratios = df3["ratio"].dropna()
if len(valid_ratios) > 0:
    best_idx = valid_ratios.idxmin()
    worst_idx = valid_ratios.idxmax()
    best_improvement = f"{df3.loc[best_idx, 'name']} ({df3.loc[best_idx, 'ratio']:.3f}x)"
    worst_regression = f"{df3.loc[worst_idx, 'name']} ({df3.loc[worst_idx, 'ratio']:.3f}x)"
else:
    best_improvement = "No valid comparisons"
    worst_regression = "No valid comparisons"

# Determine threshold based on benchmark name
# Use 30% threshold for S3 benchmarks, 10% for others
is_s3_benchmark = "s3" in benchmark_name.lower()
threshold_pct = 30 if is_s3_benchmark else 10
improvement_threshold = 1.0 - (threshold_pct / 100.0)  # e.g., 0.7 for 30%, 0.9 for 10%
regression_threshold = 1.0 + (threshold_pct / 100.0)   # e.g., 1.3 for 30%, 1.1 for 10%

# Count significant changes
significant_improvements = (df3["ratio"] < improvement_threshold).sum()
significant_regressions = (df3["ratio"] > regression_threshold).sum()

# Build summary
if pd.isna(geo_mean_ratio):
    overall_performance = "No valid comparisons available"
else:
    overall_performance = f"{geo_mean_ratio:.3f}x ({'better' if geo_mean_ratio < 1 else 'worse'} than base)"

summary_lines = [
    "## Summary",
    "",
    f"- **Overall Performance (geometric mean)**: {overall_performance}",
    f"- **Best Improvement**: {best_improvement}",
    f"- **Worst Regression**: {worst_regression}",
    f"- **Significant Changes (>{threshold_pct}%)**:",
    f"  - Improvements: {significant_improvements} queries",
    f"  - Regressions: {significant_regressions} queries",
]

# Build table
table_df = pd.DataFrame(
    {
        "name": df3["name"],
        f"PR {pr_commit_id[:8]}": df3["value_pr"],
        f"base {base_commit_id[:8]}": df3["value_base"],
        "ratio (PR/base)": df3["ratio"],
        "unit": df3["unit_base"],
    }
)

# Output complete formatted markdown
print("\n".join(summary_lines))
print("")
print("<details>")
print("<summary>Detailed Results Table</summary>")
print("")
print(table_df.to_markdown(index=False))
print("</details>")
