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

base = pd.read_json(sys.argv[1], lines=True)
pr = pd.read_json(sys.argv[2], lines=True)

base_commit_id = set(base["commit_id"].unique())
pr_commit_id = set(pr["commit_id"].unique())

assert len(base_commit_id) == 1, base_commit_id
base_commit_id = next(iter(base_commit_id))
assert len(pr_commit_id) == 1, pr_commit_id
pr_commit_id = next(iter(pr_commit_id))

if "storage" not in base:
    # For whatever reason, the base lacks storage. Might be an old database of results. Might be a
    # database of results without any storage fields.
    base["storage"] = pd.NA

if "storage" not in pr:
    # Not all benchmarks have a "storage" key. If none of the JSON objects in the PR results file
    # had a "storage" key, then the PR DataFrame will lack that key and the join will fail.
    pr["storage"] = pd.NA

# NB: `pd.merge` considers two null key values to be equal, so benchmarks without storage keys will
# match.
df3 = pd.merge(base, pr, on=["name", "storage"], how="right", suffixes=("_base", "_pr"))

# assert df3["unit_base"].equals(df3["unit_pr"]), (df3["unit_base"], df3["unit_pr"])

# Generate summary statistics
df3["ratio"] = df3["value_pr"] / df3["value_base"]

# Calculate geometric mean of ratios (better for performance ratios)
import math
geo_mean_ratio = math.exp(sum(math.log(r) for r in df3["ratio"] if r > 0) / len(df3["ratio"]))

# Find best and worst changes
best_idx = df3["ratio"].idxmin()
worst_idx = df3["ratio"].idxmax()

# Count significant changes (>10% change)
significant_improvements = (df3["ratio"] < 0.9).sum()  # More than 10% faster
significant_regressions = (df3["ratio"] > 1.1).sum()   # More than 10% slower

# Count all changes
improvements = (df3["ratio"] < 1.0).sum()
regressions = (df3["ratio"] > 1.0).sum()
unchanged = (df3["ratio"] == 1.0).sum()

# Build summary
summary_lines = [
    "## Summary",
    "",
    f"- **Overall Performance (geometric mean)**: {geo_mean_ratio:.3f}x ({'better' if geo_mean_ratio < 1 else 'worse'} than base)",
    f"- **Best Improvement**: {df3.loc[best_idx, 'name']} ({df3.loc[best_idx, 'ratio']:.3f}x)",
    f"- **Worst Regression**: {df3.loc[worst_idx, 'name']} ({df3.loc[worst_idx, 'ratio']:.3f}x)",
    f"- **Significant Changes (>10%)**:",
    f"  - Improvements: {significant_improvements} queries",
    f"  - Regressions: {significant_regressions} queries",
]
if unchanged > 0:
    summary_lines.append(f"  - Unchanged: {unchanged} queries")

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
