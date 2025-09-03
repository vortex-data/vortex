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
pr_mean = df3["value_pr"].mean()
base_mean = df3["value_base"].mean()
overall_ratio = pr_mean / base_mean if base_mean > 0 else float('nan')

# Count improvements vs regressions
improvements = (df3["value_pr"] < df3["value_base"]).sum()
regressions = (df3["value_pr"] > df3["value_base"]).sum()
unchanged = (df3["value_pr"] == df3["value_base"]).sum()

# Find best and worst changes
df3["ratio"] = df3["value_pr"] / df3["value_base"]
best_idx = df3["ratio"].idxmin()
worst_idx = df3["ratio"].idxmax()

# Print summary
print("## Summary")
print()
print(f"- **Overall Performance**: {overall_ratio:.3f}x ({'better' if overall_ratio < 1 else 'worse'} than base)")
print(f"- **Improvements**: {improvements} queries")
print(f"- **Regressions**: {regressions} queries")
if unchanged > 0:
    print(f"- **Unchanged**: {unchanged} queries")
print(f"- **Best Improvement**: {df3.loc[best_idx, 'name']} ({df3.loc[best_idx, 'ratio']:.3f}x)")
print(f"- **Worst Regression**: {df3.loc[worst_idx, 'name']} ({df3.loc[worst_idx, 'ratio']:.3f}x)")
print()

# Print detailed table
print(
    pd.DataFrame(
        {
            "name": df3["name"],
            f"PR {pr_commit_id[:8]}": df3["value_pr"],
            f"base {base_commit_id[:8]}": df3["value_base"],
            "ratio (PR/base)": df3["ratio"],
            "unit": df3["unit_base"],
        }
    ).to_markdown(index=False)
)
