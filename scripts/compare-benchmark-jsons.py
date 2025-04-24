# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///
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

print(
    pd.DataFrame(
        {
            "name": df3["name"],
            f"PR {pr_commit_id[:8]}": df3["value_pr"],
            f"base {base_commit_id[:8]}": df3["value_base"],
            "ratio (PR/base)": df3["value_pr"] / df3["value_base"],
            "unit": df3["unit_base"],
        }
    ).to_markdown(index=False)
)
