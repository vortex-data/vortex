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
    # This means the base commit was generated in the pre-object-store days. We cannot give a true
    # diff because we're comparing different storage systems.
    pr
    print(
        pd.DataFrame(
            {
                "name": pr["name"],
                f"PR {pr_commit_id[:8]}": pr["value"],
                f"base {base_commit_id[:8]} (no S3 results found)": pd.NA,
                "ratio (PR/base)": pd.NA,
                "unit": pr["unit"],
            }
        ).to_markdown(index=False)
    )
    sys.exit(0)

df3 = pd.merge(base, pr, on=["name", "storage"], how="right", suffixes=("_base", "_pr"))

assert df3["unit_base"].equals(df3["unit_pr"]), (df3["unit_base"], df3["unit_pr"])

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
