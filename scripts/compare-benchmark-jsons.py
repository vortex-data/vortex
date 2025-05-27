# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///
import argparse
import subprocess

import pandas as pd

PARSER = argparse.ArgumentParser(description="Compare two benchmark JSON files.")
PARSER.add_argument(
    "base",
    type=str,
    help="Path to the commit base benchmark JSON file.",
)
PARSER.add_argument(
    "results",
    type=str,
    help="Path to the pull request results JSON file.",
)
PARSER.add_argument(
    "--base-commit",
    default=None,
    help="The base git commit to compare against. If not given, will find common ancestor to develop branch",
)


def _merge_results(base: pd.DataFrame, pr: pd.DataFrame, commit_id) -> pd.DataFrame | None:
    """Merges the base and PR results DataFrames on the 'name' and 'storage' columns.

    Returns `None` if the commit ID is not found in the base DataFrame.
    """
    commit_base = base[base["commit_id"] == commit_id]
    if commit_base.empty:
        return None

    merged = pd.merge(commit_base, pr, on=["name", "storage"], how="right", suffixes=("_base", "_pr"))

    if merged["value_base"].isna().all():
        # If all results are null, then this benchmark must have failed for this commit.
        return None

    return merged


def _print_results(base_commit: str, pr_commit: str, merged: pd.DataFrame):
    print(
        pd.DataFrame(
            {
                "name": merged["name"],
                f"PR {pr_commit[:8]}": merged["value_pr"],
                f"base {base_commit[:8]}": merged["value_base"],
                "ratio (PR/base)": merged["value_pr"] / merged["value_base"],
                "unit": merged["unit_base"],
            }
        ).to_markdown(index=False)
    )


def main(args):
    base = pd.read_json(args.base, lines=True)
    pr = pd.read_json(args.results, lines=True)

    pr_commit_id = set(pr["commit_id"].unique())
    assert len(pr_commit_id) == 1, "PR results must have exactly one commit ID."
    pr_commit_id = next(iter(pr_commit_id))

    if "storage" not in base:
        # For whatever reason, the base lacks storage. Might be an old database of results. Might be a
        # database of results without any storage fields.
        base["storage"] = pd.NA

    if "storage" not in pr:
        # Not all benchmarks have a "storage" key. If none of the JSON objects in the PR results file
        # had a "storage" key, then the PR DataFrame will lack that key and the join will fail.
        pr["storage"] = pd.NA

    # We find the commit ID to compare against
    if args.base_commit is not None:
        df = _merge_results(base, pr, args.base_commit)
        if df is not None:
            _print_results(args.base_commit, pr_commit_id, df)
            return

    # Find the common ancestor commit ID with the develop branch
    base_commit_id = (
        subprocess.check_output(["git", "merge-base", f"{pr_commit_id}^", "develop"]).decode("utf-8").strip()
    )
    df = _merge_results(base, pr, base_commit_id)
    if df is not None:
        _print_results(base_commit_id, pr_commit_id, df)
        return

    # Walk the git log until we find a completed benchmark result.
    git_log = (
        subprocess.check_output(["git", "log", "--format=%H", "-n", "30", "--skip=1", base_commit_id])
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    for commit_id in git_log:
        df = _merge_results(base, pr, commit_id)
        if df is not None:
            _print_results(commit_id, pr_commit_id, df)
            return


if __name__ == "__main__":
    args = PARSER.parse_args()
    main(args)
