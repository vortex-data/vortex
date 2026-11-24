#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Count the tests and benchmarks that a PR adds and removes, and render a Markdown report.

The counts come from the diff alone, so the script needs no build and never runs the PR's code:

    python3 scripts/pr_test_bench_counts.py --base <base-sha> --head <head-sha> > comment.md

The diff is taken against the merge base of the two commits, so the PR's commits are counted and
the commits merged into the base branch since it forked are not. A line that adds a test attribute
counts one, a line that removes one counts minus one, so tests moved between files net to zero.

What is counted:

- Rust test functions: `#[test]`, `#[rstest]` and other `#[...::test]` attributes such as
  `#[tokio::test]`. `#[case]` lines of rstest tests are counted separately.
- Python tests: `def test_*` functions in `test_*.py`, `*_test.py` and `conftest.py` files.
- Rust benchmark functions: `#[divan::bench]`, `#[bench]`, and Criterion `bench_function` and
  `bench_with_input` calls.
- CodSpeed benchmarks: the change in the total of `scripts/codspeed_bench_counts.json`, which counts
  each argument of a parameterized benchmark.

When `GITHUB_OUTPUT` is set, the script writes `has-changes` (whether any count is non-zero) and
`over-threshold` (whether any net count exceeds its warning threshold) to it.
"""

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import PurePosixPath

COUNTS_PATH = "scripts/codspeed_bench_counts.json"
COMMENT_MARKER = "<!-- pr-test-bench-counts -->"

RUST_TEST_RE = re.compile(r"^#\[(?:\w+::)*(?:test|rstest|cuda_test)\b")
RUST_CASE_RE = re.compile(r"^#\[case\b")
RUST_BENCH_ATTR_RE = re.compile(r"^#\[(?:divan::)?bench\b")
RUST_BENCH_CALL_RE = re.compile(r"\.bench_(?:function|with_input)\s*\(")
PYTHON_TEST_RE = re.compile(r"^(?:async\s+)?def\s+test_\w*\s*\(")

CATEGORIES = {
    "rust_tests": "Rust test functions",
    "rust_cases": "rstest `#[case]`s",
    "python_tests": "Python tests",
    "rust_benches": "Rust benchmark functions",
}
TEST_CATEGORIES = ("rust_tests", "rust_cases", "python_tests")
BENCH_CATEGORIES = ("rust_benches",)


@dataclasses.dataclass
class Count:
    added: int = 0
    removed: int = 0

    @property
    def net(self) -> int:
        return self.added - self.removed


def is_python_test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def classify(path: str, line: str) -> list[str]:
    """Return the categories that one added or removed source line counts towards."""
    stripped = line.strip()
    if path.endswith(".rs"):
        found = []
        if RUST_TEST_RE.match(stripped):
            found.append("rust_tests")
        if RUST_CASE_RE.match(stripped):
            found.append("rust_cases")
        if RUST_BENCH_ATTR_RE.match(stripped):
            found.append("rust_benches")
        if not stripped.startswith("//"):
            found.extend("rust_benches" for _ in RUST_BENCH_CALL_RE.finditer(stripped))
        return found
    if is_python_test_file(path) and PYTHON_TEST_RE.match(stripped):
        return ["python_tests"]
    return []


def group_of(path: str) -> str:
    """Name the crate or package that a file belongs to: the path before its `src`, `tests`..."""
    parts = PurePosixPath(path).parts
    for i, part in enumerate(parts):
        if part in ("src", "tests", "benches", "examples", "python") and i > 0:
            return "/".join(parts[:i])
    return parts[0] if len(parts) > 1 else "."


def count_diff(diff: str) -> dict[str, dict[str, Count]]:
    """Count the tests and benchmarks in a unified diff, by category and then by group."""
    counts: dict[str, dict[str, Count]] = defaultdict(lambda: defaultdict(Count))
    old_path = new_path = ""
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
            old_path = new_path = ""
            continue
        if not in_hunk:
            if line.startswith("--- "):
                old_path = line[4:].removeprefix("a/")
            elif line.startswith("+++ "):
                new_path = line[4:].removeprefix("b/")
            elif line.startswith("@@"):
                in_hunk = True
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+"):
            path, added = new_path, True
        elif line.startswith("-"):
            path, added = old_path, False
        else:
            continue
        if path == "/dev/null":
            continue
        for category in classify(path, line[1:]):
            count = counts[category][group_of(path)]
            if added:
                count.added += 1
            else:
                count.removed += 1
    return counts


def total(by_group: dict[str, Count]) -> Count:
    return Count(
        added=sum(c.added for c in by_group.values()),
        removed=sum(c.removed for c in by_group.values()),
    )


def codspeed_total(counts_json: str | None) -> int:
    if not counts_json:
        return 0
    return sum(sum(targets.values()) for targets in json.loads(counts_json).values())


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def git_show(rev: str, path: str) -> str | None:
    result = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else None


def signed(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def render(
    counts: dict[str, dict[str, Count]],
    codspeed_delta: int,
    tests_warn: int,
    benches_warn: int,
    codspeed_warn: int,
) -> tuple[str, bool]:
    """Render the Markdown report. Return it and whether any threshold was exceeded."""
    totals = {category: total(counts.get(category, {})) for category in CATEGORIES}
    net_tests = sum(totals[c].net for c in TEST_CATEGORIES)
    net_benches = sum(totals[c].net for c in BENCH_CATEGORIES)

    warnings = []
    if net_tests > tests_warn:
        warnings.append(f"adds {net_tests} tests, more than the {tests_warn} expected in one PR")
    if net_benches > benches_warn:
        warnings.append(f"adds {net_benches} benchmark functions, more than the {benches_warn} expected in one PR")
    if codspeed_delta > codspeed_warn:
        warnings.append(f"adds {codspeed_delta} CodSpeed benchmarks, more than the {codspeed_warn} expected in one PR")

    lines = [COMMENT_MARKER, "### Tests and benchmarks", ""]
    if warnings:
        lines.append("> [!WARNING]")
        lines.append(f"> This PR {'; it '.join(warnings)}.")
        lines.append(
            "> Check that each one covers a distinct behaviour: every test and benchmark adds to CI "
            "time, and CodSpeed limits the benchmarks it runs. Consider `rstest` cases or fewer "
            "benchmark arguments."
        )
        lines.append("")

    lines.append("| | Added | Removed | Net |")
    lines.append("|---|---:|---:|---:|")
    for category, label in CATEGORIES.items():
        t = totals[category]
        lines.append(f"| {label} | {t.added} | {t.removed} | {signed(t.net)} |")
    lines.append(f"| CodSpeed benchmarks (`{COUNTS_PATH}`) | | | {signed(codspeed_delta)} |")
    lines.append("")

    groups = sorted({g for by_group in counts.values() for g in by_group})
    rows = []
    for group in groups:
        tests = sum(counts.get(c, {}).get(group, Count()).net for c in TEST_CATEGORIES)
        benches = sum(counts.get(c, {}).get(group, Count()).net for c in BENCH_CATEGORIES)
        if tests or benches:
            rows.append(f"| `{group}` | {signed(tests)} | {signed(benches)} |")
    if rows:
        lines.append("<details><summary>Net change by crate</summary>")
        lines.append("")
        lines.append("| Crate | Tests | Benchmark functions |")
        lines.append("|---|---:|---:|")
        lines.extend(rows)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append(
        f"<sub>Counted from the diff. Warns above {tests_warn} net tests, {benches_warn} net "
        f"benchmark functions or {codspeed_warn} net CodSpeed benchmarks.</sub>"
    )
    return "\n".join(lines) + "\n", bool(warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="the base branch commit")
    parser.add_argument("--head", required=True, help="the PR head commit")
    parser.add_argument("--tests-warn", type=int, default=40)
    parser.add_argument("--benches-warn", type=int, default=10)
    parser.add_argument("--codspeed-warn", type=int, default=100)
    args = parser.parse_args()

    merge_base = git("merge-base", args.base, args.head).strip()
    diff = git(
        "diff", "--no-color", "--no-ext-diff", "--unified=0", "-M",
        merge_base, args.head, "--", "*.rs", "*.py",
    )  # fmt: skip
    counts = count_diff(diff)
    codspeed_delta = codspeed_total(git_show(args.head, COUNTS_PATH)) - codspeed_total(
        git_show(merge_base, COUNTS_PATH)
    )

    report, over = render(counts, codspeed_delta, args.tests_warn, args.benches_warn, args.codspeed_warn)
    sys.stdout.write(report)

    has_changes = codspeed_delta != 0 or any(
        c.added or c.removed for by_group in counts.values() for c in by_group.values()
    )
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as f:
            f.write(f"has-changes={str(has_changes).lower()}\n")
            f.write(f"over-threshold={str(over).lower()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
