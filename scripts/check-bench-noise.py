# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Check benchmark results for noisy measurements.

Reads a JSONL benchmark results file and checks the coefficient of variation
(CoV = std / mean) of each benchmark's runtimes. Exits with code 0 if the
run is clean, or code 1 if too many benchmarks are noisy and a rerun is
recommended.

When --history is provided, per-benchmark adaptive thresholds are computed
from historical data so that inherently-variable benchmarks are not penalized.

Usage:
    check-bench-noise.py <results.json> [--cov-threshold 0.15] [--noisy-fraction 0.25]
    check-bench-noise.py <results.json> --history data.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def coefficient_of_variation(runtimes: list[float]) -> float:
    """Compute CoV = std / mean for a list of runtimes."""
    if len(runtimes) < 2:
        return 0.0
    mean = sum(runtimes) / len(runtimes)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in runtimes) / (len(runtimes) - 1)
    return math.sqrt(variance) / mean


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear interpolation percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    lo = int(math.floor(k))
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = k - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def load_history(path: Path, max_commits: int = 20) -> dict[str, float]:
    """Build per-benchmark noise thresholds from historical data.

    Returns a dict mapping benchmark name to its adaptive CoV threshold.
    """
    commit_order: list[str] = []
    commit_set: set[str] = set()
    rows_by_commit: dict[str, list[dict]] = defaultdict(list)

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("commit_id", "")
            if not cid:
                continue
            if cid not in commit_set:
                commit_set.add(cid)
                commit_order.append(cid)
            rows_by_commit[cid].append(row)

    recent_commits = commit_order[-max_commits:] if len(commit_order) > max_commits else commit_order

    bench_covs: dict[str, list[float]] = defaultdict(list)
    for cid in recent_commits:
        for row in rows_by_commit[cid]:
            name = row.get("name", "")
            runtimes = row.get("all_runtimes")
            if not name or not runtimes or not isinstance(runtimes, list) or len(runtimes) < 2:
                continue
            cov = coefficient_of_variation([float(r) for r in runtimes])
            bench_covs[name].append(cov)

    thresholds: dict[str, float] = {}
    for name, covs in bench_covs.items():
        if len(covs) < 3:
            continue
        sorted_covs = sorted(covs)
        median_cov = percentile(sorted_covs, 0.5)
        p95_cov = percentile(sorted_covs, 0.95)
        thresholds[name] = max(p95_cov * 1.5, median_cov * 3.0, 0.02)

    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Check benchmark noise levels")
    parser.add_argument("results", help="Path to JSONL benchmark results file")
    parser.add_argument(
        "--cov-threshold",
        type=float,
        default=0.15,
        help="CoV threshold above which a benchmark is considered noisy (default: 0.15)",
    )
    parser.add_argument(
        "--noisy-fraction",
        type=float,
        default=0.25,
        help="Fraction of benchmarks that must be noisy to trigger rerun (default: 0.25)",
    )
    parser.add_argument(
        "--history",
        default="",
        help="Path to historical JSONL (.json or .json.gz) for adaptive thresholds",
    )
    parser.add_argument(
        "--history-commits",
        type=int,
        default=20,
        help="Number of recent commits to use from history (default: 20)",
    )
    args = parser.parse_args()

    # Load adaptive thresholds from history if available.
    adaptive_thresholds: dict[str, float] = {}
    if args.history:
        history_path = Path(args.history)
        if history_path.exists():
            adaptive_thresholds = load_history(history_path, max_commits=args.history_commits)
            print(f"check-bench-noise: loaded adaptive thresholds for {len(adaptive_thresholds)} benchmarks")
        else:
            print(f"check-bench-noise: history file not found: {history_path}, using fixed threshold")

    benchmarks = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            benchmarks.append(json.loads(line))

    if not benchmarks:
        print("check-bench-noise: no benchmarks found, skipping noise check")
        sys.exit(0)

    noisy: list[tuple[str, float, float]] = []  # (name, cov, threshold)
    checked = 0
    for bench in benchmarks:
        runtimes = bench.get("all_runtimes")
        if not runtimes or not isinstance(runtimes, list) or len(runtimes) < 2:
            continue
        checked += 1
        name = bench.get("name", "unknown")
        cov = coefficient_of_variation([float(r) for r in runtimes])

        threshold = adaptive_thresholds.get(name, args.cov_threshold)
        if cov > threshold:
            noisy.append((name, cov, threshold))

    if checked == 0:
        print("check-bench-noise: no benchmarks with multiple samples, skipping")
        sys.exit(0)

    noisy_fraction = len(noisy) / checked
    print(f"check-bench-noise: {len(noisy)}/{checked} benchmarks noisy")

    if noisy:
        for name, cov, threshold in sorted(noisy, key=lambda x: -x[1])[:10]:
            source = "adaptive" if name in adaptive_thresholds else "fixed"
            print(f"  {name}: CoV={cov:.3f} (threshold={threshold:.3f}, {source})")

    if noisy_fraction >= args.noisy_fraction:
        print(f"check-bench-noise: {noisy_fraction:.0%} noisy >= {args.noisy_fraction:.0%} threshold, rerun recommended")
        sys.exit(1)
    else:
        print(f"check-bench-noise: {noisy_fraction:.0%} noisy < {args.noisy_fraction:.0%} threshold, results acceptable")
        sys.exit(0)


if __name__ == "__main__":
    main()
