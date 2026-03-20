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

Usage:
    check-bench-noise.py <results.json> [--cov-threshold 0.15] [--noisy-fraction 0.25]
"""

import json
import math
import sys


def coefficient_of_variation(runtimes: list[float]) -> float:
    """Compute CoV = std / mean for a list of runtimes."""
    if len(runtimes) < 2:
        return 0.0
    mean = sum(runtimes) / len(runtimes)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in runtimes) / (len(runtimes) - 1)
    return math.sqrt(variance) / mean


def main() -> None:
    import argparse

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
    args = parser.parse_args()

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

    noisy = []
    checked = 0
    for bench in benchmarks:
        runtimes = bench.get("all_runtimes")
        if not runtimes or not isinstance(runtimes, list) or len(runtimes) < 2:
            continue
        checked += 1
        cov = coefficient_of_variation([float(r) for r in runtimes])
        if cov > args.cov_threshold:
            noisy.append((bench.get("name", "unknown"), cov))

    if checked == 0:
        print("check-bench-noise: no benchmarks with multiple samples, skipping")
        sys.exit(0)

    noisy_fraction = len(noisy) / checked
    print(f"check-bench-noise: {len(noisy)}/{checked} benchmarks noisy (CoV > {args.cov_threshold})")

    if noisy:
        for name, cov in sorted(noisy, key=lambda x: -x[1])[:10]:
            print(f"  {name}: CoV={cov:.3f}")

    if noisy_fraction >= args.noisy_fraction:
        print(f"check-bench-noise: {noisy_fraction:.0%} noisy >= {args.noisy_fraction:.0%} threshold, rerun recommended")
        sys.exit(1)
    else:
        print(f"check-bench-noise: {noisy_fraction:.0%} noisy < {args.noisy_fraction:.0%} threshold, results acceptable")
        sys.exit(0)


if __name__ == "__main__":
    main()
