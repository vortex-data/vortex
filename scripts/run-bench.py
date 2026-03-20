# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark runner with automatic noise-aware retry.

Runs SQL benchmark binaries (datafusion-bench, duckdb-bench, lance-bench) for
the given engine:format targets, checks per-run noise via coefficient of
variation (CoV), and reruns noisy engines keeping the cleanest results.

Usage:
    run-bench.py <subcommand> <targets> [options]

Example:
    run-bench.py tpch "datafusion:parquet,datafusion:vortex,duckdb:parquet" \
        --scale-factor 1.0 --iterations 5 --max-retries 2

This script is a drop-in replacement for .github/scripts/run-sql-bench.sh
with smarter noise handling: it keeps the least-noisy run across retries
instead of always using the last attempt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Noise analysis
# ---------------------------------------------------------------------------

def coefficient_of_variation(runtimes: list[float]) -> float:
    """CoV = std / mean for a list of runtimes."""
    if len(runtimes) < 2:
        return 0.0
    mean = sum(runtimes) / len(runtimes)
    if mean <= 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in runtimes) / (len(runtimes) - 1)
    return math.sqrt(variance) / mean


@dataclass
class NoiseReport:
    """Summary of noise levels for a single engine run."""
    total: int
    noisy_count: int
    max_cov: float
    mean_cov: float
    noisy_benchmarks: list[tuple[str, float]]

    @property
    def noisy_fraction(self) -> float:
        return self.noisy_count / self.total if self.total > 0 else 0.0

    def is_acceptable(self, noisy_fraction_threshold: float) -> bool:
        return self.noisy_fraction < noisy_fraction_threshold


def analyze_noise(results_path: Path, cov_threshold: float) -> NoiseReport:
    """Analyze a JSONL results file for noisy benchmarks."""
    benchmarks = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                benchmarks.append(json.loads(line))

    noisy: list[tuple[str, float]] = []
    covs: list[float] = []
    checked = 0

    for bench in benchmarks:
        runtimes = bench.get("all_runtimes")
        if not runtimes or not isinstance(runtimes, list) or len(runtimes) < 2:
            continue
        checked += 1
        cov = coefficient_of_variation([float(r) for r in runtimes])
        covs.append(cov)
        if cov > cov_threshold:
            noisy.append((bench.get("name", "unknown"), cov))

    return NoiseReport(
        total=checked,
        noisy_count=len(noisy),
        max_cov=max(covs) if covs else 0.0,
        mean_cov=sum(covs) / len(covs) if covs else 0.0,
        noisy_benchmarks=sorted(noisy, key=lambda x: -x[1]),
    )


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineRun:
    """Configuration for running a single benchmark engine."""
    label: str
    binary: str
    formats: str
    output_file: str
    extra_args: list[str]


def parse_targets(targets: str) -> tuple[str, str, bool]:
    """Extract per-engine format lists from the comma-separated targets string.

    Returns (df_formats, ddb_formats, has_lance).
    """
    parts = targets.split(",")

    df_formats = ",".join(
        p.removeprefix("datafusion:")
        for p in parts
        if p.startswith("datafusion:") and not p.endswith(":lance")
    )
    ddb_formats = ",".join(
        p.removeprefix("duckdb:")
        for p in parts
        if p.startswith("duckdb:")
    )
    has_lance = any(p == "datafusion:lance" for p in parts)

    return df_formats, ddb_formats, has_lance


def build_engine_runs(
    subcommand: str,
    targets: str,
    opts: list[str],
    is_remote: bool,
    benchmark_id: str,
) -> list[EngineRun]:
    """Build the list of engine runs from targets and options."""
    df_formats, ddb_formats, has_lance = parse_targets(targets)

    if is_remote and has_lance:
        print(
            f"ERROR: Lance format is not supported for remote storage benchmarks. "
            f"Remove 'datafusion:lance' from targets for benchmark '{benchmark_id or 'unknown'}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    runs: list[EngineRun] = []

    if df_formats:
        runs.append(EngineRun(
            label="datafusion",
            binary="target/release_debug/datafusion-bench",
            formats=df_formats,
            output_file="df-results.json",
            extra_args=[subcommand, "-d", "gh-json", "--formats", df_formats] + opts,
        ))

    if ddb_formats:
        runs.append(EngineRun(
            label="duckdb",
            binary="target/release_debug/duckdb-bench",
            formats=ddb_formats,
            output_file="ddb-results.json",
            extra_args=[
                subcommand, "-d", "gh-json", "--formats", ddb_formats,
                "--delete-duckdb-database",
            ] + opts,
        ))

    lance_binary = Path("target/release_debug/lance-bench")
    if not is_remote and has_lance and lance_binary.exists():
        runs.append(EngineRun(
            label="lance",
            binary=str(lance_binary),
            formats="lance",
            output_file="lance-results.json",
            extra_args=[subcommand, "-d", "gh-json"] + opts,
        ))

    return runs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_engine(
    engine: EngineRun,
    taskset_script: str | None,
) -> None:
    """Run a single engine benchmark, writing results to engine.output_file."""
    cmd: list[str] = []
    if taskset_script:
        cmd.extend(["bash", taskset_script])
    cmd.append(engine.binary)
    cmd.extend(engine.extra_args)
    cmd.extend(["-o", engine.output_file])

    print(f"  running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_engine_with_retry(
    engine: EngineRun,
    taskset_script: str | None,
    max_retries: int,
    cov_threshold: float,
    noisy_fraction_threshold: float,
) -> NoiseReport:
    """Run an engine, check noise, retry if needed, keep the cleanest run."""
    best_report: NoiseReport | None = None
    best_results: str | None = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        if attempt > 0:
            print(f"  retry {attempt}/{max_retries} for {engine.label}")

        run_engine(engine, taskset_script)
        report = analyze_noise(Path(engine.output_file), cov_threshold)

        print(
            f"  {engine.label}: {report.noisy_count}/{report.total} noisy "
            f"(mean CoV={report.mean_cov:.3f}, max CoV={report.max_cov:.3f})"
        )

        # Keep the run with the fewest noisy benchmarks (tie-break on mean CoV).
        if (
            best_report is None
            or report.noisy_count < best_report.noisy_count
            or (report.noisy_count == best_report.noisy_count and report.mean_cov < best_report.mean_cov)
        ):
            best_report = report
            best_results = Path(engine.output_file).read_text()

        if report.is_acceptable(noisy_fraction_threshold):
            break

        if report.noisy_benchmarks:
            for name, cov in report.noisy_benchmarks[:5]:
                print(f"    {name}: CoV={cov:.3f}")
    else:
        print(
            f"  {engine.label}: still noisy after {total_attempts} attempts, "
            f"using best run ({best_report.noisy_count}/{best_report.total} noisy)"
        )

    # Write back the best results if we didn't end on the best run.
    if best_results is not None:
        Path(engine.output_file).write_text(best_results)

    assert best_report is not None
    return best_report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_opts(args: argparse.Namespace) -> list[str]:
    """Build the shared options list from parsed arguments."""
    opts: list[str] = []
    if args.iterations:
        opts.extend(["-i", str(args.iterations)])
    if args.scale_factor:
        opts.extend(["--opt", f"scale-factor={args.scale_factor}"])
    if args.remote_storage:
        opts.extend(["--opt", f"remote-data-dir={args.remote_storage}"])
    return opts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SQL benchmarks with noise-aware retry",
    )
    parser.add_argument("subcommand", help="Benchmark subcommand (tpch, clickbench, tpcds, ...)")
    parser.add_argument("targets", help="Comma-separated engine:format pairs")
    parser.add_argument("--scale-factor", help="Scale factor for the benchmark")
    parser.add_argument("--iterations", type=int, help="Number of iterations per query")
    parser.add_argument("--remote-storage", help="Remote storage URL (e.g. s3://...)")
    parser.add_argument("--benchmark-id", default="", help="Benchmark ID for error messages")
    parser.add_argument(
        "--max-retries", type=int,
        default=int(os.environ.get("BENCH_MAX_RETRIES", "2")),
        help="Max retry attempts for noisy runs (default: 2)",
    )
    parser.add_argument(
        "--cov-threshold", type=float,
        default=float(os.environ.get("BENCH_COV_THRESHOLD", "0.15")),
        help="Per-benchmark CoV threshold to flag as noisy (default: 0.15)",
    )
    parser.add_argument(
        "--noisy-fraction", type=float,
        default=float(os.environ.get("BENCH_NOISY_FRACTION", "0.25")),
        help="Fraction of noisy benchmarks that triggers a rerun (default: 0.25)",
    )
    parser.add_argument(
        "--taskset-script",
        default=os.environ.get("BENCH_TASKSET_SCRIPT", ""),
        help="Path to CPU-pinning wrapper script (e.g. scripts/bench-taskset.sh)",
    )
    parser.add_argument(
        "-o", "--output", default="results.json",
        help="Combined output file (default: results.json)",
    )
    args = parser.parse_args()

    is_remote = bool(args.remote_storage)
    opts = build_opts(args)
    taskset = args.taskset_script or None

    engines = build_engine_runs(
        subcommand=args.subcommand,
        targets=args.targets,
        opts=opts,
        is_remote=is_remote,
        benchmark_id=args.benchmark_id,
    )

    if not engines:
        print("No engines to run for the given targets.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(engines)} engine(s) with up to {args.max_retries} retries for noise")

    reports: list[tuple[str, NoiseReport]] = []
    for engine in engines:
        print(f"\n{'='*60}")
        print(f"Engine: {engine.label} (formats: {engine.formats})")
        print(f"{'='*60}")

        report = run_engine_with_retry(
            engine=engine,
            taskset_script=taskset,
            max_retries=args.max_retries,
            cov_threshold=args.cov_threshold,
            noisy_fraction_threshold=args.noisy_fraction,
        )
        reports.append((engine.label, report))

    # Combine all engine results into the final output file.
    with open(args.output, "w") as out:
        for engine in engines:
            path = Path(engine.output_file)
            if path.exists():
                out.write(path.read_text())

    # Print summary.
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    total_noisy = 0
    total_checked = 0
    for label, report in reports:
        status = "OK" if report.is_acceptable(args.noisy_fraction) else "NOISY"
        print(
            f"  {label}: {report.noisy_count}/{report.total} noisy "
            f"(mean CoV={report.mean_cov:.3f}) [{status}]"
        )
        total_noisy += report.noisy_count
        total_checked += report.total

    overall_fraction = total_noisy / total_checked if total_checked > 0 else 0.0
    print(f"\n  Overall: {total_noisy}/{total_checked} noisy ({overall_fraction:.0%})")
    print(f"  Results written to {args.output}")


if __name__ == "__main__":
    main()
