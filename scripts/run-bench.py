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

When historical data is available (via --history), each benchmark's CoV is
compared against its own historical distribution rather than a fixed threshold.
A benchmark is flagged noisy only when its CoV exceeds its expected range,
so inherently-variable benchmarks (e.g. S3) aren't penalized.

Usage:
    run-bench.py <subcommand> <targets> [options]

Example:
    run-bench.py tpch "datafusion:parquet,datafusion:vortex,duckdb:parquet" \\
        --scale-factor 1.0 --iterations 5 --max-retries 2

    # With historical noise profiles:
    run-bench.py tpch "datafusion:parquet,datafusion:vortex" \\
        --history data.json.gz --max-retries 2
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
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


@dataclass
class NoiseProfile:
    """Historical noise statistics for a single benchmark name."""
    median_cov: float
    p90_cov: float
    p95_cov: float
    sample_count: int

    def threshold(self, headroom: float = 1.5) -> float:
        """Adaptive threshold: p95 of historical CoV with headroom multiplier.

        Uses max(p95 * headroom, median * 3) to handle benchmarks with very
        tight distributions while still allowing for natural variation.
        """
        return max(self.p95_cov * headroom, self.median_cov * 3.0, 0.02)


@dataclass
class NoiseProfiles:
    """Collection of per-benchmark noise profiles from historical data."""
    profiles: dict[str, NoiseProfile] = field(default_factory=dict)
    fallback_threshold: float = 0.15

    def threshold_for(self, bench_name: str) -> float:
        """Get the noise threshold for a benchmark, falling back to the global default."""
        if bench_name in self.profiles:
            return self.profiles[bench_name].threshold()
        return self.fallback_threshold

    def describe(self, bench_name: str) -> str:
        """Human-readable description of the threshold source."""
        if bench_name in self.profiles:
            p = self.profiles[bench_name]
            t = p.threshold()
            return f"adaptive={t:.3f} (median={p.median_cov:.3f}, p95={p.p95_cov:.3f}, n={p.sample_count})"
        return f"fixed={self.fallback_threshold:.3f}"


def load_history(path: Path, max_commits: int = 20) -> NoiseProfiles:
    """Build per-benchmark noise profiles from historical JSONL data.

    Reads the most recent `max_commits` commits from the history file and
    computes CoV statistics per benchmark name.
    """
    # Read all rows, grouping by commit then by benchmark name.
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

    # Take only the most recent commits (they appear in file order = chronological).
    recent_commits = commit_order[-max_commits:] if len(commit_order) > max_commits else commit_order

    # Compute per-benchmark CoV across recent commits.
    bench_covs: dict[str, list[float]] = defaultdict(list)
    for cid in recent_commits:
        for row in rows_by_commit[cid]:
            name = row.get("name", "")
            runtimes = row.get("all_runtimes")
            if not name or not runtimes or not isinstance(runtimes, list) or len(runtimes) < 2:
                continue
            cov = coefficient_of_variation([float(r) for r in runtimes])
            bench_covs[name].append(cov)

    # Build profiles.
    profiles: dict[str, NoiseProfile] = {}
    for name, covs in bench_covs.items():
        if len(covs) < 3:
            # Not enough data for a reliable profile.
            continue
        sorted_covs = sorted(covs)
        profiles[name] = NoiseProfile(
            median_cov=percentile(sorted_covs, 0.5),
            p90_cov=percentile(sorted_covs, 0.9),
            p95_cov=percentile(sorted_covs, 0.95),
            sample_count=len(covs),
        )

    return NoiseProfiles(profiles=profiles)


def download_history(url: str) -> Path | None:
    """Download the history file from S3, returning local path or None on failure."""
    suffix = ".json.gz" if url.endswith(".gz") else ".json"
    tmp = Path(tempfile.mktemp(suffix=suffix))

    # Try the s3-download.py helper first, then fall back to aws cli.
    repo_root = Path(__file__).resolve().parent.parent
    download_script = repo_root / "scripts" / "s3-download.py"

    if download_script.exists():
        cmd = ["python3", str(download_script), url, str(tmp), "--no-sign-request", "--max-retries", "3"]
    else:
        cmd = ["aws", "s3", "cp", url, str(tmp), "--no-sign-request"]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  warning: could not download history from {url}: {e}", file=sys.stderr)
        return None


@dataclass
class BenchNoise:
    """Noise assessment for a single benchmark within a run."""
    name: str
    cov: float
    threshold: float

    @property
    def is_noisy(self) -> bool:
        return self.cov > self.threshold


@dataclass
class NoiseReport:
    """Summary of noise levels for a single engine run."""
    total: int
    noisy_count: int
    max_cov: float
    mean_cov: float
    details: list[BenchNoise]

    @property
    def noisy_fraction(self) -> float:
        return self.noisy_count / self.total if self.total > 0 else 0.0

    def is_acceptable(self, noisy_fraction_threshold: float) -> bool:
        return self.noisy_fraction < noisy_fraction_threshold

    @property
    def noisy_benchmarks(self) -> list[tuple[str, float]]:
        return [(d.name, d.cov) for d in self.details if d.is_noisy]


def analyze_noise(
    results_path: Path,
    profiles: NoiseProfiles,
    fixed_cov_threshold: float,
) -> NoiseReport:
    """Analyze a JSONL results file for noisy benchmarks.

    Uses per-benchmark adaptive thresholds when historical profiles are
    available, falling back to the fixed threshold otherwise.
    """
    benchmarks = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                benchmarks.append(json.loads(line))

    details: list[BenchNoise] = []
    covs: list[float] = []

    for bench in benchmarks:
        runtimes = bench.get("all_runtimes")
        if not runtimes or not isinstance(runtimes, list) or len(runtimes) < 2:
            continue
        name = bench.get("name", "unknown")
        cov = coefficient_of_variation([float(r) for r in runtimes])
        covs.append(cov)

        # Use adaptive threshold if we have history for this benchmark,
        # otherwise fall back to the fixed threshold.
        if profiles.profiles:
            threshold = profiles.threshold_for(name)
        else:
            threshold = fixed_cov_threshold

        details.append(BenchNoise(name=name, cov=cov, threshold=threshold))

    noisy_count = sum(1 for d in details if d.is_noisy)
    return NoiseReport(
        total=len(details),
        noisy_count=noisy_count,
        max_cov=max(covs) if covs else 0.0,
        mean_cov=sum(covs) / len(covs) if covs else 0.0,
        details=details,
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
    profiles: NoiseProfiles,
    fixed_cov_threshold: float,
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
        report = analyze_noise(Path(engine.output_file), profiles, fixed_cov_threshold)

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
                desc = profiles.describe(name)
                print(f"    {name}: CoV={cov:.3f} ({desc})")
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

HISTORY_S3_URL = "s3://vortex-ci-benchmark-results/data.json.gz"


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
        help="Per-benchmark CoV threshold when no history available (default: 0.15)",
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
        "--history",
        default=os.environ.get("BENCH_HISTORY", ""),
        help="Path to historical JSONL (.json or .json.gz) for adaptive noise thresholds. "
             "Can also be an s3:// URL which will be downloaded automatically.",
    )
    parser.add_argument(
        "--history-commits", type=int,
        default=int(os.environ.get("BENCH_HISTORY_COMMITS", "20")),
        help="Number of recent commits to use from history (default: 20)",
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

    # Load historical noise profiles if available.
    profiles = NoiseProfiles(fallback_threshold=args.cov_threshold)
    history_source = args.history
    history_tmp: Path | None = None

    if history_source:
        if history_source.startswith("s3://"):
            print(f"Downloading history from {history_source}...")
            history_tmp = download_history(history_source)
            if history_tmp:
                history_path = history_tmp
            else:
                history_path = None
        else:
            history_path = Path(history_source)
            if not history_path.exists():
                print(f"  warning: history file not found: {history_path}", file=sys.stderr)
                history_path = None

        if history_path:
            print(f"Loading noise profiles from {history_path} (last {args.history_commits} commits)...")
            profiles = load_history(history_path, max_commits=args.history_commits)
            profiles.fallback_threshold = args.cov_threshold
            print(f"  loaded profiles for {len(profiles.profiles)} benchmarks")

    print(f"Running {len(engines)} engine(s) with up to {args.max_retries} retries for noise")
    if profiles.profiles:
        print(f"  using adaptive thresholds from {len(profiles.profiles)} historical benchmarks")
    else:
        print(f"  using fixed CoV threshold: {args.cov_threshold}")

    reports: list[tuple[str, NoiseReport]] = []
    for engine in engines:
        print(f"\n{'='*60}")
        print(f"Engine: {engine.label} (formats: {engine.formats})")
        print(f"{'='*60}")

        report = run_engine_with_retry(
            engine=engine,
            taskset_script=taskset,
            max_retries=args.max_retries,
            profiles=profiles,
            fixed_cov_threshold=args.cov_threshold,
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

    # Cleanup.
    if history_tmp and history_tmp.exists():
        history_tmp.unlink()


if __name__ == "__main__":
    main()
