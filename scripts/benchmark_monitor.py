# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark regression monitor using sample-indexed change detection.

Each commit is one logical step — wall-clock time between commits is irrelevant.
The system is organized around *checks*: each check is a function that receives
an ordered series of values and decides whether the latest point is anomalous.

Built-in checks: EWMA, CUSUM, threshold. Custom checks can be registered via
the `register_check` function.

History is stored as one JSON file per benchmark suite (e.g. tpch-nvme.json),
keeping each benchmark's data isolated and independently manageable.

Usage:
    uv run --no-project scripts/benchmark_monitor.py \
        --history-dir ./benchmark-history \
        --commits commits.json \
        --results results.json \
        --benchmark-id tpch-nvme \
        --config scripts/benchmark_monitor_config.yaml \
        [--current-commit <sha>] \
        [--webhook-url <incident.io URL>] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A detected regression or anomaly."""

    benchmark: str
    commit_id: str
    check_name: str
    current_value: float
    expected_value: float
    deviation_sigma: float
    message: str


# ---------------------------------------------------------------------------
# Check framework
# ---------------------------------------------------------------------------

class Check(ABC):
    """Base class for all series checks.

    A check receives an ordered list of values (one per logical step) and
    returns an Alert if the most recent observation is anomalous.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this check type (e.g. 'ewma', 'cusum')."""

    @abstractmethod
    def run(self, values: list[float], params: dict[str, Any]) -> Alert | None:
        """Analyze the series and optionally return an alert.

        Args:
            values: Ordered observations (one per logical step, oldest first).
            params: Check-specific parameters from the benchmark config.

        Returns:
            An Alert if the latest point is anomalous, else None.
        """


# Global check registry
_CHECK_REGISTRY: dict[str, Check] = {}


def register_check(check: Check) -> None:
    """Register a check so it can be referenced by name in config."""
    _CHECK_REGISTRY[check.name] = check


def get_check(name: str) -> Check:
    """Look up a registered check by name."""
    if name not in _CHECK_REGISTRY:
        available = ", ".join(sorted(_CHECK_REGISTRY.keys()))
        raise ValueError(f"Unknown check '{name}'. Available: {available}")
    return _CHECK_REGISTRY[name]


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------

class EWMACheck(Check):
    """Exponentially Weighted Moving Average control chart.

    Parameters (via config):
        span: int = 10           Number of observations for EWMA smoothing.
        sigma: float = 3.0       Deviation threshold (in standard deviations).
        min_observations: int = 5  Minimum history length before alerting.
    """

    @property
    def name(self) -> str:
        return "ewma"

    def run(self, values: list[float], params: dict[str, Any]) -> Alert | None:
        span = params.get("span", 10)
        sigma = params.get("sigma", 3.0)
        min_obs = params.get("min_observations", 5)

        if len(values) < min_obs:
            return None

        alpha = 2.0 / (span + 1)

        # Compute EWMA and exponentially-weighted variance up to second-to-last point
        ewma = values[0]
        ewma_var = 0.0
        for v in values[1:-1]:
            diff = v - ewma
            ewma_var = alpha * diff * diff + (1.0 - alpha) * ewma_var
            ewma = alpha * v + (1.0 - alpha) * ewma

        std = math.sqrt(ewma_var) if ewma_var > 0 else 0.0
        if std == 0:
            return None

        current = values[-1]
        deviation = (current - ewma) / std

        if abs(deviation) >= sigma:
            direction = "regression" if deviation > 0 else "improvement"
            return Alert(
                benchmark="",
                commit_id="",
                check_name=self.name,
                current_value=current,
                expected_value=ewma,
                deviation_sigma=deviation,
                message=(
                    f"EWMA {direction}: {deviation:+.2f}σ "
                    f"(value={current:.1f}, expected={ewma:.1f}±{std:.1f})"
                ),
            )
        return None


class CUSUMCheck(Check):
    """Tabular CUSUM — detects sustained shifts in the mean.

    Parameters (via config):
        drift: float = 0.5              Allowance parameter (in std devs).
        cusum_threshold: float = 5.0    Decision interval (in std devs).
        min_observations: int = 5       Minimum history length before alerting.
    """

    @property
    def name(self) -> str:
        return "cusum"

    def run(self, values: list[float], params: dict[str, Any]) -> Alert | None:
        drift = params.get("drift", 0.5)
        threshold = params.get("cusum_threshold", 5.0)
        min_obs = params.get("min_observations", 5)

        if len(values) < min_obs:
            return None

        import numpy as np

        history = np.array(values[:-1], dtype=float)
        mu = float(np.mean(history))
        std = float(np.std(history, ddof=1))
        if std == 0:
            return None

        # Normalize and run CUSUM over all values
        s_pos = 0.0
        s_neg = 0.0
        for v in values:
            z = (v - mu) / std
            s_pos = max(0.0, s_pos + z - drift)
            s_neg = max(0.0, s_neg - z - drift)

        if s_pos >= threshold or s_neg >= threshold:
            current = values[-1]
            deviation = (current - mu) / std
            direction = "regression" if s_pos >= threshold else "improvement"
            return Alert(
                benchmark="",
                commit_id="",
                check_name=self.name,
                current_value=current,
                expected_value=mu,
                deviation_sigma=deviation,
                message=(
                    f"CUSUM {direction}: S+={s_pos:.2f} S-={s_neg:.2f} "
                    f"(threshold={threshold})"
                ),
            )
        return None


class ThresholdCheck(Check):
    """Simple static threshold — alerts when value exceeds a fixed bound.

    Parameters (via config):
        max_value: float | None = None   Upper bound.
        min_value: float | None = None   Lower bound.
        min_observations: int = 1        Minimum history before alerting.
    """

    @property
    def name(self) -> str:
        return "threshold"

    def run(self, values: list[float], params: dict[str, Any]) -> Alert | None:
        max_value = params.get("max_value")
        min_value = params.get("min_value")
        min_obs = params.get("min_observations", 1)

        if len(values) < min_obs:
            return None

        current = values[-1]

        if max_value is not None and current > max_value:
            return Alert(
                benchmark="",
                commit_id="",
                check_name=self.name,
                current_value=current,
                expected_value=max_value,
                deviation_sigma=0.0,
                message=f"Threshold exceeded: {current:.1f} > {max_value:.1f}",
            )
        if min_value is not None and current < min_value:
            return Alert(
                benchmark="",
                commit_id="",
                check_name=self.name,
                current_value=current,
                expected_value=min_value,
                deviation_sigma=0.0,
                message=f"Below threshold: {current:.1f} < {min_value:.1f}",
            )
        return None


class PctChangeCheck(Check):
    """Percentage change from rolling mean — alerts on sudden jumps.

    Parameters (via config):
        window: int = 5             Rolling window size.
        pct_threshold: float = 20.0   Percent change to trigger alert.
        min_observations: int = 5   Minimum history before alerting.
    """

    @property
    def name(self) -> str:
        return "pct_change"

    def run(self, values: list[float], params: dict[str, Any]) -> Alert | None:
        window = params.get("window", 5)
        pct_threshold = params.get("pct_threshold", 20.0)
        min_obs = params.get("min_observations", 5)

        if len(values) < min_obs:
            return None

        baseline_values = values[-window - 1:-1] if len(values) > window else values[:-1]
        if not baseline_values:
            return None

        baseline_mean = sum(baseline_values) / len(baseline_values)
        if baseline_mean == 0:
            return None

        current = values[-1]
        pct_change = ((current - baseline_mean) / baseline_mean) * 100.0

        if abs(pct_change) >= pct_threshold:
            direction = "regression" if pct_change > 0 else "improvement"
            return Alert(
                benchmark="",
                commit_id="",
                check_name=self.name,
                current_value=current,
                expected_value=baseline_mean,
                deviation_sigma=pct_change / pct_threshold,
                message=(
                    f"Pct change {direction}: {pct_change:+.1f}% "
                    f"(value={current:.1f}, baseline={baseline_mean:.1f})"
                ),
            )
        return None


# Register all built-in checks
register_check(EWMACheck())
register_check(CUSUMCheck())
register_check(ThresholdCheck())
register_check(PctChangeCheck())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCheckConfig:
    """Configuration for a single check on a single benchmark."""

    check: str = "ewma"
    params: dict[str, Any] = field(default_factory=dict)
    regressions_only: bool = True


@dataclass
class BenchmarkConfig:
    """Per-benchmark configuration — one or more checks to run."""

    checks: list[BenchmarkCheckConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.checks:
            self.checks = [BenchmarkCheckConfig(
                check="ewma",
                params={"span": 10, "sigma": 3.0, "min_observations": 5},
            )]


@dataclass
class MonitorConfig:
    """Top-level monitoring configuration."""

    defaults: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    benchmarks: dict[str, BenchmarkConfig] = field(default_factory=dict)

    def config_for(self, benchmark_name: str) -> BenchmarkConfig:
        """Look up config for a benchmark, falling back to defaults."""
        if benchmark_name in self.benchmarks:
            return self.benchmarks[benchmark_name]
        for pattern, cfg in self.benchmarks.items():
            if pattern.endswith("*") and benchmark_name.startswith(pattern[:-1]):
                return cfg
            if pattern in benchmark_name:
                return cfg
        return self.defaults


def load_config(path: Path) -> MonitorConfig:
    """Load YAML config, falling back to defaults if file doesn't exist."""
    if not path.exists():
        return MonitorConfig()

    try:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except ImportError:
        raw = _parse_simple_yaml(path)

    config = MonitorConfig()

    if "defaults" in raw:
        config.defaults = _parse_benchmark_section(raw["defaults"])

    if "benchmarks" in raw:
        for name, section in raw["benchmarks"].items():
            config.benchmarks[name] = _parse_benchmark_section(
                section, fallback=config.defaults,
            )

    return config


def _parse_benchmark_section(
    raw: dict | None,
    fallback: BenchmarkConfig | None = None,
) -> BenchmarkConfig:
    """Parse a benchmark config section.

    Supports two forms:
      # Simple (single check, params at top level):
      tpch*:
        check: ewma
        sigma: 4.0

      # Multi-check:
      compress*:
        checks:
          - check: ewma
            sigma: 2.5
          - check: pct_change
            pct_threshold: 15.0
    """
    if not raw:
        return fallback or BenchmarkConfig()

    if "checks" in raw and isinstance(raw["checks"], list):
        checks = []
        for entry in raw["checks"]:
            check_name = entry.pop("check", "ewma")
            regressions_only = entry.pop("regressions_only", True)
            checks.append(BenchmarkCheckConfig(
                check=check_name,
                params=entry,
                regressions_only=regressions_only,
            ))
        return BenchmarkConfig(checks=checks)

    # Simple form: extract check name, treat rest as params
    check_name = raw.pop("check", None)
    regressions_only = raw.pop("regressions_only", True)

    if check_name:
        return BenchmarkConfig(checks=[BenchmarkCheckConfig(
            check=check_name,
            params=dict(raw),
            regressions_only=regressions_only,
        )])

    # Just parameter overrides on the default check
    if fallback and fallback.checks:
        base = fallback.checks[0]
        merged_params = {**base.params, **raw}
        return BenchmarkConfig(checks=[BenchmarkCheckConfig(
            check=base.check,
            params=merged_params,
            regressions_only=base.regressions_only,
        )])

    return BenchmarkConfig(checks=[BenchmarkCheckConfig(
        check="ewma",
        params=dict(raw),
    )])


def _parse_simple_yaml(path: Path) -> dict:
    """Minimal YAML-like parser for flat/nested dicts. Prefer real PyYAML."""
    import re

    result: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(-1, result)]

    with open(path) as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())
            while stack and stack[-1][0] >= indent:
                stack.pop()

            parent = stack[-1][1]
            m = re.match(r"(\s*)(\S+):\s*(.*)", stripped)
            if not m:
                continue

            key = m.group(2)
            value_str = m.group(3).strip()

            if value_str == "" or value_str == "{}":
                child: dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                try:
                    value: Any = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str.strip("\"'")
                parent[key] = value

    return result


# ---------------------------------------------------------------------------
# Per-benchmark history files
# ---------------------------------------------------------------------------

@dataclass
class SeriesPoint:
    """A single observation in a benchmark's history."""

    commit_id: str
    value: float


def history_file_path(history_dir: Path, benchmark_id: str) -> Path:
    """Path to the history file for a given benchmark suite."""
    safe_name = benchmark_id.replace("/", "_").replace(":", "_")
    return history_dir / f"{safe_name}.json"


def load_history(
    history_dir: Path,
    benchmark_id: str,
) -> dict[str, list[SeriesPoint]]:
    """Load per-series history from the benchmark's history file.

    Returns {series_name: [SeriesPoint, ...]} in logical order.
    """
    path = history_file_path(history_dir, benchmark_id)
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    result: dict[str, list[SeriesPoint]] = {}
    for series_name, points in data.get("series", {}).items():
        result[series_name] = [
            SeriesPoint(commit_id=p["commit_id"], value=p["value"])
            for p in points
        ]
    return result


def save_history(
    history_dir: Path,
    benchmark_id: str,
    history: dict[str, list[SeriesPoint]],
    commit_order: list[str],
) -> None:
    """Save per-series history to the benchmark's history file."""
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_file_path(history_dir, benchmark_id)

    data = {
        "benchmark_id": benchmark_id,
        "commit_count": len(commit_order),
        "latest_commit": commit_order[-1] if commit_order else None,
        "series": {
            name: [{"commit_id": p.commit_id, "value": p.value} for p in points]
            for name, points in history.items()
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def ingest_results(
    history: dict[str, list[SeriesPoint]],
    results_path: Path,
    current_commit: str,
) -> dict[str, list[SeriesPoint]]:
    """Append new results from a benchmark run into the history.

    Each line in results_path is a JSON object with at least 'name' and 'value'.
    """
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = obj.get("name", "")
            value = obj.get("value")
            if not name or value is None:
                continue

            series = history.setdefault(name, [])
            # Avoid duplicate entries for the same commit
            if series and series[-1].commit_id == current_commit:
                series[-1].value = float(value)
            else:
                series.append(SeriesPoint(commit_id=current_commit, value=float(value)))

    return history


def load_commit_order(commits_path: Path) -> list[str]:
    """Load commit ordering from commits.json (one JSON object per line)."""
    commits = []
    with open(commits_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            commits.append(obj["id"])
    return commits


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    history: dict[str, list[SeriesPoint]],
    current_commit: str,
    config: MonitorConfig,
) -> list[Alert]:
    """Run all configured checks on each series for the current commit."""
    alerts = []

    for series_name, series in history.items():
        if not series or series[-1].commit_id != current_commit:
            continue

        bench_config = config.config_for(series_name)
        values = [p.value for p in series]

        for check_config in bench_config.checks:
            check = get_check(check_config.check)
            alert = check.run(values, check_config.params)

            if alert is not None:
                alert.benchmark = series_name
                alert.commit_id = current_commit

                if check_config.regressions_only and alert.deviation_sigma <= 0:
                    continue
                alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def send_incident_io_alert(
    alerts: list[Alert],
    webhook_url: str,
    commit_id: str,
    benchmark_id: str,
) -> None:
    """Post regression alerts to incident.io via the alert source API."""
    for alert in alerts:
        payload = {
            "title": f"Benchmark regression: {alert.benchmark}",
            "description": alert.message,
            "deduplication_key": f"bench-{benchmark_id}-{alert.benchmark}-{commit_id[:12]}",
            "metadata": {
                "benchmark_suite": benchmark_id,
                "series": alert.benchmark,
                "commit": commit_id,
                "check": alert.check_name,
                "current_value": alert.current_value,
                "expected_value": alert.expected_value,
                "deviation_sigma": alert.deviation_sigma,
            },
        }

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"  Alert sent for {alert.benchmark}: HTTP {resp.status}")
        except Exception as e:
            print(f"  Failed to send alert for {alert.benchmark}: {e}", file=sys.stderr)


def print_alerts(alerts: list[Alert]) -> None:
    """Print alerts to stdout in a human-readable format."""
    if not alerts:
        print("No regressions detected.")
        return

    print(f"\n{'='*60}")
    print(f"REGRESSION ALERTS ({len(alerts)} detected)")
    print(f"{'='*60}\n")

    for alert in alerts:
        print(f"  [{alert.check_name.upper()}] {alert.benchmark}")
        print(f"    {alert.message}")
        print(f"    commit: {alert.commit_id[:12]}")
        print()


def write_github_output(alerts: list[Alert]) -> None:
    """Write alerts as GitHub Actions outputs and job summary."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"alert_count={len(alerts)}\n")
            f.write(f"has_alerts={'true' if alerts else 'false'}\n")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            if alerts:
                f.write("## Benchmark Regression Alerts\n\n")
                f.write(f"**{len(alerts)} regression(s) detected**\n\n")
                f.write("| Benchmark | Check | Deviation | Current | Expected |\n")
                f.write("|-----------|-------|-----------|---------|----------|\n")
                for alert in alerts:
                    f.write(
                        f"| `{alert.benchmark}` "
                        f"| {alert.check_name} "
                        f"| {alert.deviation_sigma:+.2f}σ "
                        f"| {alert.current_value:.1f} "
                        f"| {alert.expected_value:.1f} |\n"
                    )
            else:
                f.write("## Benchmark Monitor\n\nNo regressions detected.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark regression monitor (sample-indexed, not time-indexed)",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        required=True,
        help="Directory containing per-benchmark history JSON files",
    )
    parser.add_argument(
        "--commits",
        type=Path,
        required=True,
        help="Path to commits.json (newline-delimited commit metadata)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results.json from the current benchmark run",
    )
    parser.add_argument(
        "--benchmark-id",
        type=str,
        required=True,
        help="Benchmark suite identifier (e.g. tpch-nvme, compress-bench)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/benchmark_monitor_config.yaml"),
        help="Path to monitor config YAML",
    )
    parser.add_argument(
        "--current-commit",
        type=str,
        help="The commit SHA to analyze (default: last in commits.json)",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="incident.io alert source webhook URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print alerts but don't send webhooks",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    commit_order = load_commit_order(args.commits)

    current_commit = args.current_commit
    if not current_commit:
        if commit_order:
            current_commit = commit_order[-1]
        else:
            print("No commits found.", file=sys.stderr)
            sys.exit(1)

    print(f"Benchmark suite: {args.benchmark_id}")
    print(f"Analyzing commit: {current_commit[:12]}")
    print(f"Commit history depth: {len(commit_order)}")

    # Load existing history, ingest new results, save back
    history = load_history(args.history_dir, args.benchmark_id)
    history = ingest_results(history, args.results, current_commit)
    save_history(args.history_dir, args.benchmark_id, history, commit_order)

    series_count = len(history)
    point_counts = [len(pts) for pts in history.values()]
    max_depth = max(point_counts) if point_counts else 0
    print(f"Series tracked: {series_count} (max depth: {max_depth})")

    # Run all checks
    alerts = analyze(history, current_commit, config)
    print_alerts(alerts)
    write_github_output(alerts)

    if alerts and args.webhook_url and not args.dry_run:
        print("Sending alerts to incident.io...")
        send_incident_io_alert(alerts, args.webhook_url, current_commit, args.benchmark_id)

    if alerts:
        sys.exit(1)


if __name__ == "__main__":
    main()
