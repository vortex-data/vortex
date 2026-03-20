# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Find benchmark regressions. Writes ndjson to stdout.

    # Local:
    uv run scripts/benchmark_check.py \
        --history-dir ./benchmark-history \
        --commits commits.json \
        --results results.json \
        --benchmark-id tpch-nvme | jq .

    # CI — pipe into alerting:
    uv run scripts/benchmark_check.py ... | \
        uv run scripts/benchmark_alert.py --webhook-url "$URL"
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================================
# METRIC TYPES
# ============================================================================

@dataclass(frozen=True, slots=True)
class Ewma:
    """EWMA control chart. Alerts when latest value deviates from the
    exponentially weighted mean by more than `sigma` standard deviations."""
    pattern: str
    span: int = 10
    sigma: float = 3.0
    min_observations: int = 5


@dataclass(frozen=True, slots=True)
class Cusum:
    """Tabular CUSUM. Detects sustained shifts in the mean."""
    pattern: str
    drift: float = 0.5
    cusum_threshold: float = 5.0
    min_observations: int = 5


@dataclass(frozen=True, slots=True)
class Threshold:
    """Static upper/lower bound."""
    pattern: str
    max_value: float | None = None
    min_value: float | None = None
    min_observations: int = 1


@dataclass(frozen=True, slots=True)
class PctChange:
    """Percentage change from a rolling window mean."""
    pattern: str
    window: int = 5
    pct_threshold: float = 20.0
    min_observations: int = 5


Metric = Ewma | Cusum | Threshold | PctChange


# ============================================================================
# METRICS — edit this to change what gets checked
# ============================================================================
#
# Patterns match against the --benchmark-id passed to this script.
# Trailing * = prefix match, otherwise substring match.
# First match wins. Unmatched benchmarks get DEFAULT_METRIC.
#
# Parameters were tuned against 4M real data points from S3 (2794 commits,
# 3094 series). Target: ~1% alert rate per series — roughly 1–2 alerts per
# week given daily merges, low enough to avoid fatigue, high enough to catch
# real regressions.
#
# Benchmark ID           CV% p50  MaxJ% p50  Check   Params                    Alert rate
# ─────────────────────  ───────  ─────────  ──────  ────────────────────────  ──────────
# random-access-bench      5.4%      32.8%   Ewma    span=8  σ=5.0 min=10      ~0.97%
# compress-bench          22.2%      51.8%   Ewma    span=30 σ=4.0 min=10      ~1.03%
# clickbench-nvme          8.8%      40.8%   Ewma    span=30 σ=5.0 min=10      ~1.01%
# tpch-nvme               71.8%     654.7%   Ewma    span=30 σ=5.0 min=10      ~1.92%
# tpch-s3                 74.0%    3606.6%   Ewma    span=20 σ=5.0 min=10      ~0.99%
# tpcds-nvme              10.4%      35.5%   Ewma    span=30 σ=5.0 min=10      ~1.02%
# fineweb-nvme             2.5%      12.2%   Ewma    span=8  σ=5.0 min=5       ~1.30%
# fineweb-s3              51.3%     962.5%   Ewma    span=30 σ=5.0 min=8       ~1.05%
# statpopgen               9.8%      34.4%   Ewma    span=30 σ=5.0 min=5       ~1.00%
# polarsignals             9.8%      35.1%   Ewma    span=20 σ=4.0 min=10      ~1.00%

DEFAULT_METRIC: Metric = Ewma(pattern="*", span=30, sigma=5.0, min_observations=10)

METRICS: list[Metric] = [
    # S3 benchmarks: extreme variance (CV 50-74%, max jumps 1000-3600%)
    # from network jitter. Long span + high sigma to absorb noise.
    Ewma(pattern="*-s3*",          span=20, sigma=5.0, min_observations=10),
    # Compression: high CV (22%) from ratio metrics and timing variance.
    # Long span smooths out the noise.
    Ewma(pattern="compress*",      span=30, sigma=4.0, min_observations=10),
    # Random access: low CV (5.4%) — most stable suite. Short span reacts
    # faster since the data is clean.
    Ewma(pattern="random-access*", span=8,  sigma=5.0, min_observations=10),
    # FineWeb NVMe: very stable (CV 2.5%), short history (72 pts).
    # Short span, lower min_obs to work with limited data.
    Ewma(pattern="fineweb",        span=8,  sigma=5.0, min_observations=5),
    # Polarsignals: moderate noise (CV 9.8%), only 10 series.
    Ewma(pattern="polarsignals",   span=20, sigma=4.0, min_observations=10),
]


# ============================================================================
# DATA
# ============================================================================

@dataclass(frozen=True, slots=True)
class Point:
    commit_id: str
    value: float


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Output from a check function — just the numbers, no identity."""
    check_name: str
    current: float
    expected: float
    sigma: float
    message: str


# ============================================================================
# CHECKS
# ============================================================================

def _check_ewma(values: list[float], m: Ewma) -> CheckResult | None:
    if len(values) < m.min_observations:
        return None
    alpha = 2.0 / (m.span + 1)
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
    dev = (current - ewma) / std
    if abs(dev) < m.sigma:
        return None
    direction = "regression" if dev > 0 else "improvement"
    return CheckResult(
        check_name="ewma", current=current, expected=ewma, sigma=dev,
        message=f"EWMA {direction}: {dev:+.2f}σ (value={current:.1f}, expected={ewma:.1f}±{std:.1f})",
    )


def _check_cusum(values: list[float], m: Cusum) -> CheckResult | None:
    if len(values) < m.min_observations:
        return None
    hist = np.array(values[:-1], dtype=float)
    mu, std = float(np.mean(hist)), float(np.std(hist, ddof=1))
    if std == 0:
        return None
    s_pos = s_neg = 0.0
    for v in values:
        z = (v - mu) / std
        s_pos = max(0.0, s_pos + z - m.drift)
        s_neg = max(0.0, s_neg - z - m.drift)
    if s_pos < m.cusum_threshold and s_neg < m.cusum_threshold:
        return None
    current = values[-1]
    dev = (current - mu) / std
    direction = "regression" if s_pos >= m.cusum_threshold else "improvement"
    return CheckResult(
        check_name="cusum", current=current, expected=mu, sigma=dev,
        message=f"CUSUM {direction}: S+={s_pos:.2f} S-={s_neg:.2f} (threshold={m.cusum_threshold})",
    )


def _check_threshold(values: list[float], m: Threshold) -> CheckResult | None:
    if len(values) < m.min_observations:
        return None
    current = values[-1]
    if m.max_value is not None and current > m.max_value:
        return CheckResult(
            check_name="threshold", current=current, expected=m.max_value,
            sigma=0.0, message=f"Exceeded: {current:.1f} > {m.max_value:.1f}",
        )
    if m.min_value is not None and current < m.min_value:
        return CheckResult(
            check_name="threshold", current=current, expected=m.min_value,
            sigma=0.0, message=f"Below: {current:.1f} < {m.min_value:.1f}",
        )
    return None


def _check_pct_change(values: list[float], m: PctChange) -> CheckResult | None:
    if len(values) < m.min_observations:
        return None
    baseline = values[-m.window - 1:-1] if len(values) > m.window else values[:-1]
    if not baseline:
        return None
    mean = sum(baseline) / len(baseline)
    if mean == 0:
        return None
    current = values[-1]
    pct = ((current - mean) / mean) * 100.0
    if abs(pct) < m.pct_threshold:
        return None
    direction = "regression" if pct > 0 else "improvement"
    return CheckResult(
        check_name="pct_change", current=current, expected=mean,
        sigma=pct / m.pct_threshold,
        message=f"Pct change {direction}: {pct:+.1f}% (value={current:.1f}, baseline={mean:.1f})",
    )


def run_check(values: list[float], metric: Metric) -> CheckResult | None:
    """Dispatch to the right check based on metric type."""
    match metric:
        case Ewma():       return _check_ewma(values, metric)
        case Cusum():      return _check_cusum(values, metric)
        case Threshold():  return _check_threshold(values, metric)
        case PctChange():  return _check_pct_change(values, metric)


# ============================================================================
# FILTER
# ============================================================================

def metric_for(benchmark_id: str) -> Metric:
    """First matching metric from METRICS, or DEFAULT_METRIC."""
    for m in METRICS:
        if m.pattern.endswith("*") and benchmark_id.startswith(m.pattern[:-1]):
            return m
        if not m.pattern.endswith("*") and m.pattern in benchmark_id:
            return m
    return DEFAULT_METRIC


# ============================================================================
# HISTORY
# ============================================================================

def _history_path(history_dir: Path, benchmark_id: str) -> Path:
    safe = benchmark_id.replace("/", "_").replace(":", "_")
    return history_dir / f"{safe}.json"


def load_history(history_dir: Path, benchmark_id: str) -> dict[str, list[Point]]:
    path = _history_path(history_dir, benchmark_id)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        name: [Point(p["commit_id"], p["value"]) for p in points]
        for name, points in data.get("series", {}).items()
    }


def save_history(history_dir: Path, benchmark_id: str,
                 history: dict[str, list[Point]], current_commit: str) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = _history_path(history_dir, benchmark_id)
    data = {
        "benchmark_id": benchmark_id,
        "latest_commit": current_commit,
        "series": {
            name: [{"commit_id": p.commit_id, "value": p.value} for p in points]
            for name, points in history.items()
        },
    }
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def ingest(history: dict[str, list[Point]], results_path: Path,
           commit_id: str) -> dict[str, list[Point]]:
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name, value = obj.get("name", ""), obj.get("value")
            if not name or value is None:
                continue
            series = history.setdefault(name, [])
            if series and series[-1].commit_id == commit_id:
                series[-1] = Point(commit_id, float(value))
            else:
                series.append(Point(commit_id, float(value)))
    return history


def load_commit_order(path: Path) -> list[str]:
    commits = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                commits.append(json.loads(line)["id"])
    return commits


# ============================================================================
# COMPUTE
# ============================================================================

def compute_and_emit(history: dict[str, list[Point]], commit_id: str,
                     benchmark_id: str, regressions_only: bool) -> int:
    """Run checks, write ndjson to stdout. Returns alert count."""
    metric = metric_for(benchmark_id)
    count = 0
    for series_name, points in history.items():
        if not points or points[-1].commit_id != commit_id:
            continue
        values = [p.value for p in points]
        result = run_check(values, metric)
        if result is None:
            continue
        if regressions_only and result.sigma <= 0:
            continue
        json.dump({
            "benchmark_id": benchmark_id,
            "series": series_name,
            "commit_id": commit_id,
            "check": result.check_name,
            "current": result.current,
            "expected": result.expected,
            "sigma": result.sigma,
            "message": result.message,
        }, sys.stdout)
        sys.stdout.write("\n")
        count += 1
    return count


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Find benchmark regressions. Writes ndjson to stdout.",
    )
    p.add_argument("--history-dir", type=Path, required=True)
    p.add_argument("--commits", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--benchmark-id", type=str, required=True)
    p.add_argument("--current-commit", type=str, default=None)
    p.add_argument("--all-alerts", action="store_true",
                   help="Include improvements, not just regressions")
    args = p.parse_args()

    commit_order = load_commit_order(args.commits)
    current_commit = args.current_commit
    if not current_commit:
        if not commit_order:
            print("error: no commits found", file=sys.stderr)
            sys.exit(1)
        current_commit = commit_order[-1]

    history = load_history(args.history_dir, args.benchmark_id)
    history = ingest(history, args.results, current_commit)
    save_history(args.history_dir, args.benchmark_id, history, current_commit)

    series_count = len(history)
    max_depth = max((len(pts) for pts in history.values()), default=0)
    metric = metric_for(args.benchmark_id)
    print(f"{args.benchmark_id}: {series_count} series, depth {max_depth}, "
          f"check {type(metric).__name__.lower()}, commit {current_commit[:12]}",
          file=sys.stderr)

    alert_count = compute_and_emit(
        history, current_commit, args.benchmark_id,
        regressions_only=not args.all_alerts,
    )
    print(f"{args.benchmark_id}: {alert_count} alert(s)", file=sys.stderr)
    sys.exit(1 if alert_count else 0)


if __name__ == "__main__":
    main()
