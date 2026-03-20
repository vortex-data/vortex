# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Find benchmark regressions. Writes ndjson to stdout.

This is the "check" half of the pipeline. It loads history, ingests new
results, runs statistical checks, and emits one JSON object per alert to
stdout. Pipe to benchmark_alert.py to send them somewhere, or read them
locally with jq.

    # Local — just see what regressed:
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


# ============================================================================
# METRIC TYPES — one struct per check, all fields typed with defaults
# ============================================================================

@dataclass(frozen=True, slots=True)
class Ewma:
    """EWMA control chart config."""
    pattern: str = "*"
    span: int = 10
    sigma: float = 3.0
    min_observations: int = 5


@dataclass(frozen=True, slots=True)
class Cusum:
    """Tabular CUSUM config."""
    pattern: str = "*"
    drift: float = 0.5
    cusum_threshold: float = 5.0
    min_observations: int = 5


@dataclass(frozen=True, slots=True)
class Threshold:
    """Static upper/lower bound config."""
    pattern: str = "*"
    max_value: float | None = None
    min_value: float | None = None
    min_observations: int = 1


@dataclass(frozen=True, slots=True)
class PctChange:
    """Percentage change from rolling mean config."""
    pattern: str = "*"
    window: int = 5
    pct_threshold: float = 20.0
    min_observations: int = 5


Metric = Ewma | Cusum | Threshold | PctChange


# ============================================================================
# METRICS — edit this to change what gets checked
# ============================================================================

DEFAULT_METRIC: Metric = Ewma()

METRICS: list[Metric] = [
    Ewma(pattern="tpch-s3*",       sigma=4.0, span=15, min_observations=8),
    Ewma(pattern="fineweb-s3*",    sigma=4.0, span=15, min_observations=8),
    Ewma(pattern="compress*",      sigma=2.5, span=8),
    Cusum(pattern="random-access*", cusum_threshold=4.0, drift=0.3, min_observations=8),
]


# ============================================================================
# DATA
# ============================================================================

@dataclass(frozen=True, slots=True)
class Point:
    commit_id: str
    value: float


@dataclass(frozen=True, slots=True)
class Alert:
    series: str
    commit_id: str
    check_name: str
    current: float
    expected: float
    sigma: float
    message: str

    def to_json(self, benchmark_id: str) -> dict:
        return {
            "benchmark_id": benchmark_id,
            "series": self.series,
            "commit_id": self.commit_id,
            "check": self.check_name,
            "current": self.current,
            "expected": self.expected,
            "sigma": self.sigma,
            "message": self.message,
        }


# ============================================================================
# CHECKS
# ============================================================================

def _check_ewma(values: list[float], m: Ewma) -> Alert | None:
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
    return Alert(
        series="", commit_id="", check_name="ewma",
        current=current, expected=ewma, sigma=dev,
        message=f"EWMA {direction}: {dev:+.2f}σ (value={current:.1f}, expected={ewma:.1f}±{std:.1f})",
    )


def _check_cusum(values: list[float], m: Cusum) -> Alert | None:
    if len(values) < m.min_observations:
        return None
    import numpy as np
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
    return Alert(
        series="", commit_id="", check_name="cusum",
        current=current, expected=mu, sigma=dev,
        message=f"CUSUM {direction}: S+={s_pos:.2f} S-={s_neg:.2f} (threshold={m.cusum_threshold})",
    )


def _check_threshold(values: list[float], m: Threshold) -> Alert | None:
    if len(values) < m.min_observations:
        return None
    current = values[-1]
    if m.max_value is not None and current > m.max_value:
        return Alert(
            series="", commit_id="", check_name="threshold",
            current=current, expected=m.max_value, sigma=0.0,
            message=f"Exceeded: {current:.1f} > {m.max_value:.1f}",
        )
    if m.min_value is not None and current < m.min_value:
        return Alert(
            series="", commit_id="", check_name="threshold",
            current=current, expected=m.min_value, sigma=0.0,
            message=f"Below: {current:.1f} < {m.min_value:.1f}",
        )
    return None


def _check_pct_change(values: list[float], m: PctChange) -> Alert | None:
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
    return Alert(
        series="", commit_id="", check_name="pct_change",
        current=current, expected=mean, sigma=pct / m.pct_threshold,
        message=f"Pct change {direction}: {pct:+.1f}% (value={current:.1f}, baseline={mean:.1f})",
    )


def run_check(values: list[float], metric: Metric) -> Alert | None:
    match metric:
        case Ewma():       return _check_ewma(values, metric)
        case Cusum():      return _check_cusum(values, metric)
        case Threshold():  return _check_threshold(values, metric)
        case PctChange():  return _check_pct_change(values, metric)


# ============================================================================
# FILTER
# ============================================================================

def metric_for(series_name: str) -> Metric:
    for m in METRICS:
        p = m.pattern
        if p.endswith("*") and series_name.startswith(p[:-1]):
            return m
        if not p.endswith("*") and p in series_name:
            return m
    return DEFAULT_METRIC


# ============================================================================
# HISTORY — file-locked reads and writes
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
# COMPUTE — run checks, emit ndjson to stdout
# ============================================================================

def compute_and_emit(history: dict[str, list[Point]], commit_id: str,
                     benchmark_id: str, regressions_only: bool) -> int:
    """Run checks, write ndjson to stdout. Returns alert count."""
    count = 0
    for series_name, points in history.items():
        if not points or points[-1].commit_id != commit_id:
            continue
        metric = metric_for(series_name)
        values = [p.value for p in points]
        alert = run_check(values, metric)
        if alert is None:
            continue
        alert = Alert(
            series=series_name, commit_id=commit_id,
            check_name=alert.check_name, current=alert.current,
            expected=alert.expected, sigma=alert.sigma, message=alert.message,
        )
        if regressions_only and alert.sigma <= 0:
            continue
        json.dump(alert.to_json(benchmark_id), sys.stdout)
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
    print(f"{args.benchmark_id}: {series_count} series, depth {max_depth}, "
          f"commit {current_commit[:12]}", file=sys.stderr)

    alert_count = compute_and_emit(
        history, current_commit, args.benchmark_id,
        regressions_only=not args.all_alerts,
    )

    print(f"{args.benchmark_id}: {alert_count} alert(s)", file=sys.stderr)
    sys.exit(1 if alert_count else 0)


if __name__ == "__main__":
    main()
