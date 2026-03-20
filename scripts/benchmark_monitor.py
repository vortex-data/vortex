# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Benchmark regression monitor.

Pipeline: load → filter → compute → emit

Each commit is one logical step. Wall-clock time is irrelevant.
Edit METRICS below to change what gets checked and how.

    # CI (alerts go to incident.io, summary to GitHub Actions):
    uv run --no-project scripts/benchmark_monitor.py \
        --history-dir ./benchmark-history \
        --commits commits.json \
        --results results.json \
        --benchmark-id tpch-nvme \
        --current-commit abc123 \
        --webhook-url https://...

    # Local (just prints to terminal):
    uv run --no-project scripts/benchmark_monitor.py \
        --history-dir ./benchmark-history \
        --commits commits.json \
        --results results.json \
        --benchmark-id tpch-nvme
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


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
# METRICS — this is the only section you need to edit
# ============================================================================
#
# Each entry is a typed struct. Pattern matching: trailing * = prefix,
# otherwise substring. First match wins. Unmatched series get DEFAULT_METRIC.

DEFAULT_METRIC: Metric = Ewma()

METRICS: list[Metric] = [
    # S3 benchmarks — noisier, wider bounds
    Ewma(pattern="tpch-s3*",       sigma=4.0, span=15, min_observations=8),
    Ewma(pattern="fineweb-s3*",    sigma=4.0, span=15, min_observations=8),
    # Compression — stable, tighter bounds
    Ewma(pattern="compress*",      sigma=2.5, span=8),
    # Random access — CUSUM catches gradual drift
    Cusum(pattern="random-access*", cusum_threshold=4.0, drift=0.3, min_observations=8),
]


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass(frozen=True, slots=True)
class Point:
    """One observation: a commit and a value."""
    commit_id: str
    value: float


@dataclass(frozen=True, slots=True)
class Alert:
    """A detected anomaly."""
    series: str
    commit_id: str
    check_name: str
    current: float
    expected: float
    sigma: float
    message: str


# ============================================================================
# CHECKS — one function per metric type, dispatched via match
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
    """Dispatch to the right check based on metric type."""
    match metric:
        case Ewma():       return _check_ewma(values, metric)
        case Cusum():      return _check_cusum(values, metric)
        case Threshold():  return _check_threshold(values, metric)
        case PctChange():  return _check_pct_change(values, metric)


# ============================================================================
# FILTER — match series names to metrics
# ============================================================================

def metric_for(series_name: str) -> Metric:
    """First matching metric from METRICS, or DEFAULT_METRIC."""
    for m in METRICS:
        p = m.pattern
        if p.endswith("*") and series_name.startswith(p[:-1]):
            return m
        if not p.endswith("*") and p in series_name:
            return m
    return DEFAULT_METRIC


# ============================================================================
# HISTORY — per-benchmark files with file locking
# ============================================================================

def _history_path(history_dir: Path, benchmark_id: str) -> Path:
    safe = benchmark_id.replace("/", "_").replace(":", "_")
    return history_dir / f"{safe}.json"


def load_history(history_dir: Path, benchmark_id: str) -> dict[str, list[Point]]:
    """Load history, returns {series_name: [Point, ...]} in logical order."""
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
    """Save history with an exclusive file lock (safe for concurrent writers)."""
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
    """Append new results into history. Deduplicates by commit."""
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
    """Commit IDs in chronological order from commits.json (ndjson)."""
    commits = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                commits.append(json.loads(line)["id"])
    return commits


# ============================================================================
# COMPUTE — run checks on every series
# ============================================================================

def compute(history: dict[str, list[Point]], commit_id: str,
            regressions_only: bool = True) -> list[Alert]:
    """Run the configured check on each series that has data for commit_id."""
    alerts: list[Alert] = []
    for series_name, points in history.items():
        if not points or points[-1].commit_id != commit_id:
            continue

        metric = metric_for(series_name)
        values = [p.value for p in points]
        alert = run_check(values, metric)
        if alert is None:
            continue

        alert = Alert(
            series=series_name,
            commit_id=commit_id,
            check_name=alert.check_name,
            current=alert.current,
            expected=alert.expected,
            sigma=alert.sigma,
            message=alert.message,
        )

        if regressions_only and alert.sigma <= 0:
            continue
        alerts.append(alert)

    return alerts


# ============================================================================
# EMIT — structured output to terminal, GitHub Actions, and incident.io
# ============================================================================

class Emitter(Protocol):
    def emit(self, alerts: list[Alert], benchmark_id: str, commit_id: str) -> None: ...


class TerminalEmitter:
    """Pretty-prints to stderr. Always active."""

    def emit(self, alerts: list[Alert], benchmark_id: str, commit_id: str) -> None:
        if not alerts:
            print("No regressions detected.", file=sys.stderr)
            return
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"REGRESSIONS ({len(alerts)}) — {benchmark_id} @ {commit_id[:12]}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        for a in alerts:
            print(f"  [{a.check_name.upper()}] {a.series}", file=sys.stderr)
            print(f"    {a.message}", file=sys.stderr)
        print(file=sys.stderr)


class GitHubActionsEmitter:
    """Writes GITHUB_OUTPUT vars and GITHUB_STEP_SUMMARY markdown.
    Auto-skips when not running in GitHub Actions."""

    def emit(self, alerts: list[Alert], benchmark_id: str, commit_id: str) -> None:
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"alert_count={len(alerts)}\n")
                f.write(f"has_alerts={'true' if alerts else 'false'}\n")

        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if not summary_file:
            return
        with open(summary_file, "a") as f:
            if not alerts:
                f.write(f"## {benchmark_id}\nNo regressions detected.\n")
                return
            f.write(f"## {benchmark_id} — {len(alerts)} regression(s)\n\n")
            f.write("| Series | Check | σ | Current | Expected |\n")
            f.write("|--------|-------|---|---------|----------|\n")
            for a in alerts:
                f.write(f"| `{a.series}` | {a.check_name} | {a.sigma:+.2f} "
                        f"| {a.current:.1f} | {a.expected:.1f} |\n")


class IncidentIOEmitter:
    """Posts each alert to incident.io. No-op if no webhook URL."""

    def __init__(self, webhook_url: str | None) -> None:
        self.webhook_url = webhook_url

    def emit(self, alerts: list[Alert], benchmark_id: str, commit_id: str) -> None:
        if not self.webhook_url or not alerts:
            return
        for alert in alerts:
            payload = json.dumps({
                "title": f"Benchmark regression: {alert.series}",
                "description": alert.message,
                "deduplication_key": f"bench-{benchmark_id}-{alert.series}-{commit_id[:12]}",
                "metadata": {
                    "benchmark_suite": benchmark_id,
                    "series": alert.series,
                    "commit": commit_id,
                    "check": alert.check_name,
                    "current_value": alert.current,
                    "expected_value": alert.expected,
                    "deviation_sigma": alert.sigma,
                },
            }).encode()
            req = urllib.request.Request(
                self.webhook_url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    print(f"  incident.io: {alert.series} → HTTP {resp.status}", file=sys.stderr)
            except Exception as e:
                print(f"  incident.io: {alert.series} FAILED: {e}", file=sys.stderr)


class JsonEmitter:
    """Writes alerts as ndjson to stdout. Useful for piping."""

    def emit(self, alerts: list[Alert], benchmark_id: str, commit_id: str) -> None:
        for a in alerts:
            json.dump({
                "benchmark_id": benchmark_id,
                "series": a.series,
                "commit_id": a.commit_id,
                "check": a.check_name,
                "current": a.current,
                "expected": a.expected,
                "sigma": a.sigma,
                "message": a.message,
            }, sys.stdout)
            sys.stdout.write("\n")


# ============================================================================
# PIPELINE — wire it all together
# ============================================================================

def run(*, history_dir: Path, commits_path: Path, results_path: Path,
        benchmark_id: str, current_commit: str | None,
        webhook_url: str | None, json_output: bool,
        regressions_only: bool = True) -> list[Alert]:
    """The full pipeline: load → ingest → compute → emit."""

    # --- load ---
    commit_order = load_commit_order(commits_path)
    if not current_commit:
        if not commit_order:
            print("No commits found.", file=sys.stderr)
            sys.exit(1)
        current_commit = commit_order[-1]

    history = load_history(history_dir, benchmark_id)
    history = ingest(history, results_path, current_commit)
    save_history(history_dir, benchmark_id, history, current_commit)

    series_count = len(history)
    max_depth = max((len(pts) for pts in history.values()), default=0)
    print(f"{benchmark_id}: {series_count} series, depth {max_depth}, "
          f"commit {current_commit[:12]}", file=sys.stderr)

    # --- compute ---
    alerts = compute(history, current_commit, regressions_only=regressions_only)

    # --- emit ---
    emitters: list[Emitter] = [TerminalEmitter()]
    if json_output:
        emitters.append(JsonEmitter())
    if os.environ.get("GITHUB_ACTIONS"):
        emitters.append(GitHubActionsEmitter())
    if webhook_url:
        emitters.append(IncidentIOEmitter(webhook_url))

    for emitter in emitters:
        emitter.emit(alerts, benchmark_id, current_commit)

    return alerts


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark monitor (sample-indexed)")
    p.add_argument("--history-dir", type=Path, required=True)
    p.add_argument("--commits", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--benchmark-id", type=str, required=True)
    p.add_argument("--current-commit", type=str, default=None)
    p.add_argument("--webhook-url", type=str, default=None,
                   help="incident.io webhook URL (omit for local use)")
    p.add_argument("--json", action="store_true", dest="json_output",
                   help="Emit alerts as ndjson to stdout (for piping)")
    p.add_argument("--regressions-only", action="store_true", default=True)
    p.add_argument("--all-alerts", action="store_true",
                   help="Include improvements, not just regressions")
    args = p.parse_args()

    alerts = run(
        history_dir=args.history_dir,
        commits_path=args.commits,
        results_path=args.results,
        benchmark_id=args.benchmark_id,
        current_commit=args.current_commit,
        webhook_url=args.webhook_url,
        json_output=args.json_output,
        regressions_only=not args.all_alerts,
    )

    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
