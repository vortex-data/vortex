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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONTROL_FORMAT = "parquet"


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


def _ewma_deviation(values: list[float], span: int) -> float | None:
    """Compute the EWMA z-score for the latest value without applying a threshold.

    Returns the deviation in sigma units, or None if there is insufficient data.
    Used by the concordance check to measure how much *any* series moved, even
    those that did not trigger an alert.
    """
    if len(values) < 3:
        return None
    alpha = 2.0 / (span + 1)
    ewma = values[0]
    ewma_var = 0.0
    for v in values[1:-1]:
        diff = v - ewma
        ewma_var = alpha * diff * diff + (1.0 - alpha) * ewma_var
        ewma = alpha * v + (1.0 - alpha) * ewma
    std = math.sqrt(ewma_var) if ewma_var > 0 else 0.0
    if std == 0:
        return None
    return (values[-1] - ewma) / std


# ============================================================================
# SERIES IDENTITY
# ============================================================================

@dataclass(frozen=True, slots=True)
class SeriesIdentity:
    """Parsed components of a SQL benchmark series name.

    Series names follow the pattern ``{dataset}_q{N}/{engine}:{format}``,
    e.g. ``tpch_q01/datafusion:vortex``.  Non-SQL benchmarks (compress-bench,
    random-access-bench) do not match this pattern and return None from
    ``parse_series_identity``.
    """
    query: str      # e.g. "01"
    engine: str     # e.g. "datafusion"
    file_format: str  # e.g. "vortex"
    prefix: str     # e.g. "tpch_q01" — everything before "/{engine}:..."


_SERIES_RE = re.compile(r'^(.+_q\d+)/([^:]+):(.+)$')


def parse_series_identity(name: str) -> SeriesIdentity | None:
    """Extract query / engine / format from a series name, or None."""
    m = _SERIES_RE.match(name)
    if m is None:
        return None
    prefix = m.group(1)
    query = prefix.rsplit("_q", 1)[-1]
    return SeriesIdentity(query=query, engine=m.group(2),
                          file_format=m.group(3), prefix=prefix)


# ============================================================================
# CONCORDANCE — cross-engine / cross-format noise detection
# ============================================================================

@dataclass(frozen=True, slots=True)
class NoiseAssessment:
    """Annotation added to an alert when cross-series concordance suggests
    the spike is environmental noise rather than a real regression.

    ``classification`` is one of:
      - ``engine_noise``: the control format (parquet) for the same engine
        and query also spiked — noise likely hit during that engine's run.
      - ``global_noise``: multiple engines show the same spike for this query
        — system-wide environmental noise.
      - ``dep_upgrade``: the alerting engine's dependency was upgraded in this
        commit, so cross-engine comparison is unreliable.
      - ``vortex_only``: only Vortex formats moved; likely a real change.
    """
    classification: str
    control_deviation: float | None = None
    cross_engine_deviations: dict[str, float] = field(default_factory=dict)
    message: str = ""


# Threshold (in σ) above which we consider a non-alerting series to have
# "also moved" in the same direction as the alerting series.
_CONCORDANCE_SIGMA = 2.0


def _build_concordance(
    alert_name: str,
    alert_sigma: float,
    history: dict[str, list[Point]],
    commit_id: str,
    metric: Metric,
    dep_upgrade_engines: set[str],
) -> NoiseAssessment | None:
    """Check whether an alerting series' spike is corroborated by controls.

    Only applies to SQL-style series with engine:format naming.
    """
    identity = parse_series_identity(alert_name)
    if identity is None:
        return None

    span = metric.span if isinstance(metric, Ewma) else 10
    alert_direction = 1.0 if alert_sigma > 0 else -1.0

    # 1. Check if this engine had a dependency upgrade.
    if identity.engine in dep_upgrade_engines:
        return NoiseAssessment(
            classification="dep_upgrade",
            message=f"engine '{identity.engine}' had a dependency upgrade — "
                    f"cross-engine comparison skipped",
        )

    # 2. Check same-engine control format (e.g. datafusion:parquet for the
    #    same query).  If the control also spiked in the same direction, the
    #    noise hit during this engine's sequential run window.
    control_name = f"{identity.prefix}/{identity.engine}:{CONTROL_FORMAT}"
    control_dev = _series_deviation(control_name, history, commit_id, span)

    # 3. Check other engines' *control* formats for the same query.
    #    If duckdb:parquet also spiked when datafusion:vortex alerts, the
    #    whole machine was noisy.  But if duckdb:vortex spiked while
    #    duckdb:parquet stayed calm, that *confirms* a real Vortex change.
    cross_engine_control_devs: dict[str, float] = {}
    for name in history:
        if name == alert_name:
            continue
        other = parse_series_identity(name)
        if other is None:
            continue
        if other.query != identity.query:
            continue
        if other.engine == identity.engine:
            continue
        if other.engine in dep_upgrade_engines:
            continue
        if other.file_format != CONTROL_FORMAT:
            continue
        dev = _series_deviation(name, history, commit_id, span)
        if dev is not None:
            cross_engine_control_devs[other.engine] = dev

    # Classify.
    control_also_moved = (
        control_dev is not None
        and identity.file_format != CONTROL_FORMAT
        and abs(control_dev) >= _CONCORDANCE_SIGMA
        and math.copysign(1.0, control_dev) == alert_direction
    )

    cross_engine_controls_moved = sum(
        1 for d in cross_engine_control_devs.values()
        if abs(d) >= _CONCORDANCE_SIGMA
        and math.copysign(1.0, d) == alert_direction
    )

    if control_also_moved and cross_engine_controls_moved > 0:
        return NoiseAssessment(
            classification="global_noise",
            control_deviation=control_dev,
            cross_engine_deviations=cross_engine_control_devs,
            message=f"control ({CONTROL_FORMAT}) also moved {control_dev:+.1f}σ "
                    f"and {cross_engine_controls_moved} other engine(s)' controls agree — "
                    f"likely system-wide noise",
        )
    if control_also_moved:
        return NoiseAssessment(
            classification="engine_noise",
            control_deviation=control_dev,
            cross_engine_deviations=cross_engine_control_devs,
            message=f"control ({CONTROL_FORMAT}) also moved {control_dev:+.1f}σ "
                    f"for same engine — likely noise during {identity.engine} run",
        )
    if cross_engine_controls_moved > 0 and cross_engine_controls_moved == len(cross_engine_control_devs):
        return NoiseAssessment(
            classification="global_noise",
            control_deviation=control_dev,
            cross_engine_deviations=cross_engine_control_devs,
            message=f"all {cross_engine_controls_moved} other engine(s)' controls "
                    f"also moved — likely system-wide noise",
        )

    return NoiseAssessment(
        classification="vortex_only",
        control_deviation=control_dev,
        cross_engine_deviations=cross_engine_control_devs,
        message="only this series moved — likely a real change",
    )


def _series_deviation(
    name: str,
    history: dict[str, list[Point]],
    commit_id: str,
    span: int,
) -> float | None:
    """Compute the EWMA deviation for an arbitrary series at the given commit."""
    points = history.get(name)
    if not points or points[-1].commit_id != commit_id:
        return None
    values = [p.value for p in points]
    return _ewma_deviation(values, span)


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
                     benchmark_id: str, regressions_only: bool,
                     dep_upgrade_engines: set[str] | None = None) -> int:
    """Run checks, write ndjson to stdout. Returns alert count.

    Two-pass approach:
    1. Compute check results for every series.
    2. For each alert, run the concordance check against control formats and
       other engines to classify the spike as noise vs. real.

    The ``noise_assessment`` field is informational — alerts are still emitted
    regardless, so downstream consumers (benchmark_alert.py) can decide how
    to handle noise-classified alerts.
    """
    metric = metric_for(benchmark_id)
    dep_engines = dep_upgrade_engines or set()

    # Pass 1: collect alerts.
    alerts: list[tuple[str, CheckResult]] = []
    for series_name, points in history.items():
        if not points or points[-1].commit_id != commit_id:
            continue
        values = [p.value for p in points]
        result = run_check(values, metric)
        if result is None:
            continue
        if regressions_only and result.sigma <= 0:
            continue
        alerts.append((series_name, result))

    # Pass 2: emit with concordance annotation.
    count = 0
    noise_count = 0
    for series_name, result in alerts:
        record: dict = {
            "benchmark_id": benchmark_id,
            "series": series_name,
            "commit_id": commit_id,
            "check": result.check_name,
            "current": result.current,
            "expected": result.expected,
            "sigma": result.sigma,
            "message": result.message,
        }
        assessment = _build_concordance(
            series_name, result.sigma, history, commit_id, metric, dep_engines,
        )
        if assessment is not None:
            record["noise_assessment"] = {
                "classification": assessment.classification,
                "message": assessment.message,
            }
            if assessment.control_deviation is not None:
                record["noise_assessment"]["control_deviation"] = round(assessment.control_deviation, 2)
            if assessment.cross_engine_deviations:
                record["noise_assessment"]["cross_engine_deviations"] = {
                    k: round(v, 2) for k, v in assessment.cross_engine_deviations.items()
                }
            if assessment.classification in ("engine_noise", "global_noise", "dep_upgrade"):
                noise_count += 1

        json.dump(record, sys.stdout)
        sys.stdout.write("\n")
        count += 1

    if noise_count:
        print(f"{benchmark_id}: {noise_count}/{count} alert(s) classified as noise",
              file=sys.stderr)

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
    p.add_argument("--dep-upgrade-engines", type=str, default="",
                   help="Comma-separated engines whose dependencies were "
                        "upgraded in this commit (e.g. 'duckdb,datafusion'). "
                        "Cross-engine comparison is skipped for these engines.")
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

    dep_engines = set()
    if args.dep_upgrade_engines:
        dep_engines = {e.strip() for e in args.dep_upgrade_engines.split(",") if e.strip()}
    if dep_engines:
        print(f"{args.benchmark_id}: dependency upgrade engines: {dep_engines}",
              file=sys.stderr)

    alert_count = compute_and_emit(
        history, current_commit, args.benchmark_id,
        regressions_only=not args.all_alerts,
        dep_upgrade_engines=dep_engines,
    )
    print(f"{args.benchmark_id}: {alert_count} alert(s)", file=sys.stderr)
    sys.exit(1 if alert_count else 0)


if __name__ == "__main__":
    main()
