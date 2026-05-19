#!/usr/bin/env python3
"""Compare Vortex benchmark --show-metrics text logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

METRIC_RE = re.compile(r"^\s*([A-Za-z0-9_.\[\]-]+)=(.+?)\s*$")
NUMBER_UNIT_RE = re.compile(r"^\s*([-+]?[0-9]*\.?[0-9]+)\s*([A-Za-z/]+)?\s*$")

UNIT_SCALE = {
    "": 1.0,
    "K": 1_000.0,
    "M": 1_000_000.0,
    "B": 1.0,
    "KB": 1_000.0,
    "MB": 1_000_000.0,
    "GB": 1_000_000_000.0,
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}


def parse_value(raw: str) -> tuple[float | None, str]:
    normalized = raw.strip().replace("\u00b5", "µ")
    match = NUMBER_UNIT_RE.match(normalized)
    if not match:
        return None, raw.strip()
    value = float(match.group(1))
    unit = match.group(2) or ""
    scale = UNIT_SCALE.get(unit)
    if scale is None:
        return value, unit
    return value * scale, unit


def parse_metrics(path: Path) -> dict[str, tuple[float | None, str, str]]:
    metrics: dict[str, tuple[float | None, str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = METRIC_RE.match(line)
            if not match:
                continue
            name, raw = match.groups()
            parsed, unit = parse_value(raw)
            metrics[name] = (parsed, raw.strip(), unit)
    return metrics


def default_metrics(all_metrics: list[dict[str, tuple[float | None, str, str]]]) -> list[str]:
    preferred = [
        "vortex.io.read.duration_count",
        "vortex.io.read.total_size",
        "vortex.io.read.duration_max",
        "vortex.io.read.size_max",
        "vortex.file.segments.cache.misses",
        "vortex.file.segments.cache.hits",
        "io.requests.individual",
        "io.requests.coalesced",
        "time_elapsed_opening",
        "time_elapsed_processing",
        "time_elapsed_scanning_total",
        "time_elapsed_scanning_until_data",
        "output_rows",
        "output_bytes",
    ]
    present = set().union(*(m.keys() for m in all_metrics))
    return [metric for metric in preferred if metric in present]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--metrics",
        help="Comma-separated metric names. Defaults to common Vortex scan metrics.",
    )
    args = parser.parse_args()

    parsed = [(path, parse_metrics(path)) for path in args.logs]
    metric_names = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else default_metrics([metrics for _, metrics in parsed])
    )

    if not metric_names:
        print("No metrics found.")
        return 1

    baseline_metrics = parsed[0][1]
    print("metric\t" + "\t".join(str(path) for path, _ in parsed))
    for metric in metric_names:
        cells = []
        baseline_value = baseline_metrics.get(metric, (None, "", ""))[0]
        for _, metrics in parsed:
            value, raw, _unit = metrics.get(metric, (None, "", ""))
            if not raw:
                cells.append("-")
            elif baseline_value and value is not None:
                cells.append(f"{raw} ({value / baseline_value:.2f}x)")
            else:
                cells.append(raw)
        print(metric + "\t" + "\t".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
