#!/usr/bin/env python3
"""Summarize Vortex benchmark gh-json / JSONL output with target ratios."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "target" in record and "value" in record:
                    record["_source"] = str(path)
                    records.append(record)
    return records


def target_name(record: dict[str, Any]) -> str:
    target = record.get("target") or {}
    engine = target.get("engine", "?")
    fmt = target.get("format", "?")
    return f"{engine}:{fmt}"


def query_name(record: dict[str, Any]) -> str:
    name = str(record.get("name", ""))
    if "/" in name:
        return name.split("/", 1)[0]
    return name or str(record.get("_source", "unknown"))


def ns_to_ms(value: float) -> float:
    return value / 1_000_000.0


def runtime_summary(record: dict[str, Any]) -> str:
    runtimes = record.get("all_runtimes")
    if not isinstance(runtimes, list) or not runtimes:
        value = float(record["value"])
        return f"{ns_to_ms(value):.3f}/{ns_to_ms(value):.3f}/{ns_to_ms(value):.3f}"
    values = sorted(float(v) for v in runtimes if isinstance(v, (int, float)))
    if not values:
        value = float(record["value"])
        return f"{ns_to_ms(value):.3f}/{ns_to_ms(value):.3f}/{ns_to_ms(value):.3f}"
    return f"{ns_to_ms(values[0]):.3f}/{ns_to_ms(statistics.median(values)):.3f}/{ns_to_ms(values[-1]):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path, help="gh-json JSONL files")
    parser.add_argument(
        "--baseline",
        help="Baseline target such as datafusion:parquet. Defaults to first target per query.",
    )
    args = parser.parse_args()

    records = load_records(args.jsonl)
    if not records:
        print("No benchmark records found.")
        return 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[query_name(record)].append(record)

    print("query\ttarget\tvalue_ms\tratio\tmin/median/max_ms\tsource")
    for query in sorted(groups):
        rows = sorted(groups[query], key=target_name)
        baseline = None
        if args.baseline:
            baseline = next((r for r in rows if target_name(r) == args.baseline), None)
        if baseline is None:
            baseline = rows[0]
        baseline_value = float(baseline["value"])

        for record in rows:
            value = float(record["value"])
            ratio = value / baseline_value if baseline_value else float("nan")
            print(
                f"{query}\t{target_name(record)}\t{ns_to_ms(value):.3f}\t"
                f"{ratio:.2f}x\t{runtime_summary(record)}\t{record['_source']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
