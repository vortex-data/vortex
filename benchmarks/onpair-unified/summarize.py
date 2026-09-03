#!/usr/bin/env python3
"""Summarize raw OnPair JSONL with Type-7 quantiles and decimal GB/s."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values):
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return {
        "samples": len(values),
        "median": quantile(values, 0.5),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p99": quantile(values, 0.99),
    }


def load(paths):
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("type") == "result" and record.get("phase", "measurement") == "measurement":
                    records.append(record)
    return records


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize_vx(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["variant"])].append(record)

    per_dataset = []
    for (dataset, variant), rows in sorted(grouped.items()):
        wall = [row["wall_s"] for row in rows]
        source_gbps = [row["parquet_gbps"] for row in rows]
        result = {"dataset": dataset, "variant": variant}
        for prefix, values in (("source_gbps", source_gbps), ("wall_s", wall)):
            result.update({f"{prefix}_{key}": value for key, value in stats(values).items()})
        slow_wall = quantile(wall, 0.99)
        result["gbps_at_p99_wall"] = rows[0]["parquet_bytes"] / slow_wall / 1e9
        if "parquet_uncompressed_gbps" in rows[0]:
            values = [row["parquet_uncompressed_gbps"] for row in rows]
            result.update(
                {f"uncompressed_gbps_{key}": value for key, value in stats(values).items()}
            )
            result["uncompressed_gbps_at_p99_wall"] = (
                rows[0]["parquet_uncompressed_bytes"] / slow_wall / 1e9
            )
        result["vortex_bytes"] = rows[0]["vortex_bytes"]
        result["vortex_over_parquet"] = rows[0]["vortex_bytes"] / rows[0]["parquet_bytes"]
        per_dataset.append(result)

    sweeps = defaultdict(list)
    for record in records:
        sweeps[(record["variant"], record["repetition"])].append(record)
    aggregate_samples = defaultdict(list)
    for (variant, _), rows in sweeps.items():
        source_bytes = sum(row["parquet_bytes"] for row in rows)
        wall = sum(row["wall_s"] for row in rows)
        aggregate_samples[(variant, "source_gbps")].append(source_bytes / wall / 1e9)
        if all("parquet_uncompressed_bytes" in row for row in rows):
            uncompressed = sum(row["parquet_uncompressed_bytes"] for row in rows)
            aggregate_samples[(variant, "uncompressed_gbps")].append(uncompressed / wall / 1e9)

    aggregate = []
    for (variant, metric), values in sorted(aggregate_samples.items()):
        aggregate.append({"variant": variant, "metric": metric, **stats(values)})
    return per_dataset, aggregate


def summarize_isolated(records):
    grouped = defaultdict(list)
    for record in records:
        block_mib = record["block_target_bytes"] // (1024 * 1024)
        grouped[(record["corpus"], block_mib, record["algorithm"])].append(record)

    per_dataset = []
    for (dataset, block_mib, algorithm), rows in sorted(grouped.items()):
        timed = [
            (row, sample_ms / 1e3)
            for row in rows
            for sample_ms in row["samples_ms"]
        ]
        times_s = [elapsed for _, elapsed in timed]
        throughputs = [row["payload_bytes"] / elapsed / 1e9 for row, elapsed in timed]
        result = {"dataset": dataset, "block_mib": block_mib, "algorithm": algorithm}
        result.update({f"gbps_{key}": value for key, value in stats(throughputs).items()})
        result.update({f"seconds_{key}": value for key, value in stats(times_s).items()})
        slow_time = quantile(times_s, 0.99)
        result["gbps_at_p99_seconds"] = rows[0]["payload_bytes"] / slow_time / 1e9
        result["payload_ratio"] = rows[0]["payload_ratio"]
        result["logical_encoded_bytes"] = rows[0]["sizes"]["logical_encoded_bytes"]
        result["correct"] = all(row["correct"] for row in rows)
        per_dataset.append(result)

    aggregate_cells = defaultdict(list)
    for record in records:
        block_mib = record["block_target_bytes"] // (1024 * 1024)
        aggregate_cells[(block_mib, record["algorithm"])].append(record)
    aggregate_samples = defaultdict(list)
    for (block_mib, algorithm), rows in aggregate_cells.items():
        sample_count = len(rows[0]["samples_ms"])
        if any(len(row["samples_ms"]) != sample_count for row in rows):
            raise ValueError(f"unequal sample counts for block={block_mib} algorithm={algorithm}")
        payload = sum(row["payload_bytes"] for row in rows)
        for sample_index in range(sample_count):
            elapsed = sum(row["samples_ms"][sample_index] for row in rows) / 1e3
            aggregate_samples[(block_mib, algorithm)].append(payload / elapsed / 1e9)
    aggregate = []
    for (block_mib, algorithm), values in sorted(aggregate_samples.items()):
        aggregate.append(
            {"block_mib": block_mib, "algorithm": algorithm, **stats(values)}
        )
    return per_dataset, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("vx", "isolated"))
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    records = load(args.inputs)
    if not records:
        raise SystemExit("no measurement records")
    if args.kind == "vx":
        per_dataset, aggregate = summarize_vx(records)
    else:
        per_dataset, aggregate = summarize_isolated(records)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "-per-dataset.csv"), per_dataset)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "-aggregate.csv"), aggregate)


if __name__ == "__main__":
    main()
