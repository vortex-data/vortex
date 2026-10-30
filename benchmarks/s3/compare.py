# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare native benchmark per-iteration results, retaining real-S3 repeat controls."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import measure


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + end - 1) / 2
        start = end
    return result


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(ranks(left), ranks(right))


def identity(record: dict) -> tuple:
    fields = ("kind", "dataset", "dataset_variant", "scale_factor", "query_idx", "engine", "format", "open_mode", "op")
    return tuple(str(record.get(field, "")).replace("-s3-simulated", "-s3") for field in fields)


def compare(directory: Path) -> tuple[list[dict], dict]:
    groups = defaultdict(dict)
    hashes, environments = set(), set()
    for path in sorted(directory.glob("*.jsonl")):
        backend = next((name for name in ("real-repeat", "simulated", "real") if path.stem.endswith("-" + name)), None)
        if backend is None:
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["kind"] not in ("query_measurement", "random_access_time", "compression_time"):
                continue
            key = identity(record)
            if backend in groups[key]:
                raise ValueError(f"Duplicate benchmark result: {path}: {key}")
            values = record["all_runtimes_ns"]
            if len(values) < 3 or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"Need at least three positive iteration times: {path}: {key}")
            groups[key][backend] = [value / 1e6 for value in values]
            hashes.add(record["commit_sha"])
            environments.add(record["env_triple"])
    if not groups or len(hashes) != 1 or len(environments) != 1:
        raise ValueError("Comparison needs one source revision and architecture with nonempty workloads")
    rows = []
    for key, backends in sorted(groups.items()):
        if set(backends) != {"real", "real-repeat", "simulated"}:
            raise ValueError(f"Missing backend result: {key}")
        row = dict(
            zip(
                ("kind", "dataset", "variant", "scale", "query", "engine", "format", "open_mode", "op"),
                key,
                strict=True,
            )
        )
        for name, values in backends.items():
            row[f"{name}_n"] = len(values)
            row[f"{name}_median_ms"] = statistics.median(values)
            row[f"{name}_p90_ms"] = percentile(values, 0.9)
        reference = backends["real"] + backends["real-repeat"]
        row["pooled_real_median_ms"] = statistics.median(reference)
        row["pooled_real_p90_ms"] = percentile(reference, 0.9)
        row["model_ratio"] = row["simulated_median_ms"] / row["pooled_real_median_ms"]
        row["real_repeat_ratio"] = row["real-repeat_median_ms"] / row["real_median_ms"]
        row["outside_10pct"] = not 0.9 <= row["model_ratio"] <= 1.1
        row["outside_30pct"] = not 0.7 <= row["model_ratio"] <= 1.3
        rows.append(row)
    summary = {
        "rows": len(rows),
        "within_10pct": sum(not row["outside_10pct"] for row in rows),
        "within_30pct": sum(not row["outside_30pct"] for row in rows),
        "median_absolute_error": statistics.median(abs(row["model_ratio"] - 1) for row in rows),
        "spearman": correlation([row["real_median_ms"] for row in rows], [row["simulated_median_ms"] for row in rows]),
        "ratio_reference": "pooled real and repeat-real samples",
        "spearman_reference": "first real control",
        "scope": "native benchmark cache sequence; SQL first/warm iterations pooled, not isolated cold/warm trials",
        "request_parity": "Simulated command HTTP counters retained; real per-query parity is not measured here",
        "recommendation": "Keep simulation opt-in pending isolated cache trials and real per-query request counts",
    }
    return rows, summary


def probes(directory: Path):
    sources = {}
    for label, name in (("real", "heldout-real"), ("simulated", "probe-simulated"), ("zero", "probe-zero")):
        _, rows = measure.read_window(directory / f"{name}.jsonl")
        sources[label] = {(r["round"], r["bytes_per_request"], r["concurrency"]): r for r in rows}
    if any(set(rows) != set(sources["real"]) for rows in sources.values()):
        raise ValueError("Held-out request sweeps have different coverage")
    comparison = []
    for key, real in sources["real"].items():
        row = {"round": key[0], "bytes_per_request": key[1], "concurrency": key[2]}
        for label, values in sources.items():
            actual = values[key]
            if (actual["requests"], actual["bytes"]) != (real["requests"], real["bytes"]):
                raise ValueError("Held-out GET/byte counts differ")
            for phase in ("headers", "first_byte", "body", "total"):
                for statistic in ("median", "p90"):
                    field = f"{phase}_{statistic}_ms"
                    row[f"{label}_{field}"] = actual[field]
        row["model_total_ratio"] = row["simulated_total_median_ms"] / row["real_total_median_ms"]
        comparison.append(row)
    with (directory / "probe-comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    probes(args.directory)
    rows, summary = compare(args.directory)
    with (args.directory / "comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [
        "# Native S3 model comparison",
        "",
        json.dumps(summary, indent=2),
        "",
        "SQL: all configured queries, with normal adaptive plans and client caching. "
        "Random access reports cached/reopen separately. File decompression uses full GET plus decode; "
        "Lance uses its native range reader. Calibration GETs, preparation and query timing are separate.",
        "",
        "| Kind/dataset/query | Engine/format/mode | Pooled real ms | Model ms | "
        "Model/pooled real | Repeat/first real | ±10% |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        text.append(
            f"| {row['kind']}/{row['dataset']}/{row['query']} | "
            f"{row['engine']}/{row['format']}/{row['open_mode']} | {row['pooled_real_median_ms']:.3f} | "
            f"{row['simulated_median_ms']:.3f} | {row['model_ratio']:.3f} | "
            f"{row['real_repeat_ratio']:.3f} | {'flag' if row['outside_10pct'] else 'pass'} |"
        )
    (args.directory / "comparison.md").write_text("\n".join(text) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
