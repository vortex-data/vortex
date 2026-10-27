# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare model errors with independent S3 repetitions in the same CI rounds."""

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import compare


def bootstrap_indices(n: int, rng: random.Random, block: int) -> list[int]:
    indices = []
    while len(indices) < n:
        start = rng.randrange(n)
        indices.extend((start + offset) % n for offset in range(block))
    return indices[:n]


def interval(values: list[float]) -> tuple[float, float]:
    return compare.quantile(values, 0.025), compare.quantile(values, 0.975)


def analyze(records: list[compare.Record], draws: int = 1000) -> compare.Record:
    environments = [r for r in records if r.get("kind") == "environment"]
    groups = defaultdict(dict)
    for record in records:
        if record.get("kind") != "sample":
            continue
        key = (record["suite"], record["query"], record["cache"])
        sample = (record["backend"], record["iteration"])
        if sample in groups[key]:
            raise ValueError(f"Duplicate repeatability sample: {key} {sample}")
        groups[key][sample] = record
    rows = []
    for (suite, query, cache), samples in sorted(groups.items()):
        environment = next(e for e in environments if e["suite"] == suite)
        n = environment.get("iteration_overrides", {}).get(str(query), environment["iterations"])
        failures, a, b, m = [], [], [], []
        if any(index < 0 or index >= n for _, index in samples):
            failures.append("iteration outside the declared count")
        for iteration in range(n):
            triple = [samples.get((backend, iteration)) for backend in ("real", "real_repeat", "model")]
            if any(record is None for record in triple):
                failures.append(f"missing control or model iteration {iteration}")
                continue
            reference, repeated, model = triple
            if any(r["error"] or r["counts"]["errors"] or r["seconds"] <= 0 for r in triple):
                failures.append(f"failed measurement iteration {iteration}")
                continue
            for other in (repeated, model):
                if reference["rows"] != other["rows"] or compare.signature(reference) != compare.signature(other):
                    failures.append(f"request/range/row mismatch iteration {iteration}")
                for field in ("get", "put", "list", "head", "bytes", "errors"):
                    for phase in ("counts", "setup_counts"):
                        if reference[phase][field] != other[phase][field]:
                            failures.append(f"{phase} {field} mismatch iteration {iteration}")
            a.append(reference["seconds"])
            b.append(repeated["seconds"])
            m.append(model["seconds"])
        row = {
            "suite": suite,
            "query": query,
            "cache": cache,
            "iterations": n,
            "requests_match": not failures,
            "failures": "; ".join(sorted(set(failures))),
        }
        if len(a) != n or failures:
            rows.append(row)
            continue
        block = max(1, min(4, n // 4))
        differences = [abs(math.log(x / z)) - abs(math.log(y / z)) for x, y, z in zip(m, b, a, strict=True)]
        rng = random.Random(20261028 + query * 2 + (cache == "warm"))
        median_ratios, noise_ratios, excesses = [], [], []
        for _ in range(draws):
            indices = bootstrap_indices(n, rng, block)
            aa, bb, mm = ([values[index] for index in indices] for values in (a, b, m))
            median_ratios.append(statistics.median(mm) / statistics.median(aa + bb))
            noise_ratios.append(statistics.median(bb) / statistics.median(aa))
            excesses.append(statistics.mean(differences[index] for index in indices))
        row.update(
            bootstrap_block_rounds=block,
            real_median_ms=statistics.median(a) * 1000,
            real_repeat_median_ms=statistics.median(b) * 1000,
            model_median_ms=statistics.median(m) * 1000,
            real_p90_ms=compare.quantile(a, 0.9) * 1000,
            real_repeat_p90_ms=compare.quantile(b, 0.9) * 1000,
            model_p90_ms=compare.quantile(m, 0.9) * 1000,
            real_repeat_median_ratio=statistics.median(b) / statistics.median(a),
            model_median_ratio=statistics.median(m) / statistics.median(a),
            model_vs_pooled_real_ratio=statistics.median(m) / statistics.median(a + b),
            real_repeat_p90_ratio=compare.quantile(b, 0.9) / compare.quantile(a, 0.9),
            model_p90_ratio=compare.quantile(m, 0.9) / compare.quantile(a, 0.9),
            real_repeat_log_mae=statistics.mean(abs(math.log(x / y)) for x, y in zip(b, a, strict=True)),
            model_log_mae=statistics.mean(abs(math.log(x / y)) for x, y in zip(m, a, strict=True)),
            excess_log_mae=statistics.mean(differences),
        )
        for field, values in (
            ("model_ratio", median_ratios),
            ("real_repeat_ratio", noise_ratios),
            ("excess_log_mae", excesses),
        ):
            row[field + "_ci_low"], row[field + "_ci_high"] = interval(values)
        row["bias_ci_excludes_one"] = row["model_ratio_ci_low"] > 1 or row["model_ratio_ci_high"] < 1
        row["error_exceeds_s3_repeatability"] = row["excess_log_mae_ci_low"] > 0
        row["model_outside_band"] = not 0.7 <= row["model_median_ratio"] <= 1.3
        row["real_repeat_outside_band"] = not 0.7 <= row["real_repeat_median_ratio"] <= 1.3
        rows.append(row)
    return {
        "scope": "focused same-host diagnostics, not full-suite validation",
        "confidence": 0.95,
        "bootstrap_draws": draws,
        "bootstrap": "circular blocks of consecutive query rounds; controls resampled together",
        "metric": "excess mean absolute log(model/real) minus mean absolute log(real_repeat/real)",
        "environments": environments,
        "rows": rows,
    }


def write_reports(result: compare.Record, destination: Path):
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in result["rows"] for key in row))
    with destination.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
    lines = [
        "# S3 repeatability and model error",
        "",
        result["scope"],
        "",
        "Independent real S3 controls and the frozen model run in balanced orders. Query rounds "
        "spread repetitions over time. Confidence intervals use a paired circular block bootstrap. "
        "A confidence interval including one does not establish equivalence. The fixed 0.7–1.3 "
        "flags are retained separately. Tail estimates remain limited by sample count.",
        "",
        "| Suite/query/cache | Real/real median | Model/real median | Model/pooled real 95% CI | "
        "Error exceeds S3 repeatability |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in result["rows"]:
        label = f"{row['suite']}/{row['query']}/{row['cache']}"
        if not row["requests_match"]:
            lines.append(f"| {label} | invalid | invalid | {row['failures']} | invalid |")
        else:
            lines.append(
                f"| {label} | {row['real_repeat_median_ratio']:.3f} | {row['model_median_ratio']:.3f} | "
                f"{row['model_ratio_ci_low']:.3f}–{row['model_ratio_ci_high']:.3f} | "
                f"{row['error_exceeds_s3_repeatability']} |"
            )
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n")
