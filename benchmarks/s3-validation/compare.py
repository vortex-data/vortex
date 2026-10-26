# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Calibrate from S3 request traces and report paired benchmark measurements."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def read(paths: Iterable[str | Path]) -> list[Record]:
    return [json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line]


def quantile(values: Sequence[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end][1] == ordered[start][1]:
            end += 1
        for index, _ in ordered[start:end]:
            result[index] = (start + end - 1) / 2
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    x, y = ranks(left), ranks(right)
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def calibrate(records: list[Record]) -> Record:
    requests = [
        request
        for record in records
        if record.get("kind") == "sample" and record["backend"] == "real" and not record["error"]
        for request in record["requests"]
        if request["operation"] == "GET" and request["first_byte_ms"] is not None
    ]
    small = [r["first_byte_ms"] for r in requests if r["bytes"] <= 65536 and r["inflight_at_start"] <= 4]
    if not small:
        small = [r["first_byte_ms"] for r in requests if r["bytes"] <= 65536]
    if not small:
        raise ValueError("No small S3 requests available to estimate TTFB")
    observed_ttfb = statistics.median(small)
    timer_overheads = [
        request["first_byte_ms"]
        - record["model"]["ttfb_ms"]
        - request["bytes"] / 1048576 / record["model"]["request_mib_per_sec"] * 1000
        for record in records
        if record.get("kind") == "sample" and record["backend"] == "model" and not record["error"]
        for request in record["requests"]
        if request["operation"] == "GET"
        and request["bytes"] <= 65536
        and request["first_byte_ms"] is not None
        and request["inflight_at_start"] <= 4
    ]
    timer_overhead = max(0.0, statistics.median(timer_overheads)) if timer_overheads else 0.0
    ttfb = max(0.0, observed_ttfb - timer_overhead)
    rates = [
        r["bytes"] / 1048576 / ((r["duration_ms"] - observed_ttfb) / 1000)
        for r in requests
        if r["bytes"] >= 262144 and r["duration_ms"] > observed_ttfb + 0.1 and r["inflight_at_start"] <= 4
    ]
    if not rates:
        rates = [
            r["bytes"] / 1048576 / ((r["duration_ms"] - observed_ttfb) / 1000)
            for r in requests
            if r["bytes"] >= 262144 and r["duration_ms"] > observed_ttfb + 0.1
        ]
    if not rates:
        raise ValueError("No large S3 requests available to estimate throughput")
    return {
        "ttfb_ms": round(ttfb, 3),
        "observed_s3_ttfb_ms": round(observed_ttfb, 3),
        "timer_overhead_ms": round(timer_overhead, 3),
        "request_mib_per_sec": round(min(10000.0, max(1.0, quantile(rates, 0.75))), 3),
        # Observed concurrency alone does not establish a service-side limit.
        "max_inflight": 0,
        "calibration_small_requests": len(small),
        "calibration_large_requests": len(rates),
    }


def signature(record: Record) -> Counter[tuple[str, str, str, int]]:
    return Counter((r["operation"], r["path"], r["requested_range"], r["bytes"]) for r in record["requests"])


def compare(records: list[Record], iterations: int, suites: list[str]) -> Record:
    groups = defaultdict(dict)
    environments = [r for r in records if r.get("kind") == "environment"]
    for record in records:
        if record.get("kind") != "sample":
            continue
        key = (record["suite"], record["query"], record["cache"])
        sample_key = (record["backend"], record["iteration"])
        if sample_key in groups[key]:
            raise ValueError(f"Duplicate sample: {key} {sample_key}")
        groups[key][sample_key] = record
    rows = []
    for suite in suites:
        for query in range(1, {"tpch": 22, "tpcds": 99}[suite] + 1):
            for cache in ("cold", "warm"):
                samples = groups.get((suite, query, cache), {})
                real, model = [], []
                failures = []
                for iteration in range(iterations):
                    a, b = samples.get(("real", iteration)), samples.get(("model", iteration))
                    if a is None or b is None:
                        failures.append(f"missing iteration {iteration}")
                        continue
                    if a["error"] or b["error"]:
                        failures.append(f"query error iteration {iteration}: {a['error'] or b['error']}")
                        continue
                    for field in ("get", "put", "list", "head", "bytes", "errors"):
                        if a["counts"][field] != b["counts"][field]:
                            failures.append(f"{field} mismatch iteration {iteration}")
                        if a["setup_counts"][field] != b["setup_counts"][field]:
                            failures.append(f"setup {field} mismatch iteration {iteration}")
                    if signature(a) != signature(b):
                        failures.append(f"request trace mismatch iteration {iteration}")
                    if a["rows"] != b["rows"]:
                        failures.append(f"result row mismatch iteration {iteration}")
                    if a["counts"]["errors"] or b["counts"]["errors"]:
                        failures.append(f"store errors iteration {iteration}")
                    real.append(a["seconds"])
                    model.append(b["seconds"])
                ratio = statistics.median(model) / statistics.median(real) if real else None
                row = {
                    "suite": suite,
                    "query": query,
                    "cache": cache,
                    "real_median_ms": statistics.median(real) * 1000 if real else None,
                    "model_median_ms": statistics.median(model) * 1000 if model else None,
                    "real_p90_ms": quantile(real, 0.9) * 1000 if real else None,
                    "model_p90_ms": quantile(model, 0.9) * 1000 if model else None,
                    "ratio": ratio,
                    "outside_band": ratio is None or not 0.7 <= ratio <= 1.3,
                    "requests_match": not failures,
                    "failures": "; ".join(sorted(set(failures))),
                }
                for backend in ("real", "model"):
                    observed = [s for (b, _), s in samples.items() if b == backend and not s["error"]]
                    for field in (
                        "get",
                        "put",
                        "list",
                        "head",
                        "bytes",
                        "peak_inflight",
                        "get_calls",
                        "tail_cache_hits",
                    ):
                        values = [s["counts"][field] for s in observed if field in s["counts"]]
                        row[f"{backend}_{field}"] = (
                            statistics.median(values) if observed and len(values) == len(observed) else None
                        )
                    row[f"{backend}_avg_inflight"] = (
                        statistics.median(s.get("avg_inflight", 0) for s in observed) if observed else None
                    )
                rows.append(row)
    correlations = {}
    for suite in [*suites, "all"]:
        for cache in ("cold", "warm"):
            eligible = [
                r
                for r in rows
                if r["cache"] == cache
                and r["requests_match"]
                and r["ratio"] is not None
                and (suite == "all" or r["suite"] == suite)
            ]
            correlations[f"{suite}/{cache}"] = spearman(
                [r["real_median_ms"] for r in eligible], [r["model_median_ms"] for r in eligible]
            )
    adequate = all(r["requests_match"] and not r["outside_band"] for r in rows)
    adequate &= all(c is not None and c >= 0.95 for c in correlations.values())
    adequate &= all(not e.get("local_smoke") and e["scale_factor"] in ("10", "10.0") for e in environments)
    adequate &= iterations >= 3 and set(suites) == {"tpch", "tpcds"} and len(environments) == len(suites)
    provenance = (
        "ci_run_id",
        "sha",
        "region",
        "instance_id",
        "format",
        "cpus",
        "total_memory",
        "adaptive_io",
        "tail_cache_bytes",
    )
    adequate &= all(all(e.get(key) is not None for key in provenance) for e in environments)
    adequate &= len({tuple(e.get(key) for key in provenance) for e in environments}) == 1
    return {
        "adequate": adequate,
        "default_ci_recommended": adequate and all(e.get("adaptive_io") for e in environments),
        "correlations": correlations,
        "rows": rows,
        "environments": environments,
    }


def markdown(result: Record) -> str:
    lines = [
        "# Modelled S3 validation",
        "",
        "Recommendation: "
        + (
            "use the model as the default for this measured engine, format and runner."
            if result["default_ci_recommended"]
            else "use the model for the fixed-read profile; keep real S3 for adaptive SQL CI benchmarks."
            if result["adequate"]
            else "keep real S3 as the reference; the model has not met the default CI gate."
        ),
        "",
        "Fit gate: complete TPC-H/TPC-DS SF10 coverage, at least three paired samples per cache state, exact "
        "request/byte/row-count agreement, every median ratio in 0.7–1.3, and Spearman ≥0.95 per suite/cache "
        "state.",
        "",
        "Cold samples use new DataFusion and Vortex sessions. Warm samples immediately repeat the same query "
        "in that session. Footer metadata and statistics caches are cleared after registration. "
        "The file catalog is prepared outside the timer; file listings stay cached to avoid duplicate LIST races. "
        "Registration is outside the query timer and its request counts are checked "
        "separately. Both stores retain the same immutable RAM snapshot during measurements. The real backend "
        "uses a reused S3 HTTP client; cold does not mean a new TLS connection or a cleared S3 service cache.",
        "",
        "The fixed-read validation profile disables opportunistic read coalescing, file repartitioning, "
        "dynamic filter pushdown and Vortex predicate pushdown for both backends. DataFusion evaluates "
        "filters after the scan. Query operators and intra-file scans remain parallel. A successful fit "
        "is conditional on this profile and does not validate adaptive scans.",
        "",
        "Both backends share identical per-sample caching of ranges in each file's final 4 MiB. This "
        "coalesces duplicate footer fetches; the cache resets before every sample. GET/byte counters "
        "measure underlying operations after the cache. get_calls records total logical GET calls and "
        "tail_cache_hits counts calls served by the cache. Real cache misses read from the source store.",
        "",
        "Request metrics count underlying object-store API operations. The snapshot has fewer than 1,000 objects, so "
        "flat LISTs need one S3 page; internal HTTP retries and 503 responses retried by the S3 client are "
        "not counted separately. In-flight averages are the integral of active response lifetimes divided by "
        "query wall time. No randomness or per-query runtime multipliers are used.",
        "",
        "## Parameters and host",
        "",
        "```json",
        json.dumps(result["environments"], indent=2),
        "```",
        "",
        "## Rank correlation",
        "",
        "| Suite/cache | Spearman |",
        "| --- | ---: |",
    ]
    for key, value in result["correlations"].items():
        lines.append(f"| {key} | {value:.4f} |" if value is not None else f"| {key} | unavailable |")
    lines += [
        "",
        "## Per-query comparison",
        "",
        "| Suite | Query | Cache | Real median ms | Model median ms | Real p90 ms | Model p90 ms | Ratio | "
        "Requests | Status |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["rows"]:
        values = [
            f"{row[key]:.2f}" if row[key] is not None else "—"
            for key in ("real_median_ms", "model_median_ms", "real_p90_ms", "model_p90_ms", "ratio")
        ]
        status = "INVALID" if not row["requests_match"] else "OUTSIDE" if row["outside_band"] else "OK"
        lines.append(
            f"| {row['suite']} | {row['query']} | {row['cache']} | {' | '.join(values)} | "
            f"{'match' if row['requests_match'] else 'MISMATCH'} | {status} |"
        )
    problems = [r for r in result["rows"] if r["failures"]]
    if problems:
        lines += ["", "## Invalid comparisons", ""]
        lines += [f"- {r['suite']} q{r['query']} {r['cache']}: {r['failures']}" for r in problems]
    lines += [
        "",
        "## Limits",
        "",
        "A fixed per-request model cannot yet reproduce service throttling, retries, connection "
        "establishment, shared network bandwidth, or latency tails. Highly parallel scans and many small "
        "reads need particular attention. Matching row counts does not establish full result equality; both "
        "backends read identical source bytes and use the same query engine. A recommendation applies only to "
        "the measured engine, format, scale and hardware.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "compare"))
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--suites", nargs="+", default=["tpch", "tpcds"])
    args = parser.parse_args()
    records = read(args.inputs)
    if args.mode == "calibrate":
        args.output.write_text(json.dumps(calibrate(records), indent=2) + "\n")
        return
    result = compare(records, args.iterations, args.suites)
    write_reports(result, args.output)
    print(
        json.dumps(
            {
                "adequate": result["adequate"],
                "correlations": result["correlations"],
                "outside_band": sum(r["outside_band"] for r in result["rows"]),
                "invalid": sum(not r["requests_match"] for r in result["rows"]),
            }
        )
    )


def write_reports(result: Record, path: Path):
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    path.write_text(markdown(result))
    with path.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=result["rows"][0].keys())
        writer.writeheader()
        writer.writerows(result["rows"])


if __name__ == "__main__":
    main()
