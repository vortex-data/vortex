#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare base V2, head V2, and head V1 on the same prepared SQL benchmark files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any

VARIANTS = ("base-v2", "head-v2", "head-v1")


def binary_provenance(binary: Path, commit: str) -> dict[str, str]:
    build = json.loads((binary.parent / "build-manifest.json").read_text(encoding="utf-8"))
    if build["commit"] != commit:
        raise ValueError(f"Wrong build revision for {binary}: {build['commit']} != {commit}")
    with binary.open("rb") as executable:
        digest = hashlib.file_digest(executable, "sha256").hexdigest()
    if build["binaries"][binary.name] != digest:
        raise ValueError(f"Binary hash does not match its build manifest: {binary}")
    return {"commit": commit, "sha256": digest}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def binary_environment(binary: Path, threads: int, temporary: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("VORTEX_SCAN_V2", "VORTEX_USE_SCAN_API", "BLOCKING_MAX_THREADS"):
        env.pop(key, None)
    env.update(
        RUST_LOG="warn",
        TOKIO_WORKER_THREADS=str(threads),
        DATAFUSION_EXECUTION_TARGET_PARTITIONS=str(threads),
        TMPDIR=str(temporary),
    )
    libraries = sorted(binary.parent.glob("build/vortex-duckdb-cache/duckdb-lib-*-prebuilt"))
    if libraries:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*(str(path) for path in libraries), env.get("LD_LIBRARY_PATH", "")])
    return env


def command(
    binary: Path,
    suite: str,
    options: list[str],
    engine: str,
    threads: int,
    version: int,
) -> list[str]:
    args = [
        str(binary),
        suite,
        "--formats",
        "vortex",
        "--scan-v1",
        str(version == 1).lower(),
        "--runner",
        "scan-v2-comparison",
    ]
    for option in options:
        args.extend(["--opt", option])
    if engine == "duckdb":
        args.extend(["--threads", str(threads)])
    return args


def measurement(path: Path, *, query: int, engine: str, commit: str, iterations: int, warmup: int) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError(f"Expected one query record in {path}, found {len(records)}")
    record = records[0]
    name = record["name"].split("/", 1)[0]
    if int(name.rsplit("_q", 1)[1]) != query:
        raise ValueError(f"Wrong query in {path}: {name}")
    if record["commit_id"] != commit:
        raise ValueError(f"Wrong measurement checkout in {path}: {record['commit_id']} != {commit}")
    if record["target"] != {"engine": engine, "format": "vortex"} or record["unit"] != "ns":
        raise ValueError(f"Unexpected measurement target or unit in {path}")
    samples = record["all_runtimes"]
    if len(samples) != iterations or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError(f"Invalid runtime samples in {path}")
    return {"cold_ns": samples[0], "warm_ns": samples[warmup:], "record": record}


def ratio_interval(
    numerators: list[list[list[float]]], denominators: list[list[list[float]]], resamples: int = 1000
) -> tuple[float, float]:
    """Resample paired process rounds, then warm samples, retaining the fixed query workload."""
    rng = random.Random(0)
    ratios = []
    for _ in range(resamples):
        query_ratios = []
        for top, bottom in zip(numerators, denominators, strict=True):
            selected = rng.choices(range(len(top)), k=len(top))
            numerator = [value for index in selected for value in rng.choices(top[index], k=len(top[index]))]
            denominator = [value for index in selected for value in rng.choices(bottom[index], k=len(bottom[index]))]
            query_ratios.append(math.log(statistics.median(numerator) / statistics.median(denominator)))
        ratios.append(math.exp(statistics.mean(query_ratios)))
    ratios.sort()
    return ratios[int(0.025 * resamples)], ratios[min(int(0.975 * resamples), resamples - 1)]


def summarize(rows: list[dict[str, Any]], engines: list[str], queries: list[int], rounds: int) -> dict[str, Any]:
    expected = {
        (engine, query, turn, variant)
        for engine in engines
        for query in queries
        for turn in range(rounds)
        for variant in VARIANTS
    }
    keys = [(row["engine"], row["query"], row["round"], row["variant"]) for row in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Comparison has duplicate or missing engine/query/round/variant measurements")
    result: dict[str, Any] = {"complete": True, "engines": {}}
    for engine in engines:
        per_query: list[dict[str, Any]] = []
        samples_by_variant: dict[str, list[list[list[float]]]] = {variant: [] for variant in VARIANTS}
        for query in queries:
            samples = {
                variant: [
                    value
                    for row in rows
                    if row["engine"] == engine and row["query"] == query and row["variant"] == variant
                    for value in row["warm_ns"]
                ]
                for variant in VARIANTS
            }
            medians = {variant: statistics.median(values) for variant, values in samples.items()}
            per_query.append(
                {
                    "query": query,
                    "median_ms": {variant: value / 1e6 for variant, value in medians.items()},
                    "head_v2_vs_base_v2": medians["head-v2"] / medians["base-v2"],
                    "head_v2_vs_head_v1": medians["head-v2"] / medians["head-v1"],
                }
            )
            for variant in VARIANTS:
                samples_by_variant[variant].append(
                    [
                        row["warm_ns"]
                        for row in sorted(rows, key=lambda row: row["round"])
                        if row["engine"] == engine and row["query"] == query and row["variant"] == variant
                    ]
                )
        comparisons = {}
        for label, baseline in (("head_v2_vs_base_v2", "base-v2"), ("head_v2_vs_head_v1", "head-v1")):
            ratio = math.exp(statistics.mean(math.log(row[label]) for row in per_query))
            low, high = ratio_interval(samples_by_variant["head-v2"], samples_by_variant[baseline])
            comparisons[label] = {
                "ratio": ratio,
                "interval_95": [low, high],
                "verdict": "improved" if high < 1 else "regressed" if low > 1 else "unclear",
            }
        result["engines"][engine] = {"queries": per_query, "comparisons": comparisons}
    return result


def markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        f"# Scan comparison: {manifest['suite']} SF={manifest['scale_factor'] or 'n/a'}",
        "",
        f"Head `{manifest['head_sha']}`; base `{manifest['base_sha']}`.",
        f"{manifest['rounds']} reversed-order rounds; {manifest['iterations']} iterations per process; "
        f"first {manifest['warmup']} excluded. {manifest['threads']} workers, "
        f"{len(manifest['affinity'])} CPUs in the affinity mask; identical prepared files.",
        "",
        "Ratios below 1 mean faster head V2. "
        "Intervals resample paired process rounds and warm samples within each query.",
        "",
        "| Engine | Queries | Head V2 / base V2 (95% interval) | Head V2 / head V1 (95% interval) |",
        "| --- | ---: | --- | --- |",
    ]
    for engine, data in summary["engines"].items():
        cells = []
        for label in ("head_v2_vs_base_v2", "head_v2_vs_head_v1"):
            value = data["comparisons"][label]
            low, high = value["interval_95"]
            cells.append(f"{value['ratio']:.3f} ({low:.3f}–{high:.3f}), {value['verdict']}")
        lines.append(f"| {engine} | {len(data['queries'])} | {' | '.join(cells)} |")
    for engine, data in summary["engines"].items():
        lines.extend(
            [
                "",
                f"## {engine}",
                "",
                "| Query | Base V2 ms | Head V2 ms | Head V1 ms | V2/base | V2/V1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in data["queries"]:
            values = row["median_ms"]
            lines.append(
                f"| {row['query']} | {values['base-v2']:.3f} | {values['head-v2']:.3f} | "
                f"{values['head-v1']:.3f} | {row['head_v2_vs_base_v2']:.3f} | "
                f"{row['head_v2_vs_head_v1']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("tpch", "tpcds", "clickbench"))
    parser.add_argument("--scale-factor")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--engines", default="duckdb,datafusion")
    parser.add_argument("--queries", help="Comma-separated query indices; omitted runs the complete suite")
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--opt", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.iterations or args.rounds < 1 or args.threads < 1:
        parser.error("Require 0 <= warmup < iterations and positive rounds/threads")
    engines = args.engines.split(",")
    if len(set(engines)) != len(engines) or not set(engines) <= {"duckdb", "datafusion"}:
        parser.error("Engines must be unique members of duckdb,datafusion")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "tmp"
    temporary.mkdir(exist_ok=True)
    options = list(args.opt)
    if args.scale_factor:
        options.append(f"scale-factor={args.scale_factor}")
    manifest: dict[str, Any] = {
        "suite": args.suite,
        "scale_factor": args.scale_factor,
        "head_sha": args.head_sha,
        "base_sha": args.base_sha,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "threads": args.threads,
        "options": options,
        "affinity": sorted(os.sched_getaffinity(0)),
        "binaries": {},
        "queries": {},
        "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    rows: list[dict[str, Any]] = []
    for engine in engines:
        binaries = {
            "base": args.base_dir.resolve() / f"{engine}-bench",
            "head": args.head_dir.resolve() / f"{engine}-bench",
        }
        envs = {label: binary_environment(binary, args.threads, temporary) for label, binary in binaries.items()}
        queries_by_revision = {}
        for label, binary in binaries.items():
            manifest["binaries"][f"{label}-{engine}"] = {
                "path": str(binary),
                "bytes": binary.stat().st_size,
                **binary_provenance(binary, args.base_sha if label == "base" else args.head_sha),
                "elf_notes": subprocess.check_output(["readelf", "-n", str(binary)], text=True),
            }
            discover = command(binary, args.suite, options, engine, args.threads, 2) + ["--print-queries"]
            queries_by_revision[label] = [
                int(line) for line in subprocess.check_output(discover, env=envs[label], text=True).splitlines()
            ]
        if queries_by_revision["base"] != queries_by_revision["head"]:
            raise ValueError(f"Base and head enumerate different queries for {engine}")
        available = queries_by_revision["head"]
        queries = [int(query) for query in args.queries.split(",")] if args.queries else available
        if not queries or len(set(queries)) != len(queries) or not set(queries) <= set(available):
            raise ValueError(f"Invalid requested query set for {engine}: {queries}")
        manifest["queries"][engine] = queries
        write_json(output / "manifest.json", manifest)
        for turn in range(args.rounds):
            for index, query in enumerate(queries):
                start = index % len(VARIANTS)
                order = VARIANTS[start:] + VARIANTS[:start]
                if turn % 2:
                    order = order[::-1]
                for variant in order:
                    revision, version = variant.split("-")
                    binary = binaries[revision]
                    stem = output / f"{engine}-q{query}-r{turn}-{variant}"
                    cmd = command(binary, args.suite, options, engine, args.threads, int(version[1:]))
                    cmd.extend(
                        [
                            "--queries",
                            str(query),
                            "--iterations",
                            str(args.iterations),
                            "--hide-progress-bar",
                            "--display-format",
                            "gh-json",
                            "-o",
                            str(stem.with_suffix(".jsonl")),
                        ]
                    )
                    print(f"START {engine} q{query} round={turn} {variant}", flush=True)
                    log_path = stem.with_suffix(".log")
                    try:
                        with log_path.open("w", encoding="utf-8") as log:
                            subprocess.run(cmd, env=envs[revision], stdout=log, stderr=subprocess.STDOUT, check=True)
                    except subprocess.CalledProcessError:
                        print(log_path.read_text(encoding="utf-8")[-8000:], flush=True)
                        raise
                    value = measurement(
                        stem.with_suffix(".jsonl"),
                        query=query,
                        engine=engine,
                        commit=manifest["checkout_commit"],
                        iterations=args.iterations,
                        warmup=args.warmup,
                    )
                    rows.append(
                        {"engine": engine, "query": query, "round": turn, "variant": variant, "command": cmd, **value}
                    )
                    write_json(output / "measurements.json", rows)
                    print(
                        f"DONE {engine} q{query} {variant}: {statistics.median(value['warm_ns']) / 1e6:.3f} ms",
                        flush=True,
                    )
    query_sets = list(manifest["queries"].values())
    if any(queries != query_sets[0] for queries in query_sets):
        raise ValueError("Engines enumerate different query sets; report them separately")
    summary = summarize(rows, engines, query_sets[0], args.rounds)
    write_json(output / "summary.json", summary)
    report = markdown(summary, manifest)
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
