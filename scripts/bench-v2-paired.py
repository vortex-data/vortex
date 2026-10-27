#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Isolated PR #10849 experiment; apply identical cache controls to both revisions."""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

BASE = "a59294e35e7109fac894a7a38a68f42db0c3c967"
HEAD = "a5e4a997da7f8bbc61c2db9a9e03918656abde37"
CACHE_BYTES = 1024**3

CACHE_SETUP = """
// Benchmark-only control, applied identically to the base and candidate checkouts.
static BENCH_SEGMENT_CACHE: std::sync::LazyLock<Option<vortex::layout::segments::MokaSegmentCache>> =
    std::sync::LazyLock::new(|| {
        std::env::var("VORTEX_PAIRED_SEGMENT_CACHE_BYTES")
            .ok()
            .map(|value| {
                vortex::layout::segments::MokaSegmentCache::new(
                    value.parse::<u64>().expect("benchmark cache size must be an integer"),
                    vortex::layout::segments::SegmentEviction::Lru,
                )
            })
    });
"""


def checked_replace(path: Path, before: str, after: str):
    source = path.read_text()
    if source.count(before) != 1:
        raise RuntimeError(f"Expected exactly one patch anchor in {path}")
    path.write_text(source.replace(before, after))


def patch(root: Path):
    df = root / "vortex-datafusion/src/persistent/opener.rs"
    duck = root / "vortex-duckdb/src/file_reader.rs"
    for path in (df, duck):
        with path.open("a") as output:
            output.write(CACHE_SETUP)
    checked_replace(
        df,
        "            let cached_footer = file_metadata_cache",
        """            if let Some(cache) = BENCH_SEGMENT_CACHE.as_ref() {
                open_opts = open_opts.with_segment_cache(Arc::new(
                    cache.for_file(file.path().to_string()),
                ));
            }

            let cached_footer = file_metadata_cache""",
    )
    checked_replace(
        duck,
        "let file = open_cached(&SESSION, Some(&path), source, None, &|options| options).await?;",
        """let file = open_cached(&SESSION, Some(&path), source, None, &|options| {
            if let Some(cache) = BENCH_SEGMENT_CACHE.as_ref() {
                options.with_segment_cache(Arc::new(cache.for_file(path.clone())))
            } else {
                options
            }
        }).await?;""",
    )
    diff = subprocess.check_output(["git", "diff", "--", str(df), str(duck)], cwd=root)
    print(json.dumps({"benchmark_patch_sha256": hashlib.sha256(diff).hexdigest()}))


def geomean(values: Iterable[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def summarize(output: Path):
    manifest = json.loads((output / "manifest.json").read_text())
    rows = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines()]
    pairs = {}
    for row in rows:
        key = (row["engine"], row["mode"], row["query"], row["round"])
        pairs.setdefault(key, {})[row["revision"]] = row
    ratios = {}
    for (engine, mode, query, round_idx), pair in pairs.items():
        if set(pair) != {"base", "head"}:
            continue
        for measure in ["cold", "hot"] if mode == "uncached" else ["segment-cache"]:
            field = "cold_ns" if measure == "cold" else "hot_ns"
            ratio = pair["head"][field] / pair["base"][field]
            ratios.setdefault((engine, measure, query), []).append((round_idx, ratio))
    query_results = [
        {
            "engine": engine,
            "mode": mode,
            "query": query,
            "rounds": len(values),
            "ratio": statistics.median(ratio for _, ratio in values),
            "round_ratios": [ratio for _, ratio in sorted(values)],
        }
        for (engine, mode, query), values in sorted(ratios.items())
    ]
    groups = {}
    for row in query_results:
        groups.setdefault((row["engine"], row["mode"]), []).append(row)
    report = [
        "# Explicit V2 paired comparison",
        "",
        f"Base `{BASE}`; PR `{HEAD}`.",
        "",
        f"Negative latency change means faster. Median paired ratio across {manifest['rounds']} rounds;",
        "geometric mean across queries. Cold is the first iteration after a successful",
        "OS cache drop; hot excludes the first two iterations. Segment-cache mode uses",
        "a process-wide 1 GiB LRU byte cache and excludes its first two iterations.",
        "",
        "| Engine | Mode | Queries | Latency change |",
        "|---|---|---:|---:|",
    ]
    for (engine, mode), group in sorted(groups.items()):
        change = (geomean(row["ratio"] for row in group) - 1) * 100
        report.append(f"| {engine} | {mode} | {len(group)} | {change:+.2f}% |")
    failures = json.loads((output / "failures.json").read_text()) if (output / "failures.json").exists() else []
    if failures:
        report.extend(["", f"**Incomplete: {len(failures)} failed invocations. See failures.json.**"])
    if manifest.get("smoke"):
        report.extend(["", "**Smoke test only; these timings are not performance evidence.**"])
    (output / "summary.json").write_text(json.dumps(query_results, indent=2) + "\n")
    (output / "summary.md").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)


def run(args: argparse.Namespace):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary_root = args.binaries.resolve()
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("VORTEX_"):
            env.pop(key)
    env.update(TOKIO_WORKER_THREADS="8", DATAFUSION_EXECUTION_TARGET_PARTITIONS="8", RUST_LOG="error")
    options = ["--opt", f"scale-factor={args.scale}"] if args.suite in ("tpch", "tpcds") else []
    options.extend(["--runner", os.environ.get("RUNS_ON_INSTANCE_TYPE", "local")])
    rounds = 1 if args.smoke else 3
    iterations = 3 if args.smoke else 7
    failures = []
    failure_streak = 0
    (output / "measurements.jsonl").touch()
    manifest = {
        "base": BASE,
        "head": HEAD,
        "suite": args.suite,
        "scale": args.scale,
        "rounds": rounds,
        "iterations": iterations,
        "smoke": args.smoke,
        "warmup": 2,
        "scan_v2": True,
        "threads": 8,
        "segment_cache_bytes": CACHE_BYTES,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "binaries": {},
    }
    for revision, sha in (("base", BASE), ("head", HEAD)):
        metadata = json.loads((binary_root / revision / "manifest.json").read_text())
        if metadata["source_sha"] != sha:
            raise RuntimeError(f"Wrong {revision} source: {metadata}")
        manifest["binaries"][revision] = metadata
    patches = {m["benchmark_patch_sha256"] for m in manifest["binaries"].values()}
    if len(patches) != 1:
        raise RuntimeError("Base and head do not have the same benchmark patch")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for engine in ("datafusion", "duckdb"):
        query_sets = []
        for revision in ("base", "head"):
            binary = binary_root / revision / f"{engine}-bench"
            probe_env = {**env, "LD_LIBRARY_PATH": str(binary.parent)}
            query_sets.append(
                subprocess.check_output(
                    [str(binary), args.suite, "--print-queries", "--scan-v2", "true", *options],
                    env=probe_env,
                    text=True,
                ).split()
            )
        if query_sets[0] != query_sets[1]:
            raise RuntimeError(f"Different query lists for {engine}")
        queries = [int(query) for query in query_sets[0]]
        if args.smoke:
            queries = queries[1:2] or queries[:1]
        if not queries:
            raise RuntimeError(f"No queries selected for {engine}")
        for mode in ("uncached", "segment-cache"):
            for round_idx in range(rounds):
                for query in queries:
                    order = ("base", "head") if (round_idx + query) % 2 == 0 else ("head", "base")
                    for revision in order:
                        stem = f"{engine}-{mode}-r{round_idx}-q{query:03}-{revision}"
                        binary = binary_root / revision / f"{engine}-bench"
                        run_env = {**env, "LD_LIBRARY_PATH": str(binary.parent)}
                        if mode == "segment-cache":
                            run_env["VORTEX_PAIRED_SEGMENT_CACHE_BYTES"] = str(CACHE_BYTES)
                        command = [
                            str(binary),
                            args.suite,
                            "--formats",
                            "vortex",
                            "--scan-v2",
                            "true",
                            "--queries",
                            str(query),
                            "--iterations",
                            str(iterations),
                            "--threads",
                            "8",
                            "--display-format",
                            "gh-json",
                            "--hide-progress-bar",
                            *options,
                            "-o",
                            str(output / f"{stem}.jsonl"),
                        ]
                        if engine == "duckdb":
                            command.append("--delete-duckdb-database")
                        (output / f"{stem}.command.json").write_text(
                            json.dumps(
                                {
                                    "command": command,
                                    "env": {
                                        key: value
                                        for key, value in run_env.items()
                                        if key.startswith(("VORTEX_", "TOKIO_", "DATAFUSION_"))
                                    },
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        subprocess.run(["sync"], check=True)
                        subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
                        started = time.monotonic()
                        try:
                            with (output / f"{stem}.log").open("w") as log:
                                subprocess.run(
                                    command, env=run_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300
                                )
                            measurements = [
                                json.loads(line)
                                for line in (output / f"{stem}.jsonl").read_text().splitlines()
                                if line.startswith("{")
                            ]
                            measurement = next(row for row in measurements if row.get("kind") == "query_measurement")
                            samples = measurement["all_runtimes_ns"]
                            if len(samples) != iterations or any(value <= 0 for value in samples):
                                raise RuntimeError(f"Invalid timing samples: {samples}")
                            record = {
                                "engine": engine,
                                "mode": mode,
                                "query": query,
                                "round": round_idx,
                                "revision": revision,
                                "cold_ns": samples[0],
                                "hot_ns": statistics.median(samples[2:]),
                                "samples_ns": samples,
                                "elapsed_s": time.monotonic() - started,
                            }
                            with (output / "measurements.jsonl").open("a") as result:
                                result.write(json.dumps(record) + "\n")
                            failure_streak = 0
                            print(
                                f"{stem}: cold {samples[0] / 1e6:.3f} ms, hot {record['hot_ns'] / 1e6:.3f} ms",
                                flush=True,
                            )
                        except (
                            subprocess.SubprocessError,
                            ValueError,
                            KeyError,
                            StopIteration,
                            OSError,
                            RuntimeError,
                        ) as error:
                            failures.append({"invocation": stem, "error": str(error)})
                            failure_streak += 1
                            print(f"FAILED {stem}: {error}", flush=True)
                            log_path = output / f"{stem}.log"
                            if log_path.exists():
                                print(log_path.read_text(errors="replace")[-6000:], flush=True)
                            (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                            if args.smoke or failure_streak >= 3:
                                summarize(output)
                                raise SystemExit(
                                    "Stopping after smoke failure or three consecutive failures"
                                ) from error
                summarize(output)
    summarize(output)
    if failures:
        raise SystemExit(f"{len(failures)} benchmark invocations failed; partial results saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    patch_parser = commands.add_parser("patch")
    patch_parser.add_argument("root", type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--suite", required=True, choices=["tpch", "tpcds", "clickbench", "fineweb"])
    run_parser.add_argument("--scale", type=int, default=1)
    run_parser.add_argument("--binaries", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--smoke", action="store_true")
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "patch":
        patch(args.root.resolve())
    elif args.command == "summarize":
        summarize(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
