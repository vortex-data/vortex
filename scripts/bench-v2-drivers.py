#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Compare DuckDB and DataFusion scan drivers with serial, alternating runs.

Build baseline and candidate binaries before running. Example:
  python3 scripts/bench-v2-drivers.py \
    --baseline duckdb=target/baseline/duckdb-bench \
    --candidate duckdb=target/release_debug/duckdb-bench \
    --baseline datafusion=target/baseline/datafusion-bench \
    --candidate datafusion=target/release_debug/datafusion-bench \
    --data-root vortex-bench/data --output target/driver-comparison \
    --workload tpch:6,19 --workload clickbench:0,1,23 \
    --threads 1 8 --prefetch 0 4

Raw iteration times, commands, environment, CPU affinity and source revision are
saved with each run. Warmup iterations are excluded from summaries. Parquet is a
control; DuckDB reopens its connection each iteration, matching its default.
File preparation windows apply only to DuckDB Vortex v2. DataFusion's
environment is set as well as --threads to support older baseline binaries.
Use --baseline-projection-prefetch-bytes and --candidate-projection-prefetch-bytes
to compare V2 projection read-ahead, including two settings of the same binary.
DataFusion's per-label scan-concurrency and io-threads flags require binaries with
the corresponding CLI controls.
Optional perf-stat processes run separately. Add --perf-branches to collect
branch counts and branch misses alongside instructions and cycles. The benchmark runner enables their
counters around query callbacks after warmup, excluding initial setup and result
reporting. DuckDB callbacks include connection reopening, whereas its reported
query time uses DuckDB's internal timer. Their timings are excluded from the comparison. Both binaries must
include the current runner's perf-control support.
Explicit query lists run as individual cases with perf-stat, giving each query its own counters.
"""

import argparse
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def binary(value: str) -> tuple[str, Path]:
    engine, separator, path = value.partition("=")
    if not separator or engine not in ("datafusion", "duckdb"):
        raise argparse.ArgumentTypeError("expected datafusion=PATH or duckdb=PATH")
    executable = Path(path).resolve()
    if not executable.is_file():
        raise argparse.ArgumentTypeError(f"binary does not exist: {executable}")
    return engine, executable


def workload(value: str) -> tuple[str, str]:
    suite, _, queries = value.partition(":")
    if suite not in ("tpch", "clickbench", "tpcds", "fineweb", "statpopgen"):
        raise argparse.ArgumentTypeError(f"unsupported suite: {suite}")
    if queries and any(not query.isdecimal() for query in queries.split(",")):
        raise argparse.ArgumentTypeError("query ids must be comma-separated integers")
    return suite, queries


def environment(engine: str, threads: int, scan: str, prefetch: int) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VORTEX_SCAN_", "VORTEX_LOCAL_", "DATAFUSION_"))
        and key
        not in (
            "VORTEX_DUCKDB_FILE_PREFETCH",
            "VORTEX_BENCH_SEGMENT_CACHE_MB",
            "VORTEX_BENCH_PRELOAD_SEGMENTS",
            "VORTEX_USE_SCAN_API",
            "VORTEX_BENCH_PERF_CONTROL",
            "VORTEX_BENCH_PERF_ACK",
            "VORTEX_BENCH_PERF_WARMUP",
        )
    }
    env["RUST_LOG"] = "warn"
    if scan == "v2":
        env["VORTEX_SCAN_V2"] = "1"
    if engine == "duckdb":
        env["VORTEX_DUCKDB_FILE_PREFETCH"] = str(prefetch)
    else:
        env["TOKIO_WORKER_THREADS"] = str(threads)
        env["DATAFUSION_EXECUTION_TARGET_PARTITIONS"] = str(threads)
    return env


def run(
    args: argparse.Namespace,
    engine: str,
    executable: Path,
    case: tuple[str, str],
    fmt: str,
    threads: int,
    scan: str,
    prefetch: int,
    label: str,
    round_id: int,
) -> list[dict[str, Any]]:
    suite, queries = case
    query_label = queries.replace(",", "_") or "all"
    if len(query_label) > 64:
        query_label = f"{len(queries.split(','))}q-{hashlib.sha256(queries.encode()).hexdigest()[:12]}"
    key = f"{suite}-{query_label}-{engine}-{fmt}-t{threads}-{scan}-p{prefetch}-{label}-r{round_id}"
    stem = args.output / key
    subdirectory = {
        "tpch": f"tpch/{args.scale_factor}",
        "tpcds": f"tpcds/{args.tpcds_scale_factor}",
        "clickbench": "clickbench_partitioned",
        "fineweb": "fineweb",
        "statpopgen": f"statpopgen/{args.statpopgen_scale_factor * 1000}",
    }[suite]
    data = args.data_root / subdirectory
    if not data.is_dir():
        raise ValueError(f"missing benchmark data: {data}")
    command = [
        str(executable),
        suite,
        "--formats",
        fmt,
        "--threads",
        str(threads),
        "--iterations",
        str(args.iterations),
        "--hide-progress-bar",
        "--display-format",
        "gh-json",
        "-o",
        str(stem.with_suffix(".jsonl")),
    ]
    if queries:
        command += ["--queries", queries]
    scan_concurrency = getattr(args, f"{label}_datafusion_scan_concurrency")
    if engine == "datafusion" and scan_concurrency is not None:
        command += ["--scan-concurrency", str(scan_concurrency)]
    io_threads = getattr(args, f"{label}_datafusion_io_threads")
    if engine == "datafusion" and io_threads is not None:
        command += ["--io-threads", str(io_threads)]
    if suite == "tpch":
        command += ["--opt", f"scale-factor={args.scale_factor}"]
    elif suite == "tpcds":
        command += ["--opt", f"scale-factor={args.tpcds_scale_factor}"]
    if suite == "statpopgen":
        if args.data_root != (Path.cwd() / "vortex-bench/data").resolve():
            raise ValueError("StatPopGen requires this repository's vortex-bench/data directory")
        command += ["--opt", f"scale-factor={args.statpopgen_scale_factor}"]
    else:
        command += ["--opt", f"remote-data-dir={data.as_uri()}/"]
    directory = "vortex-file-compressed" if fmt == "vortex" else fmt
    if not (data / directory).is_dir():
        raise ValueError(f"prepare {fmt} data before measuring: {data / directory}")
    env = environment(engine, threads, scan, prefetch)
    projection_budget = getattr(args, f"{label}_projection_prefetch_bytes", None)
    if projection_budget is not None and scan == "v2" and fmt != "parquet":
        env["VORTEX_SCAN_PROJECTION_PREFETCH_BYTES"] = str(projection_budget)
    if args.file_pruning and scan == "v2" and fmt != "parquet":
        env["VORTEX_SCAN_FILE_PRUNING"] = "1"
    metadata = {
        "command": command,
        "env": {
            key: value
            for key, value in env.items()
            if key.startswith(("VORTEX_", "DATAFUSION_", "TOKIO_")) or key == "RUST_LOG"
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load": os.getloadavg(),
    }
    stem.with_suffix(".command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"START {key}", flush=True)
    with stem.with_suffix(".log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = []
    for line in stem.with_suffix(".jsonl").read_text().splitlines():
        if not line.startswith("{"):
            continue
        result = json.loads(line)
        times = result["all_runtimes"]
        if len(times) != args.iterations:
            raise ValueError(f"unexpected iteration count in {stem}: {len(times)}")
        measured = [ns / 1e6 for ns in times[args.warmup :]]
        row = {
            "name": result["name"],
            "engine": engine,
            "suite": suite,
            "format": fmt,
            "threads": threads,
            "scan": scan,
            "prefetch": prefetch,
            "projection_prefetch_bytes": int(env.get("VORTEX_SCAN_PROJECTION_PREFETCH_BYTES", "0")),
            "datafusion_scan_concurrency": scan_concurrency if engine == "datafusion" else None,
            "label": label,
            "round": round_id,
            "times_ms": measured,
            "median_ms": statistics.median(measured),
        }
        rows.append(row)
        print(f"TIME {result['name']}: {row['median_ms']:.3f} ms", flush=True)
    if not rows:
        raise ValueError(f"no benchmark results in {stem}")
    if args.perf_stat:
        diagnostic_command = command.copy()
        diagnostic_command[diagnostic_command.index("-o") + 1] = str(stem.with_suffix(".perf.jsonl"))
        control = stem.with_suffix(".perf-control")
        ack = stem.with_suffix(".perf-ack")
        os.mkfifo(control)
        os.mkfifo(ack)
        perf_env = {
            **env,
            "VORTEX_BENCH_PERF_CONTROL": str(control),
            "VORTEX_BENCH_PERF_ACK": str(ack),
            "VORTEX_BENCH_PERF_WARMUP": str(args.warmup),
        }
        events = ["cycles:u", "instructions:u"]
        if args.perf_branches:
            events += ["branches:u", "branch-misses:u"]
        events += ["task-clock:u", "context-switches:u", "cpu-migrations:u", "page-faults:u"]
        perf_command = [
            "perf",
            "stat",
            "-x,",
            "-e",
            ",".join(events),
            "--delay=-1",
            f"--control=fifo:{control},{ack}",
            "-o",
            str(stem.with_suffix(".perf.csv")),
            "--",
            *diagnostic_command,
        ]
        stem.with_suffix(".perf.command.json").write_text(
            json.dumps(
                {
                    "command": perf_command,
                    "env": {
                        key: value
                        for key, value in perf_env.items()
                        if key.startswith(("VORTEX_", "DATAFUSION_", "TOKIO_"))
                    },
                    "measured_query_executions": len(rows) * (args.iterations - args.warmup),
                },
                indent=2,
            )
            + "\n"
        )
        try:
            with stem.with_suffix(".perf.log").open("w") as log:
                subprocess.run(perf_command, env=perf_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            control.unlink()
            ack.unlink()
        counts = {}
        for line in stem.with_suffix(".perf.csv").read_text().splitlines():
            fields = line.split(",")
            if len(fields) < 3:
                continue
            event = fields[2]
            if event not in ("cycles:u", "instructions:u"):
                event = event.removesuffix(":u")
            if event in (
                "instructions:u",
                "cycles:u",
                "branches",
                "branch-misses",
                "task-clock",
                "context-switches",
                "cpu-migrations",
                "page-faults",
            ):
                try:
                    counts[event] = float(fields[0])
                except ValueError as error:
                    if event == "instructions:u":
                        raise ValueError(
                            "No query perf counters; rebuild both binaries with the current benchmark runner"
                        ) from error
        if not counts.get("instructions:u"):
            raise ValueError(f"No instruction counts in {stem}; check perf support")
        executions = len(rows) * (args.iterations - args.warmup)
        per_execution = {event: value / executions for event, value in counts.items()}
        stem.with_suffix(".perf.metrics.json").write_text(
            json.dumps({"total": counts, "per_query_execution": per_execution}, indent=2) + "\n"
        )
        if len(rows) == 1:
            rows[0]["perf"] = per_execution
        print(f"PERF {key}: instructions/query={per_execution['instructions:u']:.0f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=binary, action="append", required=True)
    parser.add_argument("--candidate", type=binary, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=workload, action="append")
    parser.add_argument(
        "--formats", nargs="+", choices=("vortex", "vortex-compact", "parquet"), default=["vortex", "parquet"]
    )
    parser.add_argument("--threads", type=positive, nargs="+", default=[8])
    parser.add_argument("--scan", nargs="+", choices=("v1", "v2"), default=["v2"])
    parser.add_argument("--file-pruning", action="store_true", help="enable V2 whole-file statistics pruning")
    for label in ("baseline", "candidate"):
        parser.add_argument(
            f"--{label}-datafusion-scan-concurrency",
            type=positive,
            help=f"Vortex split tasks per available worker within each DataFusion file scan in the {label}",
        )
        parser.add_argument(f"--{label}-datafusion-io-threads", type=positive)
    parser.add_argument("--prefetch", type=int, nargs="+", default=[0, 4])
    parser.add_argument(
        "--baseline-projection-prefetch-bytes",
        type=int,
        default=0,
        help="V2 projection read-ahead byte budget per active filter split for the baseline",
    )
    parser.add_argument(
        "--candidate-projection-prefetch-bytes",
        type=int,
        default=0,
        help="V2 projection read-ahead byte budget per active filter split for the candidate",
    )
    parser.add_argument("--iterations", type=positive, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=positive, default=2)
    parser.add_argument("--scale-factor", default="10.0")
    parser.add_argument("--tpcds-scale-factor", default="1.0")
    parser.add_argument("--statpopgen-scale-factor", type=positive, default=1)
    parser.add_argument("--perf-stat", action="store_true")
    parser.add_argument("--perf-branches", action="store_true", help="also collect branch counters with perf-stat")
    args = parser.parse_args()
    if args.perf_branches and not args.perf_stat:
        parser.error("perf-branches requires perf-stat")
    if not 0 <= args.warmup < args.iterations:
        parser.error("warmup must be nonnegative and smaller than iterations")
    if any(not 0 <= value <= 256 for value in args.prefetch):
        parser.error("prefetch must be between 0 and 256")
    if min(args.baseline_projection_prefetch_bytes, args.candidate_projection_prefetch_bytes) < 0:
        parser.error("projection prefetch byte budgets must be nonnegative")
    binaries = {"baseline": dict(args.baseline), "candidate": dict(args.candidate)}
    if binaries["baseline"].keys() != binaries["candidate"].keys():
        parser.error("baseline and candidate must cover the same engines")
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "arguments": sys.argv[1:],
                "data_root": str(args.data_root),
                "warmup": args.warmup,
                "duckdb_connection_reuse": False,
                "segment_cache_mb": 0,
                "preload_segments": False,
                "perf_scope": "query callback; DuckDB includes connection reopening",
                "binaries": {
                    label: {engine: str(path) for engine, path in engines.items()}
                    for label, engines in binaries.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    (args.output / "candidate.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    (args.output / "benchmark-script.py").write_bytes(Path(__file__).read_bytes())
    perf_support = Path("vortex-bench/src/perf.rs")
    if perf_support.is_file():
        (args.output / "perf-support.rs").write_bytes(perf_support.read_bytes())
    records = []
    cases = args.workload or [("tpch", "6,19"), ("clickbench", "0,1,23")]
    if args.perf_stat:
        cases = [(suite, query) for suite, queries in cases for query in (queries.split(",") if queries else [""])]
    configs = itertools.product(cases, binaries["baseline"], args.formats, args.threads, args.scan)
    configs = list(configs)
    for round_id in range(args.rounds):
        for case, engine, fmt, threads, scan in configs:
            windows = args.prefetch if engine == "duckdb" and fmt != "parquet" and scan == "v2" else [0]
            for prefetch in windows:
                labels = ("baseline", "candidate") if round_id % 2 == 0 else ("candidate", "baseline")
                for label in labels:
                    records.extend(
                        run(args, engine, binaries[label][engine], case, fmt, threads, scan, prefetch, label, round_id)
                    )
                    (args.output / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    keys = sorted({(row["name"], row["engine"], row["threads"], row["scan"], row["prefetch"]) for row in records})
    summary = []
    for key in keys:
        medians = {
            label: statistics.median(
                [
                    time
                    for row in records
                    if (row["name"], row["engine"], row["threads"], row["scan"], row["prefetch"]) == key
                    and row["label"] == label
                    for time in row["times_ms"]
                ]
            )
            for label in binaries
        }
        delta = 100 * (medians["candidate"] / medians["baseline"] - 1)
        round_changes = []
        for round_id in range(args.rounds):
            paired = {
                label: statistics.median(
                    [
                        time
                        for row in records
                        if (row["name"], row["engine"], row["threads"], row["scan"], row["prefetch"]) == key
                        and row["label"] == label
                        and row["round"] == round_id
                        for time in row["times_ms"]
                    ]
                )
                for label in binaries
            }
            round_changes.append(100 * (paired["candidate"] / paired["baseline"] - 1))
        paired_delta = statistics.median(round_changes)
        comparison = {
            "name": key[0],
            "engine": key[1],
            "threads": key[2],
            "scan": key[3],
            "prefetch": key[4],
            **medians,
            "delta_percent": delta,
            "paired_delta_percent": paired_delta,
            "round_delta_percent": round_changes,
        }
        perf_rows = [
            row
            for row in records
            if (row["name"], row["engine"], row["threads"], row["scan"], row["prefetch"]) == key and "perf" in row
        ]
        if perf_rows:
            comparison["perf"] = {}
            for event in ("instructions:u", "cycles:u", "task-clock", "branches", "branch-misses"):
                values = {
                    label: statistics.median(row["perf"][event] for row in perf_rows if row["label"] == label)
                    for label in binaries
                    if all(event in row["perf"] for row in perf_rows)
                }
                if values:
                    comparison["perf"][event] = {
                        **values,
                        "delta_percent": (
                            100 * (values["candidate"] / values["baseline"] - 1) if values["baseline"] else None
                        ),
                    }
        summary.append(comparison)
        print(
            f"COMPARE {' '.join(map(str, key))}: paired {paired_delta:+.1f}% "
            f"(rounds: {', '.join(f'{change:+.1f}%' for change in round_changes)}); "
            f"pooled medians {medians['baseline']:.3f} -> {medians['candidate']:.3f} ms",
            flush=True,
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
