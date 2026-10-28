# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Measure four fixed S3/model windows on one retained CI host, three hours apart."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import compare
import repeatability

QUERIES = {"tpch": [1, 2, 6, 13, 15, 22], "tpcds": [1, 4, 13, 20, 30, 31, 41, 61, 63, 80, 83, 92]}
HOST_KEYS = ("ci_run_id", "sha", "region", "instance_id", "cpus", "total_memory")
DATA_KEYS = ("source", "format", "scale_factor", "objects", "adaptive_io", "tail_cache_bytes", "model")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def initialize(args: argparse.Namespace):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "schedule.json"
    if path.exists():
        raise ValueError("A schedule already exists; do not overwrite completed measurements")
    if args.iterations < 6 or args.iterations % 6:
        raise ValueError("Use a multiple of six rounds to balance all backend orders")
    if not args.source_root.startswith("s3://"):
        raise ValueError("Time-series measurements require real S3")
    anchor = time.time()
    save(
        path,
        {
            "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
            "sha": os.environ.get("GITHUB_SHA"),
            "scope": "focused SF10 diagnostics; the separate full-suite workflow covers all 121 queries",
            "interval_seconds": 10800,
            "iterations": args.iterations,
            "source_root": args.source_root,
            "profile": str(args.profile.resolve()),
            "profile_sha256": digest(args.profile),
            "binary": str(args.binary.resolve()),
            "binary_sha256": digest(args.binary),
            "queries": QUERIES,
            "windows": [
                {"index": i + 1, "planned_start_unix": anchor + i * 10800, "status": "pending"} for i in range(4)
            ],
        },
    )
    print("Scheduled four measurement windows:", flush=True)
    for window in json.loads(path.read_text())["windows"]:
        print(window["index"], utc(window["planned_start_unix"]), flush=True)


def wait_until(target: float):
    while (remaining := target - time.time()) > 0:
        print(f"Waiting for {utc(target)}; {remaining:.0f} seconds remaining", flush=True)
        time.sleep(min(55, remaining))


def annotate(path: Path, window: dict):
    temporary = path.with_suffix(".tmp")
    with path.open() as source, temporary.open("w") as output:
        header = json.loads(next(source))
        if header.get("kind") != "environment":
            raise ValueError(f"Missing environment: {path}")
        header.update(
            measurement_window_id=str(window["index"]),
            measurement_window_planned_unix=window["planned_start_unix"],
            measurement_window_started_unix=window["started_unix"],
        )
        output.write(json.dumps(header) + "\n")
        shutil.copyfileobj(source, output)
    temporary.replace(path)


def summarize(records: list[dict], iterations: int) -> dict:
    result = repeatability.analyze(records)
    expected = {(suite, q, cache) for suite, queries in QUERIES.items() for q in queries for cache in ("cold", "warm")}
    actual = {(r["suite"], r["query"], r["cache"]) for r in result["rows"]}
    if actual != expected or any(not r["requests_match"] or r["iterations"] != iterations for r in result["rows"]):
        raise ValueError("Missing measurements or mismatched requests/results; do not fit latency")
    samples = [r for r in records if r.get("kind") == "sample"]
    groups = defaultdict(list)
    signatures = []
    requests = defaultdict(list)
    for sample in samples:
        key = (sample["suite"], sample["query"], sample["cache"])
        if sample["backend"] in ("real", "real_repeat"):
            groups[key].append(sample["seconds"] * 1000)
        signatures.append(
            (
                (*key, sample["iteration"], sample["backend"]),
                sorted(compare.signature(sample).items()),
                {
                    phase: {k: sample[phase][k] for k in ("get", "put", "list", "head", "bytes", "errors")}
                    for phase in ("counts", "setup_counts")
                },
                sample["rows"],
            )
        )
        requests[sample["backend"]].extend(r for r in sample["requests"] if r["operation"] == "GET")
    for row in result["rows"]:
        values = groups[(row["suite"], row["query"], row["cache"])]
        row["pooled_real_median_ms"] = statistics.median(values)
        row["pooled_real_p90_ms"] = compare.quantile(values, 0.9)
    result["request_fingerprint"] = hashlib.sha256(json.dumps(sorted(signatures), sort_keys=True).encode()).hexdigest()
    result["get_metrics"] = {
        backend: {
            "get": len(values),
            "bytes": sum(r["bytes"] for r in values),
            "header_mean_ms": statistics.mean(r["response_ms"] for r in values),
            "header_p90_ms": compare.quantile([r["response_ms"] for r in values], 0.9),
            "body_mean_ms": statistics.mean(r["duration_ms"] - r["response_ms"] for r in values),
            "duration_p90_ms": compare.quantile([r["duration_ms"] for r in values], 0.9),
        }
        for backend, values in requests.items()
    }
    return result


def measure(output: Path, index: int):
    path = output / "schedule.json"
    schedule = json.loads(path.read_text())
    if not 1 <= index <= 4:
        raise ValueError("Window index must be between one and four")
    window = schedule["windows"][index - 1]
    if window["status"] != "pending" or any(
        w["status"] in ("pending", "running") for w in schedule["windows"][: index - 1]
    ):
        raise ValueError("Windows must run once, sequentially")
    for key in ("profile", "binary"):
        if digest(Path(schedule[key])) != schedule[key + "_sha256"]:
            raise ValueError(f"Frozen {key} changed during the experiment")
    wait_until(window["planned_start_unix"])
    window.update(status="running", started_unix=time.time())
    window["lateness_seconds"] = window["started_unix"] - window["planned_start_unix"]
    save(path, schedule)
    destination = output / f"window-{index:02}"
    destination.mkdir(exist_ok=True)
    try:
        if window["lateness_seconds"] > 300:
            raise ValueError("Window missed its planned start by more than five minutes; preserve the missed cadence")
        env = {**os.environ, "S3_ITERATION_OVERRIDES": "{}", "S3_BASELINE_PROFILE": ""}
        for suite, queries in QUERIES.items():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("run.py")),
                    "--source-root",
                    schedule["source_root"],
                    "--phase",
                    "validate",
                    "--profile",
                    schedule["profile"],
                    "--binary",
                    schedule["binary"],
                    "--format",
                    "vortex",
                    "--scale-factor",
                    "10",
                    "--output",
                    str(destination),
                    "--iterations",
                    str(schedule["iterations"]),
                    "--suites",
                    suite,
                    "--queries",
                    ",".join(map(str, queries)),
                    "--round-robin",
                    "--real-repeat",
                    "--source-warmup-seconds",
                    "20",
                ],
                env=env,
                check=True,
            )
        traces = [destination / f"tuned-{suite}.jsonl" for suite in QUERIES]
        for trace in traces:
            annotate(trace, window)
        result = summarize(compare.read(traces), schedule["iterations"])
        host = {key: result["environments"][0][key] for key in HOST_KEYS}
        data = {e["suite"]: {key: e[key] for key in DATA_KEYS} for e in result["environments"]}
        if any({key: e[key] for key in HOST_KEYS} != host for e in result["environments"]):
            raise ValueError("Suites did not run on the same host")
        if "host" not in schedule:
            schedule.update(host=host, data=data, request_fingerprint=result["request_fingerprint"])
        if any(
            schedule[key] != value
            for key, value in (("host", host), ("data", data), ("request_fingerprint", result["request_fingerprint"]))
        ):
            raise ValueError("Hardware, objects, frozen model or physical I/O changed between windows")
        repeatability.write_reports(result, destination / "window")
        window.update(status="success", traces={t.name: digest(t) for t in traces})
    except Exception as error:
        window.update(status="failed", error=str(error))
        raise
    finally:
        window["finished_unix"] = time.time()
        save(path, schedule)


def report(output: Path):
    schedule = json.loads((output / "schedule.json").read_text())
    rows, baseline, metrics, correlations = [], {}, {}, {}
    for window in schedule["windows"]:
        if window["status"] != "success":
            continue
        result = json.loads((output / f"window-{window['index']:02}" / "window.json").read_text())
        metrics[str(window["index"])] = result["get_metrics"]
        for row in result["rows"]:
            key = (row["suite"], row["query"], row["cache"])
            first = baseline.setdefault(key, row)
            rows.append(
                {
                    "window": window["index"],
                    "started_utc": utc(window["started_unix"]),
                    **row,
                    "real_drift_vs_first": row["pooled_real_median_ms"] / first["pooled_real_median_ms"],
                    "model_drift_vs_first": row["model_median_ms"] / first["model_median_ms"],
                }
            )
        for cache in ("cold", "warm"):
            subset = [r for r in result["rows"] if r["cache"] == cache]
            correlations[f"{window['index']}/{cache}"] = compare.spearman(
                [r["pooled_real_median_ms"] for r in subset], [r["model_median_ms"] for r in subset]
            )
    complete = all(w["status"] == "success" for w in schedule["windows"])
    summary = {
        "complete": complete,
        "schedule": schedule,
        "get_metrics": metrics,
        "correlations": correlations,
        "rows": rows,
    }
    save(output / "over-time.json", summary)
    if rows:
        with (output / "over-time.csv").open("w") as stream:
            writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# S3 over time",
        "",
        f"Completed windows: {sum(w['status'] == 'success' for w in schedule['windows'])}/4.",
        "",
        "One retained host and immutable objects; the model stays frozen. Starts are three hours apart. "
        "End-to-end query medians are scan proxies. A learned candidate requires separate held-out validation.",
        "",
        "| Window | Planned UTC | Actual UTC | Status | Within ±10% |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for window in schedule["windows"]:
        subset = [r for r in rows if r["window"] == window["index"]]
        lines.append(
            f"| {window['index']} | {utc(window['planned_start_unix'])} | "
            f"{utc(window['started_unix']) if 'started_unix' in window else 'pending'} | {window['status']} | "
            f"{sum(r['median_within_10pct_target'] for r in subset)}/{len(subset)} |"
        )
    (output / "over-time.md").write_text("\n".join(lines) + "\n")
    if not complete:
        raise ValueError("Four successful windows are required; partial evidence has been preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--source-root", required=True)
    init.add_argument("--profile", type=Path, required=True)
    init.add_argument("--binary", type=Path, default=Path("target/release_debug/s3-validate"))
    init.add_argument("--iterations", type=int, default=12)
    for command in (init, commands.add_parser("measure"), commands.add_parser("report")):
        command.add_argument("--output", type=Path, default=Path("s3-over-time-results"))
        if command.prog.endswith("measure"):
            command.add_argument("--window", type=int, required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args)
    elif args.command == "measure":
        measure(args.output, args.window)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
