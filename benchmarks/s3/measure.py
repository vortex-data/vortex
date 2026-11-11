# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run up to four read-only S3 measurement windows at fixed three-hour intervals."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def save(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def wait_until(target: float):
    while (remaining := target - time.time()) > 0:
        print(
            f"Next S3 window at {datetime.fromtimestamp(target, UTC).isoformat()}; waiting {remaining:.0f}s", flush=True
        )
        time.sleep(min(55, remaining))


def read_window(path: Path) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records or records[0].get("kind") != "environment" or records[0].get("schema_version") != 1:
        raise ValueError("Missing versioned measurement environment")
    if records[-1].get("kind") != "complete" or records[-1].get("errors") != 0:
        raise ValueError("Incomplete or failed measurement; do not train on this window")
    environment = records[0]
    requests = [r for r in records if r.get("kind") == "request" and r["phase"] == "measure"]
    expected = (
        environment["rounds"]
        * len(environment["sizes"])
        * len(environment["concurrency"])
        * environment["requests_per_cell"]
    )
    if len(requests) != expected or any(
        r["status"] != "complete" or r["bytes"] != r["range"]["end"] - r["range"]["start"] for r in requests
    ):
        raise ValueError("Request coverage or consumed bytes differs from the declared workload")
    for request in requests:
        timings = [request[key] for key in ("response_ms", "first_byte_ms", "duration_ms")]
        if any(t is None or not math.isfinite(t) or t < 0 for t in timings) or timings != sorted(timings):
            raise ValueError("Incomplete or inconsistent request timing")
    groups = defaultdict(list)
    for request in requests:
        groups[(request["round"], request["bytes"], request["requested_concurrency"])].append(request)
    declared = {
        (round_, size, concurrency)
        for round_ in range(environment["rounds"])
        for size in environment["sizes"]
        for concurrency in environment["concurrency"]
    }
    if set(groups) != declared or any(len(values) != environment["requests_per_cell"] for values in groups.values()):
        raise ValueError("Missing or duplicated size/concurrency cell")
    if any(
        {r["ordinal"] for r in values} != set(range(environment["requests_per_cell"])) for values in groups.values()
    ):
        raise ValueError("Missing or duplicated request ordinal")
    cell_records = [r for r in records if r.get("kind") == "cell_summary"]
    summaries = {(r["round"], r["range_bytes"], r["requested_concurrency"]): r for r in cell_records}
    if set(summaries) != declared or len(cell_records) != len(declared):
        raise ValueError("Missing cell summaries")
    rows = []
    for (round_, size, concurrency), values in sorted(groups.items()):
        row = {
            "measurement_id": environment["measurement_id"],
            "round": round_,
            "bytes_per_request": size,
            "concurrency": concurrency,
            "requests": len(values),
        }
        cell = summaries[(round_, size, concurrency)]
        if cell["requests"] != len(values) or cell["bytes"] != sum(r["bytes"] for r in values) or cell["errors"]:
            raise ValueError("Cell counters disagree with the raw requests")
        if not math.isfinite(cell["seconds"]) or cell["seconds"] <= 0:
            raise ValueError("Invalid cell duration")
        row.update({key: cell[key] for key in ("seconds", "bytes", "errors", "peak_inflight", "avg_inflight")})
        for field, timings in (
            ("headers", [r["response_ms"] for r in values]),
            ("first_byte", [r["first_byte_ms"] for r in values]),
            ("body", [r["duration_ms"] - r["response_ms"] for r in values]),
            ("total", [r["duration_ms"] for r in values]),
        ):
            if any(t is None or not math.isfinite(t) or t < 0 for t in timings):
                raise ValueError("Incomplete or nonfinite request timing")
            row[field + "_median_ms"] = statistics.median(timings)
            row[field + "_p90_ms"] = quantile(timings, 0.9)
        row["aggregate_mib_per_second"] = cell["bytes"] / 1048576 / cell["seconds"]
        rows.append(row)
    return environment, rows


def configuration(args: argparse.Namespace) -> dict:
    return {
        key: getattr(args, key)
        for key in (
            "source",
            "count",
            "interval_seconds",
            "window_timeout_seconds",
            "seed",
            "allow_local",
            "probe_args",
        )
    }


def initialize(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory already contains evidence; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.binary.open("rb") as binary:
        binary_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    anchor = time.time()
    plan = {
        "schema_version": 1,
        "binary_sha256": binary_hash,
        "interval_seconds": args.interval_seconds,
        "source": args.source,
        "configuration": configuration(args),
        "windows": [
            {"index": i + 1, "planned_unix": anchor + i * args.interval_seconds, "status": "pending"}
            for i in range(args.count)
        ],
    }
    save(args.output / "schedule.json", plan)
    return plan


def load_plan(args: argparse.Namespace) -> dict:
    plan = json.loads((args.output / "schedule.json").read_text())
    if plan["configuration"] != configuration(args):
        raise ValueError("Probe configuration changed between measurement windows")
    with args.binary.open("rb") as binary:
        if hashlib.file_digest(binary, "sha256").hexdigest() != plan["binary_sha256"]:
            raise ValueError("Collector binary changed between measurement windows")
    return plan


def identity(environment: dict) -> dict:
    return {
        key: environment.get(key)
        for key in (
            "ci_run_id",
            "sha",
            "source",
            "region",
            "instance_id",
            "instance_type",
            "cpus",
            "total_memory",
            "objects",
            "sizes",
            "concurrency",
            "requests_per_cell",
            "rounds",
            "seed",
            "warmup_seconds",
            "endpoint_override",
        )
    }


def collect(args: argparse.Namespace, env: dict[str, str], plan: dict, index: int):
    window = plan["windows"][index - 1]
    if window["status"] != "pending":
        raise ValueError("Measurement window already attempted; evidence will not be overwritten")
    window.update(started_unix=time.time(), status="running")
    save(args.output / "schedule.json", plan)
    trace = args.output / f"window-{index:02}.jsonl"
    try:
        if window["started_unix"] - window["planned_unix"] > 300:
            raise ValueError("Window missed its planned start by over five minutes")
        if window["started_unix"] < window["planned_unix"]:
            raise ValueError("Wait for the planned measurement start before collecting")
        command = [
            str(args.binary.resolve()),
            "--source",
            args.source,
            "--output",
            str(trace),
            "--measurement-id",
            f"{env.get('GITHUB_RUN_ID', 'local')}-{env.get('GITHUB_RUN_ATTEMPT', '1')}-{index}",
            "--seed",
            str(args.seed),
        ]
        if args.allow_local:
            command += ["--allow-local"]
        command += args.probe_args
        subprocess.run(command, env=env, check=True, timeout=args.window_timeout_seconds)
        environment, _ = read_window(trace)
        for previous in plan["windows"]:
            if previous["status"] == "success":
                reference, _ = read_window(args.output / f"window-{previous['index']:02}.jsonl")
                if identity(reference) != identity(environment):
                    raise ValueError("Host, objects or probe configuration changed between windows")
        window.update(status="success", sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        window.update(status="failed", error=str(error))
    window["finished_unix"] = time.time()
    save(args.output / "schedule.json", plan)
    summarize(args, plan, require_complete=False)
    if window["status"] != "success":
        raise ValueError("Measurement window failed; partial evidence was preserved")


def summarize(args: argparse.Namespace, plan: dict, *, require_complete: bool = True):
    rows = []
    for window in plan["windows"]:
        if window["status"] == "success":
            trace = args.output / f"window-{window['index']:02}.jsonl"
            if hashlib.sha256(trace.read_bytes()).hexdigest() != window["sha256"]:
                raise ValueError("Measurement trace changed after collection")
            rows.extend(read_window(trace)[1])
    if rows:
        with (args.output / "measurements.csv").open("w") as stream:
            writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if require_complete and any(w["status"] != "success" for w in plan["windows"]):
        raise ValueError("Some measurement windows failed; partial evidence was preserved")


def run(args: argparse.Namespace, env: dict[str, str]):
    plan = initialize(args)
    for window in plan["windows"]:
        wait_until(window["planned_unix"])
        try:
            collect(args, env, load_plan(args), window["index"])
        except ValueError:
            # Keep the remaining fixed starts even when an earlier collection failed.
            continue
    summarize(args, load_plan(args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("s3-measurements"))
    parser.add_argument("--binary", type=Path, default=Path("target/release/s3-measure"))
    parser.add_argument("--count", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--interval-seconds", type=float, default=10800)
    parser.add_argument("--window-timeout-seconds", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("probe_args", nargs=argparse.REMAINDER)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initialize", action="store_true", help="Save the fixed schedule without collecting")
    mode.add_argument("--wait-window", type=int, choices=range(1, 5), help="Wait before the CI credential action")
    mode.add_argument("--run-window", type=int, choices=range(1, 5), help="Collect with current ambient credentials")
    mode.add_argument("--summarize", action="store_true", help="Validate all windows and write the combined table")
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds < 0:
        parser.error("Interval must be finite and nonnegative")
    if not math.isfinite(args.window_timeout_seconds) or args.window_timeout_seconds <= 0:
        parser.error("Window timeout must be finite and positive")
    if args.probe_args[:1] == ["--"]:
        args.probe_args = args.probe_args[1:]
    if args.initialize:
        initialize(args)
    elif args.wait_window or args.run_window or args.summarize:
        plan = load_plan(args)
        index = args.wait_window or args.run_window
        if index is not None and index > len(plan["windows"]):
            parser.error("Window is outside the initialized schedule")
        if args.wait_window:
            wait_until(plan["windows"][index - 1]["planned_unix"])
        elif args.run_window:
            collect(args, os.environ.copy(), plan, index)
        else:
            summarize(args, plan)
    else:
        run(args, os.environ.copy())


if __name__ == "__main__":
    main()
