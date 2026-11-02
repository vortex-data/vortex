# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Collect independent benchmark processes and check the spread of their reported metrics.

The command must write JSON records with a `uri` and the selected value field below
CODSPEED_CARGO_WORKSPACE_ROOT. Each invocation gets a fresh directory. Walltime Divan
binaries write target/codspeed/walltime/raw_results/divan/*.json when CODSPEED_ENV is set.
Simulation commands must export their instrument's values, never process elapsed time.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any


def read_measurements(directory: Path, pattern: str, field: str, scale: float) -> dict[str, float]:
    measurements = {}
    for path in sorted(directory.glob(pattern)):
        record = json.loads(path.read_text())
        value = record
        for component in field.split("."):
            value = value[component]
        value = float(value) * scale
        uri = record["uri"]
        if not isinstance(uri, str) or not uri or uri in measurements:
            raise ValueError(f"missing or duplicate benchmark URI in {path}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid measurement for {uri}: {value}")
        measurements[uri] = value
    if not measurements:
        raise ValueError("command produced no benchmark results")
    return measurements


def summarize(
    reports: list[dict[str, Any]], minimum_runs: int, minimum_hosts: int, allow_rebuilds: bool
) -> dict[str, Any]:
    if not reports or any(not report["runs"] for report in reports):
        raise ValueError("reports contain no measurements")
    first = reports[0]
    for report in reports:
        for key in ("instrument", "value_field", "scale", "environment"):
            if report[key] != first[key]:
                raise ValueError(f"reports use different {key}")
        if not allow_rebuilds and report["command"] != first["command"]:
            raise ValueError("command changed between reports")
        if not allow_rebuilds and report["binary_sha256"] != first["binary_sha256"]:
            raise ValueError("binary identity changed, use --allow-rebuilds for a build comparison")
    runs = [run for report in reports for run in report["runs"]]
    expected = set(runs[0])
    if any(set(run) != expected for run in runs):
        raise ValueError("benchmark coverage changed between processes")
    hosts = len({report["host"] for report in reports})
    results = {}
    for uri in sorted(expected):
        values = [run[uri] for run in runs]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"invalid measurement for {uri}")
        spread = max(values) / min(values) - 1
        results[uri] = {
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "spread": spread,
            "passes": spread <= 0.05,
        }
    qualified = len(runs) >= minimum_runs and hosts >= minimum_hosts
    return {
        "processes": len(runs),
        "hosts": hosts,
        "enough_observations": qualified,
        "passes": qualified and all(result["passes"] for result in results.values()),
        "benchmarks": results,
    }


def binary_digest(binary: Path) -> str:
    with binary.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def collect(args: argparse.Namespace) -> None:
    binary = args.binary.resolve(strict=True)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("provide a command after --")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    digest = binary_digest(binary)
    cpu = Path("/proc/cpuinfo")
    report = {
        "instrument": args.instrument,
        "binary": str(binary),
        "binary_sha256": digest,
        "command": command,
        "value_field": args.value_field,
        "scale": args.scale,
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpuinfo": cpu.read_text().split("\n\n", 1)[0] if cpu.exists() else platform.processor(),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("DIVAN_", "MIMALLOC_", "MALLOC_"))
            or key
            in (
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "VORTEX_MAX_THREADS",
                "VORTEX_BENCH_VARIANT",
                "VORTEX_BENCH_PREFIX",
                "VORTEX_BENCH_SUFFIX",
            )
        },
        "rustc": subprocess.run(["rustc", "-Vv"], check=True, capture_output=True, text=True).stdout,
        "runs": [],
    }
    for index in range(args.repeats):
        directory = output / f"process-{index:03}"
        directory.mkdir()
        env = dict(os.environ, CODSPEED_CARGO_WORKSPACE_ROOT=str(directory))
        # This enables raw walltime export without uploading or starting an instrument.
        if args.instrument == "walltime":
            env["CODSPEED_ENV"] = "repeatability"
        with (directory / "process.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        values = read_measurements(directory, args.result_glob, args.value_field, args.scale)
        if report["runs"] and set(values) != set(report["runs"][0]):
            raise ValueError("benchmark coverage changed between processes")
        if binary_digest(binary) != digest:
            raise ValueError("benchmark binary changed during collection")
        report["runs"].append(values)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{index + 1}/{args.repeats}: {len(values)} benchmarks", flush=True)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("collect", help="run independent processes without uploading results")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--instrument", choices=["walltime", "simulation"], required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repeats", type=positive_int, default=30)
    run.add_argument("--timeout", type=positive_int, default=600)
    run.add_argument("--result-glob", default="target/codspeed/walltime/raw_results/divan/*.json")
    run.add_argument("--value-field", default="stats.min_ns")
    run.add_argument("--scale", type=float, default=1e-9)
    run.add_argument("command", nargs=argparse.REMAINDER)
    check = sub.add_parser("summarize", help="qualify results with a maximum 5% spread")
    check.add_argument("reports", nargs="+", type=Path)
    check.add_argument("--minimum-runs", type=positive_int, default=30)
    check.add_argument("--minimum-hosts", type=positive_int, default=3)
    check.add_argument("--allow-rebuilds", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "collect":
            collect(args)
        else:
            result = summarize(
                [json.loads(path.read_text()) for path in args.reports],
                args.minimum_runs,
                args.minimum_hosts,
                args.allow_rebuilds,
            )
            print(json.dumps(result, indent=2))
            raise SystemExit(0 if result["passes"] else 1)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
