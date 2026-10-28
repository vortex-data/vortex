# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare explicit V1 and V2 shared-ring scans on the same local CI datasets."""

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

CANDIDATE = {
    "VORTEX_SCAN_V2": "1",
    "VORTEX_IO_BACKEND": "shared-1-mapped",
    "VORTEX_IO_QUEUE_DEPTH": "64",
    "VORTEX_IO_FORCE_ASYNC": "1",
    "VORTEX_IO_CHECK_RESIDENT_PAGES": "1",
    "VORTEX_IO_REUSE_INFLIGHT": "1",
    "VORTEX_SCAN_V2_SHARE_PLANS": "1",
    "VORTEX_SCAN_V2_IO_CACHE_BYTES": "0",
    "VORTEX_SCAN_V2_CHUNK_ROWS": "32768",
    "VORTEX_SCAN_V2_CONSUMER_MAPPING": "1",
    "VORTEX_SCAN_V2_CONSUMER_MAPPING_FIXED_WIDTH_ONLY": "1",
}


def configurations(inherited: dict[str, str]) -> dict[str, dict[str, str]]:
    env = {
        key: value
        for key, value in inherited.items()
        if not key.startswith(("VORTEX_IO_", "VORTEX_SCAN_V2"))
        and key not in {"VORTEX_SCAN_CONCURRENCY", "VORTEX_BENCH_BLOCKING_THREADS"}
    }
    env.update(
        BENCH_MAX_CPUS="32",
        TOKIO_WORKER_THREADS="32",
        DATAFUSION_EXECUTION_TARGET_PARTITIONS="32",
        VORTEX_USE_SCAN_API="0",
    )
    return {
        "v1": {**env, "VORTEX_SCAN_V2": "0", "VORTEX_IO_BACKEND": "legacy"},
        "v2-shared1": {**env, **CANDIDATE},
    }


def measurements(path: Path) -> dict[str, float]:
    values = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = row["name"]
        if name in values:
            raise ValueError(f"Duplicate query measurement: {name}")
        if row["unit"] != "ns" or len(row.get("all_runtimes", [])) < 2:
            raise ValueError(f"Expected nanosecond timings including warmup and measured runs: {name}")
        values[name] = statistics.median(row["all_runtimes"][1:]) / 1e6
    if not values:
        raise ValueError(f"No query measurements in {path}")
    return values


def report(directory: Path) -> str:
    baseline = measurements(directory / "v1.jsonl")
    candidate = measurements(directory / "v2-shared1.jsonl")
    if baseline.keys() != candidate.keys():
        raise ValueError("V1 and V2 must measure exactly the same queries and formats")
    ratios = [candidate[name] / value for name, value in baseline.items()]
    geomean = math.exp(statistics.mean(math.log(ratio) for ratio in ratios))
    rows = [
        "### V1 versus V2 with one shared io_uring",
        "",
        "Same commit and local data; at most 32 CPUs, 32 DataFusion partitions/workers. "
        "V2 uses the OS page cache, resident-page bypass, in-flight reuse, and 32K-row chunks. "
        "No direct IO, SQPOLL, or additional copied segment cache.",
        "",
        f"Hot median ratio V2/V1: **{geomean:.3f}×** across {len(ratios)} measurements; "
        f"{sum(ratio < 1 for ratio in ratios)} faster, "
        f"{sum(ratio > 1.05 for ratio in ratios)} slower by more than 5%.",
        "",
        "The first iteration is excluded. These are warm-cache runs; page-cache eviction "
        "is not performed. Ratios below 1 favor V2.",
        "",
        "| Query / format | V1 ms | V2 shared1 ms | V2 / V1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in baseline.items():
        rows.append(f"| {name} | {value:.3f} | {candidate[name]:.3f} | {candidate[name] / value:.3f}× |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--targets-json", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--opt", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("scan-io-comparison"))
    args = parser.parse_args()
    targets = [
        target
        for target in json.loads(args.targets_json)
        if target["engine"] == "datafusion" and target["format"] in {"vortex", "vortex-compact"}
    ]
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    if not targets:
        (directory / "report.md").write_text("No DataFusion Vortex targets in this benchmark case.\n")
        return
    if args.iterations < 2:
        parser.error("at least two iterations are required (warmup plus measured runs)")
    if any(option.startswith("remote-data-dir=") for option in args.opt):
        parser.error("this comparison requires the prepared local dataset")
    manifest = {"targets": targets, "iterations": args.iterations, "configurations": {}}
    for mode, env in configurations(dict(os.environ)).items():
        manifest["configurations"][mode] = {
            key: value
            for key, value in env.items()
            if key in {"BENCH_MAX_CPUS", "TOKIO_WORKER_THREADS", "DATAFUSION_EXECUTION_TARGET_PARTITIONS"}
            or key.startswith(("VORTEX_IO_", "VORTEX_SCAN_V2", "VORTEX_USE_SCAN_API"))
        }
        (directory / "configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")
        command = [
            "bash",
            "scripts/bench-taskset.sh",
            "uv",
            "run",
            "--project",
            "bench-orchestrator",
            "vx-bench",
            "run",
            args.benchmark,
            "--targets-json",
            json.dumps(targets),
            "--iterations",
            str(args.iterations),
            "--output",
            str(directory / f"{mode}.jsonl"),
            "--no-build",
        ]
        for option in args.opt:
            command.extend(["--opt", option])
        print(f"Running {mode}: {' '.join(command)}", flush=True)
        subprocess.run(command, env=env, check=True)
    (directory / "report.md").write_text(report(directory))


if __name__ == "__main__":
    main()
