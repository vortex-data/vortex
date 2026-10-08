#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Compare scan I/O and warmed compressed-segment caches using query wall time.

Build datafusion-bench from the current sources first. Example:
  python3 scripts/bench-io.py --data-root /path/to/vortex-bench/data --output /tmp/io-run

Runs TPC-H Q19 SF10 and partitioned ClickBench Q23 by default. Measurements are
serial, with rotated configuration order. Two initial iterations per process are
excluded from timing. Diagnostic processes are separate from timing processes;
read-range coverage and exact scan counters describe their final iteration.
Completion traces can include cancelled background reads; discrepancies against
scan counters are saved explicitly, and overlap is a completion-window diagnostic.

The cached mode preloads all segments while visiting every table file before
query timing, then keeps Moka caches across opens, capped PER FILE.
It retains compressed bytes, not decoded arrays or query results. The adapter
control uses the same segment-source routing with NoOpSegmentCache. Direct mode
leaves the branch's default I/O routing unchanged. Cache hits are compute-only
only when the measured scan reports zero reads AND zero misses. The OS page cache
is left warm in all modes. Byte counts measure application reads, not disk traffic.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VARIANTS = {
    "v1": {},
    "v2": {"VORTEX_SCAN_V2": "1"},
    "late": {"VORTEX_SCAN_V2": "1", "VORTEX_SCAN_EARLY_ANNOUNCE": "0"},
}
TOGGLES = (
    "VORTEX_SCAN_V2",
    "VORTEX_SCAN_EARLY_ANNOUNCE",
    "VORTEX_SCAN_EXEC",
    "VORTEX_SCAN_BATCH_IO",
    "VORTEX_BENCH_SEGMENT_CACHE_MB",
    "VORTEX_USE_SCAN_API",
    "VORTEX_BENCH_PRELOAD_SEGMENTS",
    "RUST_LOG",
)
WORKLOADS = {
    **{f"tpch-q{query}": ("tpch", query) for query in range(1, 23)},
    **{f"clickbench-q{query}": ("clickbench", query) for query in range(43)},
}
DEFAULT_WORKLOADS = ["tpch-q19", "clickbench-q23"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def union_bytes(intervals: Iterable[tuple[int, int]]) -> int:
    total = end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def diagnostics(log: str) -> dict[str, Any]:
    iterations = []
    current = None
    for line in ANSI.sub("", log).splitlines():
        if line.startswith("IO_ITERATION_BEGIN"):
            current = {"metrics": collections.Counter(), "reads": [], "has_scan": False}
        elif current is not None and line.startswith("IO_METRIC"):
            current["has_scan"] = True
            match = re.search(r"name=(\S+) value=(\d+)", line)
            name, value = match[1], int(match[2])
            if name not in ("output_rows", "output_batches") and not name.startswith(
                ("io.", "vortex.io.", "vortex.file.segments.cache.")
            ):
                continue
            metrics = current["metrics"]
            if name.endswith("_max"):
                metrics[name] = max(metrics.get(name, value), value)
            elif name.endswith("_min"):
                metrics[name] = min(metrics.get(name, value), value)
            else:
                metrics[name] += value
        elif current is not None and "local object-store read" in line:
            fields = {
                k: int(v)
                for k, v in re.findall(
                    r"(offset|length|prepare_ns|allocation_ns|get_ns|queue_ns|read_ns|resume_ns)=(\d+)", line
                )
            }
            fields["path"] = re.search(r"path=(\S+)", line)[1]
            current["reads"].append(fields)
        elif current is not None and line.startswith("IO_ITERATION_END"):
            iterations.append(current)
            current = None
    if not iterations:
        raise ValueError("No complete IO_ITERATION diagnostics; rebuild datafusion-bench")
    last = iterations[-1]
    by_file = collections.defaultdict(list)
    for read in last["reads"]:
        by_file[read["path"]].append((read["offset"], read["offset"] + read["length"]))
    requested = sum(r["length"] for r in last["reads"])
    unique = sum(union_bytes(intervals) for intervals in by_file.values())
    metrics = dict(last["metrics"])
    required = (
        "vortex.io.read.duration_count",
        "vortex.io.read.total_size",
        "vortex.file.segments.cache.hits",
        "vortex.file.segments.cache.misses",
    )
    missing = [name for name in required if name not in metrics]
    if missing and last["has_scan"]:
        raise ValueError(f"Missing scan counters: {missing}")
    metadata_only = not last["has_scan"]
    if metadata_only:
        metrics.update({name: 0 for name in required})
    return {
        "iterations": len(iterations),
        "metrics": metrics,
        "metadata_only": metadata_only,
        "local_reads": len(last["reads"]),
        "requested_bytes": requested,
        "distinct_bytes": unique,
        "repeated_bytes": requested - unique,
        # Cancelled reads can complete on the blocking pool after their scan future is dropped.
        # Preserve the discrepancy instead of treating completion-log events as scan counters.
        "trace_minus_scan_reads": len(last["reads"]) - metrics[required[0]],
        "trace_minus_scan_bytes": requested - metrics[required[1]],
        "compute_only": (
            metrics[required[0]] == 0
            and metrics[required[1]] == 0
            and (metrics[required[2]] > 0 or metadata_only)
            and metrics[required[3]] == 0
            and not last["reads"]
        ),
        "read_size_bytes": {
            "min": min((r["length"] for r in last["reads"]), default=0),
            "median": statistics.median([r["length"] for r in last["reads"]] or [0]),
            "max": max((r["length"] for r in last["reads"]), default=0),
        },
        "read_phases_median_us": {
            phase: statistics.median([r[phase] / 1000 for r in last["reads"]] or [0])
            for phase in ("get_ns", "queue_ns", "read_ns", "resume_ns")
        },
        "files": {
            path: {
                "reads": len(ranges),
                "requested_bytes": sum(b - a for a, b in ranges),
                "distinct_bytes": union_bytes(ranges),
            }
            for path, ranges in sorted(by_file.items())
        },
    }


def run(
    args: argparse.Namespace,
    workload: str,
    config: tuple[str, str],
    tag: str,
    diagnostic: bool = False,
) -> dict[str, Any]:
    variant, mode = config
    suite, query = WORKLOADS[workload]
    data = args.data_root / (f"tpch/{args.scale_factor}" if suite == "tpch" else "clickbench_partitioned")
    stem = args.output / f"{workload}-{variant}-{mode}-{tag}"
    command = [
        str(args.binary),
        suite,
        "--formats",
        "vortex",
        "--queries",
        str(query),
        "--iterations",
        str(3 if diagnostic else args.iterations),
        "--hide-progress-bar",
        "--display-format",
        "gh-json",
        "--opt",
        f"remote-data-dir={data.as_uri()}/",
        "-o",
        str(stem.with_suffix(".jsonl")),
    ]
    if suite == "tpch":
        command += ["--opt", f"scale-factor={args.scale_factor}"]
    env = {k: v for k, v in os.environ.items() if k not in TOGGLES}
    overrides = dict(VARIANTS[variant])
    if mode != "direct":
        overrides["VORTEX_BENCH_SEGMENT_CACHE_MB"] = str(args.cache_mb if mode == "cached" else 0)
    if mode == "cached":
        overrides["VORTEX_BENCH_PRELOAD_SEGMENTS"] = "1"
    overrides["RUST_LOG"] = "warn,vortex_io::read_timing=debug" if diagnostic else "warn"
    if diagnostic:
        command += ["--io-diagnostics"]
    env.update(overrides)
    metadata = {
        "command": command,
        "env": overrides,
        "load": os.getloadavg(),
        "datafusion_env": {k: v for k, v in env.items() if k.startswith("DATAFUSION_")},
    }
    stem.with_suffix(".command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"START {stem.name}", flush=True)
    with stem.with_suffix(".log").open("w") as output:
        subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    if diagnostic:
        result = diagnostics(stem.with_suffix(".log").read_text())
        print(
            f"IO {stem.name}: reads={result['local_reads']} "
            f"MB={result['requested_bytes'] / 1e6:.2f} "
            f"repeated_MB={result['repeated_bytes'] / 1e6:.2f} "
            f"compute_only={result['compute_only']}",
            flush=True,
        )
        return result
    rows = [json.loads(line) for line in stem.with_suffix(".jsonl").read_text().splitlines() if line.startswith("{")]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one query result in {stem}")
    times = [value / 1e6 for value in rows[0]["all_runtimes"]]
    if len(times) != args.iterations:
        raise ValueError(f"Unexpected iteration count in {stem}: {len(times)}")
    measured = times[2:]
    print(f"TIME {stem.name}: median={statistics.median(measured):.3f} ms", flush=True)
    return {"all_ms": times, "measured_ms": measured, "median_ms": statistics.median(measured)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release_debug/datafusion-bench"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=WORKLOADS,
        default=DEFAULT_WORKLOADS,
        help="TPC-H queries are 1-indexed; ClickBench queries are 0-indexed",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--modes", nargs="+", choices=["direct", "adapter", "cached"], default=["direct", "adapter", "cached"]
    )
    parser.add_argument("--scale-factor", default="10.0")
    parser.add_argument("--cache-mb", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 3 or args.rounds < 1 or args.cache_mb < 1:
        parser.error("Require iterations >= 3, rounds >= 1, cache-mb >= 1")
    args.binary = args.binary.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = [(variant, mode) for mode in args.modes for variant in args.variants]
    with args.binary.open("rb") as binary:
        sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    summary = {
        "binary": str(args.binary),
        "sha256": sha256,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": {},
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for workload in args.workloads:
        results = {f"{v}-{m}": {"rounds": []} for v, m in configs}
        summary["results"][workload] = results
        for variant, mode in configs:
            result = run(args, workload, (variant, mode), "diagnostic", diagnostic=True)
            results[f"{variant}-{mode}"]["io"] = result
            save()
            if mode == "cached" and not result["compute_only"]:
                raise ValueError(f"{workload}/{variant}: cache is not warm; inspect counters before timing")
        for rnd in range(args.rounds):
            rotated = configs[rnd % len(configs) :] + configs[: rnd % len(configs)]
            if rnd % 2:
                rotated = list(reversed(rotated))
            for variant, mode in rotated:
                result = run(args, workload, (variant, mode), f"round{rnd}")
                entry = results[f"{variant}-{mode}"]
                entry["rounds"].append(result)
                entry["median_ms"] = statistics.median(value for r in entry["rounds"] for value in r["measured_ms"])
                save()
        print(f"\n{workload}: query end-to-end milliseconds (first two iterations discarded)")
        print("configuration       median ms   reads   read MB   repeated MB")
        for name, result in results.items():
            io = result["io"]
            print(
                f"{name:20} {result['median_ms']:9.3f} {io['local_reads']:7} "
                f"{io['requested_bytes'] / 1e6:9.2f} {io['repeated_bytes'] / 1e6:12.2f}"
            )
        print(flush=True)
    print(f"Saved {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
