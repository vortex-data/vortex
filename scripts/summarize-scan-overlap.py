# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Reconstruct IO queue depth and compute overlap from opt-in timing logs.

Enable vortex_io::read_timing, vortex_scan::compute_timing, and
vortex_bench::query_timing at DEBUG. Logging affects execution; use separate
processes with logging disabled for performance comparisons.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Read:
    queued: int
    started: int
    reading: int
    completed: int
    length: int


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)] if ordered else 0.0


def summarize(reads: list[Read], compute: list[tuple[int, int]], query: dict) -> dict:
    end = query["completed_unix_ns"]
    start = end - query["callback_ns"]
    selected = [read for read in reads if read.queued < end and read.completed > start]
    events: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    events[start]
    events[end]

    def interval(begin: int, stop: int, kind: int):
        begin, stop = max(begin, start), min(stop, end)
        if begin < stop:
            events[begin][kind] += 1
            events[stop][kind] -= 1

    for read in selected:
        interval(read.queued, read.started, 0)
        interval(read.started, read.completed, 1)
        interval(read.reading, read.completed, 2)
    for begin, stop in compute:
        interval(begin, stop, 3)

    depth = [0, 0, 0, 0]
    peak = [0, 0, 0, 0]
    integral = [0, 0, 0, 0]
    busy = [0, 0, 0, 0]
    compute_with_reads = 0
    compute_with_queued_io = 0
    outstanding_busy = 0
    outstanding_peak = 0
    previous = start
    for time, changes in sorted(events.items()):
        elapsed = time - previous
        for kind in range(4):
            integral[kind] += depth[kind] * elapsed
            busy[kind] += elapsed if depth[kind] else 0
        if depth[2]:
            compute_with_reads += depth[3] * elapsed
        if depth[0] or depth[1]:
            outstanding_busy += elapsed
            compute_with_queued_io += depth[3] * elapsed
        depth = [value + change for value, change in zip(depth, changes)]
        peak = [max(value, maximum) for value, maximum in zip(depth, peak)]
        outstanding_peak = max(outstanding_peak, depth[0] + depth[1])
        previous = time

    result = {
        "query_idx": query["query_idx"],
        "iteration": query["iteration"],
        "callback_ms": query["callback_ns"] / 1e6,
        "read_count": len(selected),
        "read_bytes": sum(read.length for read in selected),
    }
    if selected:
        last_read = min(end, max(read.completed for read in selected))
        result["last_read_completion_ms"] = (last_read - start) / 1e6
        result["callback_after_last_read_ms"] = (end - last_read) / 1e6
    for kind, name in enumerate(("queued_io", "running_io", "pread", "compute")):
        result[name] = {
            "peak": peak[kind],
            "mean_depth": integral[kind] / query["callback_ns"],
            "mean_depth_when_busy": integral[kind] / busy[kind] if busy[kind] else 0,
            "busy_percent": 100 * busy[kind] / query["callback_ns"],
            "thread_time_ms": integral[kind] / 1e6,
        }
    outstanding_time = integral[0] + integral[1]
    result["outstanding_io"] = {
        "peak": outstanding_peak,
        "mean_depth": outstanding_time / query["callback_ns"],
        "mean_depth_when_busy": outstanding_time / outstanding_busy if outstanding_busy else 0,
        "busy_percent": 100 * outstanding_busy / query["callback_ns"],
        "thread_time_ms": outstanding_time / 1e6,
    }
    result["compute_thread_time_with_pread_percent"] = 100 * compute_with_reads / integral[3] if integral[3] else 0
    result["compute_thread_time_with_outstanding_io_percent"] = (
        100 * compute_with_queued_io / integral[3] if integral[3] else 0
    )
    for name, durations in (
        ("queue", [read.started - read.queued for read in selected]),
        ("read", [read.completed - read.reading for read in selected]),
    ):
        result[f"{name}_duration_us"] = {
            "median": statistics.median(durations) / 1e3 if durations else 0,
            "p95": percentile(durations, 0.95) / 1e3,
            "max": max(durations, default=0) / 1e3,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    fields = re.compile(
        r"\b(completed_unix_ns|queue_ns|allocation_ns|read_ns|compute_ns|callback_ns|iteration|query_idx|length)=(\d+)"
    )
    reads, compute, queries = [], [], []
    for line in args.log.read_text().splitlines():
        values = {key: int(value) for key, value in fields.findall(ansi.sub("", line))}
        if "completed_unix_ns" not in values:
            continue
        end = values["completed_unix_ns"]
        if "read_ns" in values and "queue_ns" in values:
            reading = end - values["read_ns"]
            started = reading - values["allocation_ns"]
            reads.append(Read(started - values["queue_ns"], started, reading, end, values["length"]))
        elif "compute_ns" in values:
            compute.append((end - values["compute_ns"], end))
        elif "callback_ns" in values and values["iteration"] >= args.warmup:
            queries.append(values)
    if not reads or not compute or not queries:
        parser.error("log needs completed reads, compute events, and query callback windows after warmup")
    result = {
        "log": str(args.log),
        "warmup": args.warmup,
        "scope": "Vortex compute-step wall time; running IO includes buffer allocation; pread excludes it",
        "queries": [summarize(reads, compute, query) for query in queries],
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
