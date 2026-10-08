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


def parse_read(values: dict[str, int]) -> Read:
    end = values["completed_unix_ns"]
    if "queued_unix_ns" in values:
        return Read(
            values["queued_unix_ns"],
            values["started_unix_ns"],
            values["reading_unix_ns"],
            end,
            values["length"],
        )
    reading = end - values["read_ns"]
    started = reading - values["allocation_ns"]
    return Read(started - values["queue_ns"], started, reading, end, values["length"])


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)] if ordered else 0.0


def summarize(
    reads: list[Read], compute: list[tuple[int, int]], query: dict, starts: list[tuple[int, int]] | None = None
) -> dict:
    end = query["completed_unix_ns"]
    start = end - query["callback_ns"]
    selected = [read for read in reads if read.queued < end and read.completed > start]
    events: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    events[start]
    events[end]
    first_read = min((max(start, read.queued) for read in selected), default=end)
    last_read = max((min(end, read.completed) for read in selected), default=start)
    events[first_read]
    events[last_read]
    backlog = None
    backlog_updates = {}
    for timestamp, remaining in sorted(starts or []):
        if timestamp <= start:
            backlog = remaining
        elif timestamp < end:
            backlog_updates[timestamp] = remaining
            events[timestamp]

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
    interior_idle = 0
    idle_with_compute = 0
    idle_with_unstarted = 0
    longest_idle = 0
    gap_start = None
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
        if first_read <= previous < time <= last_read:
            if not depth[0] and not depth[1]:
                interior_idle += elapsed
                idle_with_compute += elapsed if depth[3] else 0
                idle_with_unstarted += elapsed if backlog else 0
                if gap_start is None:
                    gap_start = previous
                longest_idle = max(longest_idle, time - gap_start)
            else:
                gap_start = None
        depth = [value + change for value, change in zip(depth, changes)]
        peak = [max(value, maximum) for value, maximum in zip(depth, peak)]
        outstanding_peak = max(outstanding_peak, depth[0] + depth[1])
        if time in backlog_updates:
            backlog = backlog_updates[time]
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
        read_span = last_read - first_read
        result["io_idle"] = {
            "read_phase_ms": read_span / 1e6,
            "interior_empty_ms": interior_idle / 1e6,
            "interior_empty_percent": 100 * interior_idle / read_span if read_span else 0,
            "interior_empty_with_compute_ms": idle_with_compute / 1e6,
            "longest_interior_gap_ms": longest_idle / 1e6,
            "empty_with_unstarted_splits_ms": idle_with_unstarted / 1e6 if starts else None,
            "empty_with_unstarted_splits_percent": 100 * idle_with_unstarted / read_span
            if starts and read_span
            else None,
        }
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
    parser.add_argument(
        "--ignore-backlog",
        action="store_true",
        help="ignore per-scan backlog events when a query has multiple concurrent scan streams",
    )
    args = parser.parse_args()
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    fields = re.compile(
        r"\b(completed_unix_ns|queued_unix_ns|started_unix_ns|reading_unix_ns|queue_ns|allocation_ns|read_ns|compute_ns|callback_ns|iteration|query_idx|length|unstarted_splits)=(\d+)"
    )
    reads, compute, queries, starts = [], [], [], []
    for line in args.log.read_text().splitlines():
        values = {key: int(value) for key, value in fields.findall(ansi.sub("", line))}
        if "completed_unix_ns" not in values:
            continue
        end = values["completed_unix_ns"]
        if "read_ns" in values and "queue_ns" in values:
            reads.append(parse_read(values))
        elif "compute_ns" in values:
            compute.append((end - values["compute_ns"], end))
        elif "callback_ns" in values and values["iteration"] >= args.warmup:
            queries.append(values)
        elif "unstarted_splits" in values and not args.ignore_backlog:
            starts.append((end, values["unstarted_splits"]))
    if not reads or not compute or not queries:
        parser.error("log needs completed reads, compute events, and query callback windows after warmup")
    result = {
        "log": str(args.log),
        "warmup": args.warmup,
        "scope": (
            "Vortex compute-step wall time; explicit local object-store intervals cover blocking IO; "
            "legacy running IO includes allocation"
        ),
        "queries": [summarize(reads, compute, query, starts) for query in queries],
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
