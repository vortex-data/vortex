#!/usr/bin/env python3
"""Summarize thread activity over time in a Samply / Firefox profiler JSON file."""

from __future__ import annotations

import argparse
import collections
import re
import statistics
from pathlib import Path
from typing import Any

import profile_summary as ps


def sample_times(samples: dict[str, Any]) -> list[float]:
    total = 0.0
    times = []
    for delta in samples.get("timeDeltas") or []:
        if isinstance(delta, (int, float)):
            total += delta
        times.append(total)
    return times


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def compact_ranges(bad_bins: list[int], bin_ms: float) -> list[tuple[float, float]]:
    if not bad_bins:
        return []
    ranges = []
    start = prev = bad_bins[0]
    for value in bad_bins[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start * bin_ms, (prev + 1) * bin_ms))
        start = prev = value
    ranges.append((start * bin_ms, (prev + 1) * bin_ms))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--thread-regex", default="tokio-rt-worker", help="Regex matched against thread name or tid")
    parser.add_argument("--bin-ms", type=float, default=10.0, help="Timeline bin width in milliseconds")
    parser.add_argument(
        "--active-cpu-us", type=float, default=100.0, help="CPU delta needed for a thread to count as active in a bin"
    )
    parser.add_argument("--low-active", type=int, default=4, help="Report ranges with at most this many active threads")
    parser.add_argument(
        "--tail-ms", type=float, default=200.0, help="Print active-thread counts for this much tail time"
    )
    parser.add_argument("--start-ms", type=float, help="Only include samples at or after this profile time")
    parser.add_argument("--end-ms", type=float, help="Only include samples at or before this profile time")
    parser.add_argument("--show-ranges", type=int, default=12, help="Maximum low-activity ranges to print")
    args = parser.parse_args()

    profile = ps.load_profile(args.profile)
    thread_re = re.compile(args.thread_regex) if args.thread_regex else None
    active_threads_by_bin: collections.defaultdict[int, set[str]] = collections.defaultdict(set)
    cpu_ms_by_bin: collections.defaultdict[int, float] = collections.defaultdict(float)
    thread_cpu_ms = []
    thread_count = 0
    sample_count = 0
    first_time = None
    last_time = 0.0

    for thread in profile.get("threads") or []:
        name = str(thread.get("name", ""))
        tid = str(thread.get("tid", ""))
        if thread_re and not (thread_re.search(name) or thread_re.search(tid)):
            continue
        thread_count += 1
        cpu_ms = ps.thread_cpu_ms(thread)
        thread_cpu_ms.append(cpu_ms)
        samples = thread.get("samples") or {}
        deltas = samples.get("threadCPUDelta") or []
        times = sample_times(samples)
        for i, time_ms in enumerate(times):
            if args.start_ms is not None and time_ms < args.start_ms:
                continue
            if args.end_ms is not None and time_ms > args.end_ms:
                continue
            first_time = time_ms if first_time is None else min(first_time, time_ms)
            last_time = max(last_time, time_ms)
            sample_count += 1
            cpu_us = deltas[i] if i < len(deltas) and isinstance(deltas[i], (int, float)) else 0
            if cpu_us <= 0:
                continue
            bin_index = int(time_ms // args.bin_ms)
            cpu_ms_by_bin[bin_index] += cpu_us / 1000.0
            if cpu_us >= args.active_cpu_us:
                active_threads_by_bin[bin_index].add(tid)

    max_bin = int(last_time // args.bin_ms) if last_time else 0
    active_counts = [len(active_threads_by_bin.get(i, set())) for i in range(max_bin + 1)]
    cpu_bins = [cpu_ms_by_bin.get(i, 0.0) for i in range(max_bin + 1)]
    hot_threads = [value for value in thread_cpu_ms if value > 10.0]

    print(f"Profile: {args.profile}")
    print(f"thread_regex={args.thread_regex!r} matched_threads={thread_count} samples={sample_count}")
    print(f"time_ms={first_time if first_time is not None else 0.0:.3f}..{last_time:.3f} bin_ms={args.bin_ms}")
    if hot_threads:
        print(
            "thread_cpu_ms "
            f"total={sum(thread_cpu_ms):.1f} hot_threads={len(hot_threads)} "
            f"min={min(hot_threads):.1f} p50={statistics.median(hot_threads):.1f} "
            f"p90={quantile(hot_threads, 0.90):.1f} max={max(hot_threads):.1f}"
        )
    if active_counts:
        print(
            "active_threads_per_bin "
            f"mean={statistics.mean(active_counts):.2f} median={statistics.median(active_counts):.1f} "
            f"p10={quantile(active_counts, 0.10):.1f} p90={quantile(active_counts, 0.90):.1f} "
            f"max={max(active_counts)} zero_bins={sum(1 for value in active_counts if value == 0)}"
        )
        print(
            "cpu_ms_per_bin "
            f"median={statistics.median(cpu_bins):.1f} p90={quantile(cpu_bins, 0.90):.1f} "
            f"max={max(cpu_bins):.1f}"
        )

    tail_bins = int(args.tail_ms // args.bin_ms)
    if tail_bins > 0 and active_counts:
        print(f"tail_active_threads last_{args.tail_ms:g}ms={active_counts[-tail_bins:]}")

    low_bins = [i for i, value in enumerate(active_counts) if value <= args.low_active]
    ranges = compact_ranges(low_bins, args.bin_ms)
    if ranges:
        print(f"low_activity_ranges active_threads<={args.low_active}:")
        for start, end in ranges[: args.show_ranges]:
            print(f"  {start:.1f}..{end:.1f} ms")
        if len(ranges) > args.show_ranges:
            print(f"  <{len(ranges) - args.show_ranges} more>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
