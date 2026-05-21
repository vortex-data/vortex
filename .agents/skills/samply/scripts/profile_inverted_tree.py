#!/usr/bin/env python3
"""Print an inverted call tree from a Samply / Firefox profiler JSON file."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path
from typing import Any

import profile_summary as ps


class Node:
    def __init__(self) -> None:
        self.weight = 0
        self.children: collections.defaultdict[str, Node] = collections.defaultdict(Node)


def sample_times(samples: dict[str, Any]) -> list[float]:
    total = 0.0
    times = []
    for delta in samples.get("timeDeltas") or []:
        if isinstance(delta, (int, float)):
            total += delta
        times.append(total)
    return times


def thread_matches(thread: dict[str, Any], thread_re: re.Pattern[str] | None, tids: set[str]) -> bool:
    if tids and str(thread.get("tid", "")) not in tids:
        return False
    if thread_re is None:
        return True
    return bool(thread_re.search(str(thread.get("name", ""))) or thread_re.search(str(thread.get("tid", ""))))


def frame_labels(thread: dict[str, Any], symbols: dict[str, str]) -> Any:
    cache: dict[int, str] = {}

    def label(frame_index: int) -> str:
        cached = cache.get(frame_index)
        if cached is None:
            cached = ps.frame_label(thread, frame_index, symbols)
            cache[frame_index] = cached
        return cached

    return label


def add_stack(root: Node, labels: list[str], weight: int) -> None:
    root.weight += weight
    node = root
    for label in reversed(labels):
        node = node.children[label]
        node.weight += weight


def print_tree(node: Node, total: int, limit: int, depth: int, indent: int = 0) -> None:
    if depth <= 0:
        return
    children = sorted(node.children.items(), key=lambda item: item[1].weight, reverse=True)
    for idx, (label, child) in enumerate(children):
        if idx >= limit:
            remaining = sum(other.weight for _, other in children[idx:])
            pct = remaining * 100.0 / total if total else 0.0
            print(f"{'  ' * indent}{remaining:10d} {pct:6.2f}%  <{len(children) - idx} more>")
            break
        pct = child.weight * 100.0 / total if total else 0.0
        print(f"{'  ' * indent}{child.weight:10d} {pct:6.2f}%  {label}")
        print_tree(child, total, limit, depth - 1, indent + 1)


def build_tree(
    profile: dict[str, Any],
    symbols: dict[str, str],
    args: argparse.Namespace,
) -> tuple[Node, list[tuple[str, str, int, int, float]]]:
    thread_re = re.compile(args.thread_regex) if args.thread_regex else None
    contains_re = re.compile(args.contains) if args.contains else None
    leaf_re = re.compile(args.leaf_regex) if args.leaf_regex else None
    tids = {str(tid) for tid in args.tid}

    root = Node()
    matched_threads = []

    for thread in profile.get("threads") or []:
        if not thread_matches(thread, thread_re, tids):
            continue

        samples = thread.get("samples") or {}
        stacks = samples.get("stack") or []
        times = sample_times(samples)
        label = frame_labels(thread, symbols)
        thread_weight = 0
        thread_samples = 0

        for i, stack_index in enumerate(stacks):
            if i >= len(times):
                continue
            time_ms = times[i]
            if args.start_ms is not None and time_ms < args.start_ms:
                continue
            if args.end_ms is not None and time_ms > args.end_ms:
                continue

            weight = ps.sample_weight(samples, i, args.weight_mode)
            if weight <= 0:
                continue

            frames = ps.expand_stack(thread, stack_index)
            if not frames:
                continue
            labels = [label(frame) for frame in frames]
            if contains_re and not any(contains_re.search(item) for item in labels):
                continue
            if leaf_re and not leaf_re.search(labels[-1]):
                continue

            add_stack(root, labels, weight)
            thread_weight += weight
            thread_samples += 1

        if thread_samples:
            matched_threads.append(
                (
                    str(thread.get("name", "")),
                    str(thread.get("tid", "")),
                    thread_samples,
                    thread_weight,
                    ps.thread_cpu_ms(thread),
                )
            )

    matched_threads.sort(key=lambda item: item[3], reverse=True)
    return root, matched_threads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--thread-regex", help="Regex matched against thread name or tid")
    parser.add_argument("--tid", action="append", default=[], help="Thread id to include; repeatable")
    parser.add_argument("--start-ms", type=float, help="Include samples at or after this profile time")
    parser.add_argument("--end-ms", type=float, help="Include samples at or before this profile time")
    parser.add_argument("--contains", help="Only include stacks containing a frame matching this regex")
    parser.add_argument("--leaf-regex", help="Only include stacks whose leaf frame matches this regex")
    parser.add_argument("--weight-mode", choices=("samples", "cpu"), default="cpu")
    parser.add_argument("--top", type=int, default=8, help="Max children to print at each tree level")
    parser.add_argument("--depth", type=int, default=12, help="Max inverted tree depth to print")
    parser.add_argument("--binary", type=Path, help="Candidate binary for symbolication")
    parser.add_argument("--symbolicate", action="store_true", help="Use atos to symbolicate raw app offsets")
    parser.add_argument(
        "--symbol-lib",
        help="Library name to symbolicate; defaults to the basename of --binary",
    )
    parser.add_argument("--load-address", default="0x100000000", help="Mach-O load address for atos")
    args = parser.parse_args()

    profile = ps.load_profile(args.profile)
    symbols: dict[str, str] = {}
    if args.symbolicate:
        if not args.binary:
            parser.error("--symbolicate requires --binary")
        symbol_lib = args.symbol_lib or args.binary.name
        addresses = ps.collect_symbol_addresses(profile, symbol_lib)
        symbols = ps.atos_symbol_map(args.binary, addresses, int(str(args.load_address), 0))

    root, matched_threads = build_tree(profile, symbols, args)
    unit = "cpu_us" if args.weight_mode == "cpu" else "samples"
    print(f"Profile: {args.profile}")
    print(f"weight_mode={args.weight_mode} total_{unit}={root.weight}")
    if args.start_ms is not None or args.end_ms is not None:
        start_ms = args.start_ms if args.start_ms is not None else "-inf"
        end_ms = args.end_ms if args.end_ms is not None else "inf"
        print(f"time_range_ms={start_ms}..{end_ms}")
    if args.thread_regex:
        print(f"thread_regex={args.thread_regex}")
    if args.tid:
        print(f"tids={','.join(args.tid)}")
    if args.contains:
        print(f"contains={args.contains}")
    if args.leaf_regex:
        print(f"leaf_regex={args.leaf_regex}")
    if args.symbolicate:
        print(f"symbolicated={len(symbols)}")

    print("\nMatched threads:")
    for name, tid, samples, weight, cpu_ms in matched_threads[:20]:
        print(f"  {weight:10d} {unit} samples={samples:6d} thread_cpu_ms={cpu_ms:10.3f} tid={tid} name={name}")
    if len(matched_threads) > 20:
        print(f"  <{len(matched_threads) - 20} more threads>")

    print("\nInverted call tree:")
    print_tree(root, root.weight, args.top, args.depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
