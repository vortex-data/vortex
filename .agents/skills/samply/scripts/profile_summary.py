#!/usr/bin/env python3
"""Summarize Samply / Firefox profiler JSON without opening the UI."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def load_profile(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def value_at(table: dict[str, Any], key: str, index: int, default: Any = None) -> Any:
    values = table.get(key)
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return default
    return values[index]


def decode_string(thread: dict[str, Any], value: Any) -> str:
    strings = thread.get("stringArray") or []
    if isinstance(value, int) and 0 <= value < len(strings):
        return str(strings[value])
    if value is None:
        return ""
    return str(value)


def resource_name(thread: dict[str, Any], resource_index: Any) -> str:
    if not isinstance(resource_index, int):
        return ""
    resource_table = thread.get("resourceTable") or {}
    name = value_at(resource_table, "name", resource_index)
    if name is None:
        name = value_at(resource_table, "lib", resource_index)
    return decode_string(thread, name)


ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def frame_label(thread: dict[str, Any], frame_index: int, symbols: dict[str, str] | None = None) -> str:
    frame_table = thread.get("frameTable") or {}
    func_table = thread.get("funcTable") or {}
    func_index = value_at(frame_table, "func", frame_index)
    if not isinstance(func_index, int):
        return f"<frame:{frame_index}>"

    name = decode_string(thread, value_at(func_table, "name", func_index))
    resource = resource_name(thread, value_at(func_table, "resource", func_index))
    file_name = decode_string(thread, value_at(func_table, "fileName", func_index))
    line = value_at(func_table, "lineNumber", func_index)

    label = symbols.get(name, name) if symbols else name
    label = label or f"<func:{func_index}>"
    if resource and resource not in label:
        label = f"{label} [{resource}]"
    if file_name:
        label = f"{label} ({file_name}:{line})" if line else f"{label} ({file_name})"
    return label


def expand_stack(thread: dict[str, Any], stack_index: Any) -> list[int]:
    if not isinstance(stack_index, int):
        return []

    stack_table = thread.get("stackTable") or {}
    frames: list[int] = []
    seen: set[int] = set()
    current: Any = stack_index

    while isinstance(current, int) and current not in seen:
        seen.add(current)
        frame = value_at(stack_table, "frame", current)
        if isinstance(frame, int):
            frames.append(frame)
        current = value_at(stack_table, "prefix", current)

    frames.reverse()
    return frames


def sample_weight(samples: dict[str, Any], index: int, weight_mode: str) -> int:
    if weight_mode == "cpu":
        deltas = samples.get("threadCPUDelta")
        if isinstance(deltas, list) and index < len(deltas) and isinstance(deltas[index], (int, float)):
            return max(int(deltas[index]), 0)

    weights = samples.get("weight")
    if isinstance(weights, list) and index < len(weights) and isinstance(weights[index], (int, float)):
        return int(weights[index])
    return 1


def thread_cpu_ms(thread: dict[str, Any]) -> float:
    deltas = (thread.get("samples") or {}).get("threadCPUDelta") or []
    return sum(value for value in deltas if isinstance(value, (int, float))) / 1000.0


def summarize_thread(
    thread: dict[str, Any],
    symbols: dict[str, str] | None = None,
    weight_mode: str = "samples",
) -> dict[str, Any]:
    samples = thread.get("samples") or {}
    stacks = samples.get("stack") or []

    total_weight = 0
    self_counts: collections.Counter[str] = collections.Counter()
    inclusive_counts: collections.Counter[str] = collections.Counter()
    stack_counts: collections.Counter[tuple[str, ...]] = collections.Counter()

    label_cache: dict[int, str] = {}

    def label(frame_index: int) -> str:
        cached = label_cache.get(frame_index)
        if cached is None:
            cached = frame_label(thread, frame_index, symbols)
            label_cache[frame_index] = cached
        return cached

    for i, stack_index in enumerate(stacks):
        weight = sample_weight(samples, i, weight_mode)
        if weight <= 0:
            continue
        total_weight += weight

        frames = expand_stack(thread, stack_index)
        if not frames:
            continue

        labels = tuple(label(frame) for frame in frames)
        self_counts[labels[-1]] += weight
        for frame in set(labels):
            inclusive_counts[frame] += weight
        stack_counts[labels] += weight

    return {
        "name": thread.get("name", ""),
        "tid": thread.get("tid", ""),
        "samples": samples.get("length", len(stacks)),
        "weight": total_weight,
        "cpu_ms": thread_cpu_ms(thread),
        "self": self_counts,
        "inclusive": inclusive_counts,
        "stacks": stack_counts,
    }


def print_counter(title: str, counter: collections.Counter[str], total: int, limit: int) -> None:
    print(title)
    if total <= 0:
        print("  <no samples>")
        return
    for name, count in counter.most_common(limit):
        pct = count * 100.0 / total
        print(f"  {count:8d} {pct:6.2f}%  {name}")


def print_stacks(counter: collections.Counter[tuple[str, ...]], total: int, limit: int, depth: int) -> None:
    print("Top stacks:")
    if total <= 0:
        print("  <no samples>")
        return
    for stack, count in counter.most_common(limit):
        pct = count * 100.0 / total
        suffix = stack[-depth:]
        print(f"  {count:8d} {pct:6.2f}%")
        for frame in suffix:
            print(f"      {frame}")


def print_libs(profile: dict[str, Any]) -> None:
    print("Libraries:")
    for lib in profile.get("libs") or []:
        name = lib.get("name", "")
        path = lib.get("path", "")
        code_id = lib.get("codeId", "")
        arch = lib.get("arch", "")
        print(f"  {name} [{arch}] codeId={code_id} path={path}")


def binary_uuid(path: Path) -> str:
    try:
        result = subprocess.run(
            ["dwarfdump", "--uuid", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def collect_symbol_addresses(profile: dict[str, Any], lib_name: str) -> list[str]:
    addresses: set[str] = set()
    for thread in profile.get("threads") or []:
        func_table = thread.get("funcTable") or {}
        for i, name_index in enumerate(func_table.get("name") or []):
            name = decode_string(thread, name_index)
            if not ADDRESS_RE.match(name):
                continue
            resource = resource_name(thread, value_at(func_table, "resource", i))
            if resource == lib_name or os.path.basename(resource) == lib_name:
                addresses.add(name)
    return sorted(addresses, key=lambda value: int(value, 16))


def atos_symbol_map(binary: Path, addresses: list[str], load_address: int) -> dict[str, str]:
    if not addresses:
        return {}
    absolute_addresses = [hex(load_address + int(address, 16)) for address in addresses]
    try:
        result = subprocess.run(
            ["atos", "-o", str(binary), "-l", hex(load_address), *absolute_addresses],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {}
    if result.returncode != 0:
        return {}
    symbols = result.stdout.splitlines()
    return {raw: symbol.strip() for raw, symbol in zip(addresses, symbols) if symbol.strip() and symbol.strip() != raw}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=20, help="Number of functions to show")
    parser.add_argument("--threads", type=int, default=8, help="Number of hottest threads to expand")
    parser.add_argument("--stacks", type=int, default=8, help="Number of stacks to show per thread")
    parser.add_argument("--stack-depth", type=int, default=14, help="Frames to print from the leaf side")
    parser.add_argument("--libs", action="store_true", help="Print profile library metadata")
    parser.add_argument("--binary", type=Path, help="Print dwarfdump UUID for a candidate binary")
    parser.add_argument("--symbolicate", action="store_true", help="Use atos to symbolicate raw app offsets")
    parser.add_argument(
        "--symbol-lib",
        help="Library name to symbolicate; defaults to the basename of --binary",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("samples", "cpu"),
        default="samples",
        help="Use sample weights or samples.threadCPUDelta CPU microseconds for attribution",
    )
    parser.add_argument(
        "--load-address",
        default="0x100000000",
        help="Mach-O load address for atos when symbolizing Samply offsets",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    meta = profile.get("meta") or {}
    print(f"Profile: {args.profile}")
    print(f"Product: {meta.get('product', '')}  interval_ms={meta.get('interval', '')} weight_mode={args.weight_mode}")

    if args.binary:
        print(f"Candidate binary: {args.binary}")
        print(binary_uuid(args.binary))

    symbols: dict[str, str] = {}
    if args.symbolicate:
        if not args.binary:
            parser.error("--symbolicate requires --binary")
        symbol_lib = args.symbol_lib or args.binary.name
        addresses = collect_symbol_addresses(profile, symbol_lib)
        symbols = atos_symbol_map(args.binary, addresses, int(str(args.load_address), 0))
        print(f"Symbolicated {len(symbols)} / {len(addresses)} raw addresses from {symbol_lib}")

    if args.libs:
        print_libs(profile)

    summaries = [summarize_thread(thread, symbols, args.weight_mode) for thread in profile.get("threads") or []]
    summaries.sort(key=lambda s: (s["cpu_ms"], s["weight"]), reverse=True)

    print("\nThreads by CPU:")
    for summary in summaries[: max(args.threads, 20)]:
        print(
            f"  cpu_ms={summary['cpu_ms']:10.3f} weight={summary['weight']:8d} "
            f"samples={summary['samples']:7} tid={summary['tid']} name={summary['name']}"
        )

    for summary in summaries[: args.threads]:
        if summary["weight"] <= 0:
            continue
        print("\n" + "=" * 100)
        print(
            f"Thread {summary['name']} tid={summary['tid']} cpu_ms={summary['cpu_ms']:.3f} weight={summary['weight']}"
        )
        print_counter("Top self frames:", summary["self"], summary["weight"], args.top)
        print_counter("Top inclusive frames:", summary["inclusive"], summary["weight"], args.top)
        print_stacks(summary["stacks"], summary["weight"], args.stacks, args.stack_depth)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
