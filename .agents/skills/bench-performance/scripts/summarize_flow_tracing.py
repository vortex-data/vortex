#!/usr/bin/env python3
"""Summarize structured flow tracing logs from benchmark runs."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REST_RE = re.compile(r":\d+: (?P<rest>.*)$")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")
FIRST_KV_RE = re.compile(r" [A-Za-z_][A-Za-z0-9_]*=")


def parse_value(raw: str) -> int | float | str:
    raw = raw.strip().strip(",").strip('"')
    if raw in {"true", "false"}:
        return raw
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def fmt_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} min={min(values):.3f} "
        f"p50={statistics.median(values):.3f} p90={percentile(values, 0.90):.3f} "
        f"p99={percentile(values, 0.99):.3f} max={max(values):.3f} "
        f"sum={sum(values):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--top-ranges", type=int, default=10)
    args = parser.parse_args()

    messages: Counter[str] = Counter()
    elapsed: dict[str, list[float]] = defaultdict(list)
    rows: dict[str, list[float]] = defaultdict(list)
    true_counts: dict[str, list[float]] = defaultdict(list)
    send_waits: dict[str, list[float]] = defaultdict(list)
    aligned_send_waits: dict[tuple[int, int], list[float]] = defaultdict(list)
    aligned_rows: dict[tuple[int, int], list[float]] = defaultdict(list)
    aligned_pending: Counter[tuple[int, int]] = Counter()
    aligned_meta: dict[int, tuple[int, int, str]] = {}
    scan_ranges: Counter[tuple[int, int]] = Counter()
    publish_ranges: Counter[tuple[int, int]] = Counter()
    chunk_child_counts: Counter[int] = Counter()
    filtered_segments: Counter[str] = Counter()

    with args.log.open() as f:
        for line in f:
            match = REST_RE.search(line.rstrip())
            if not match:
                continue
            rest = match.group("rest")
            first_kv = FIRST_KV_RE.search(rest)
            message = rest[: first_kv.start() if first_kv else len(rest)].strip()
            fields = {k: parse_value(v) for k, v in KV_RE.findall(line)}
            messages[message] += 1

            if "elapsed_ms" in fields:
                elapsed[message].append(float(fields["elapsed_ms"]))
            if "send_elapsed_ms" in fields:
                send_waits[message].append(float(fields["send_elapsed_ms"]))
                if message == "aligned producer sent":
                    key = (int(fields["aligned_id"]), int(fields["child_idx"]))
                    aligned_send_waits[key].append(float(fields["send_elapsed_ms"]))
                    aligned_rows[key].append(float(fields.get("rows", 0)))
            if "rows" in fields:
                rows[message].append(float(fields["rows"]))
            if "true_count" in fields:
                true_counts[message].append(float(fields["true_count"]))
            if message == "scan execute":
                scan_ranges[(int(fields["row_start"]), int(fields["row_end"]))] += 1
            if message in {"publish materialised mask", "publish streaming mask"}:
                publish_ranges[(int(fields["source_start"]), int(fields["source_end"]))] += 1
            if message == "aligned new":
                aligned_meta[int(fields["aligned_id"])] = (
                    int(fields["child_count"]),
                    int(fields["buffer_depth"]),
                    str(fields.get("aligned_label", "?")),
                )
            if message == "aligned pending child":
                aligned_pending[(int(fields["aligned_id"]), int(fields["child_idx"]))] += 1
            if message == "chunked execute":
                chunk_child_counts[int(fields["chunk_count"])] += 1
            if message.startswith("filtered flat"):
                filtered_segments[str(fields.get("segment_id", "?"))] += 1

    print(f"Log: {args.log}")
    print("\nMessage counts:")
    for message, count in messages.most_common():
        print(f"  {count:8d}  {message}")

    print("\nElapsed ms:")
    for message, values in sorted(elapsed.items()):
        print(f"  {message}: {fmt_stats(values)}")

    print("\nSend wait ms:")
    for message, values in sorted(send_waits.items()):
        print(f"  {message}: {fmt_stats(values)}")

    if aligned_send_waits:
        by_label: dict[str, list[float]] = defaultdict(list)
        by_label_child: dict[tuple[str, int], list[float]] = defaultdict(list)
        by_label_rows: dict[tuple[str, int], list[float]] = defaultdict(list)
        for (aligned_id, child_idx), values in aligned_send_waits.items():
            label = aligned_meta.get(aligned_id, (0, 0, str(aligned_id)))[2]
            by_label[label].extend(values)
            by_label_child[(label, child_idx)].extend(values)
            by_label_rows[(label, child_idx)].extend(aligned_rows[(aligned_id, child_idx)])

        print("\nAligned producer waits by label:")
        for label, values in sorted(by_label.items(), key=lambda item: sum(item[1]), reverse=True):
            print(f"  {label}: {fmt_stats(values)}")

        print("\nAligned producer waits by label/child:")
        for (label, child_idx), values in sorted(by_label_child.items(), key=lambda item: sum(item[1]), reverse=True):
            row_values = by_label_rows[(label, child_idx)]
            print(f"  {label} child={child_idx}: wait=({fmt_stats(values)}) rows=({fmt_stats(row_values)})")

        print("\nAligned producer waits by stream/child:")
        ranked = sorted(
            aligned_send_waits.items(),
            key=lambda item: sum(item[1]),
            reverse=True,
        )
        for (aligned_id, child_idx), values in ranked[: args.top_ranges]:
            meta = aligned_meta.get(aligned_id, (0, 0, "?"))
            row_values = aligned_rows[(aligned_id, child_idx)]
            print(
                f"  aligned={aligned_id} child={child_idx} "
                f"label={meta[2]} arity={meta[0]} depth={meta[1]} "
                f"pending={aligned_pending[(aligned_id, child_idx)]} "
                f"wait=({fmt_stats(values)}) rows=({fmt_stats(row_values)})"
            )

    print("\nRows:")
    for message, values in sorted(rows.items()):
        print(f"  {message}: {fmt_stats(values)}")

    print("\nTrue counts:")
    for message, values in sorted(true_counts.items()):
        print(f"  {message}: {fmt_stats(values)}")

    print("\nScan ranges:")
    print(f"  unique={len(scan_ranges)} total={sum(scan_ranges.values())}")
    for (start, end), count in scan_ranges.most_common(args.top_ranges):
        print(f"  {count:5d}  {start}..{end} rows={end - start}")

    print("\nPublished mask source ranges:")
    print(f"  unique={len(publish_ranges)} total={sum(publish_ranges.values())}")
    for (start, end), count in publish_ranges.most_common(args.top_ranges):
        print(f"  {count:5d}  {start}..{end} rows={end - start}")

    print("\nChunked execute child counts:")
    for count, hits in sorted(chunk_child_counts.items()):
        print(f"  chunk_count={count}: {hits}")

    print("\nFiltered flat segment ids:")
    print(f"  unique={len(filtered_segments)} total_events={sum(filtered_segments.values())}")
    for segment, count in filtered_segments.most_common(args.top_ranges):
        print(f"  {count:5d}  {segment}")


if __name__ == "__main__":
    main()
