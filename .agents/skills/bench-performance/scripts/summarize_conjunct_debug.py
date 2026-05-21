#!/usr/bin/env python3
"""Summarize Vortex conjunct evaluation tracing lines."""

from __future__ import annotations

import argparse
import collections
import re
import statistics
from pathlib import Path

FIELD_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>\"(?:[^\"\\]|\\.)*\"|Some\([^)]+\)|None|[^\s]+)"
)
FIRST_FIELD_RE = re.compile(r" [A-Za-z_][A-Za-z0-9_]*=")


def parse_value(raw: str) -> str:
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("Some(") and raw.endswith(")"):
        return raw[5:-1]
    return raw


def as_int(fields: dict[str, str], key: str) -> int:
    value = fields.get(key)
    if value in (None, "None"):
        return 0
    return int(value)


def as_float(fields: dict[str, str], key: str) -> float:
    value = fields.get(key)
    if value in (None, "None"):
        return 0.0
    return float(value)


def message_for(line: str) -> str | None:
    rest_match = re.search(r":\d+: (?P<rest>.*)$", line.rstrip())
    rest = rest_match.group("rest") if rest_match else line.rstrip()
    first_field = FIRST_FIELD_RE.search(rest)
    message = rest[: first_field.start() if first_field else len(rest)].strip()
    if "conjunct" in message and "evaluated" in message:
        return message
    return None


def quantile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--top-orders", type=int, default=12)
    parser.add_argument("--message-regex", help="Only include messages matching this regex")
    args = parser.parse_args()

    message_re = re.compile(args.message_regex) if args.message_regex else None
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = collections.defaultdict(list)
    order_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    order_input_rows: collections.defaultdict[tuple[str, ...], int] = collections.defaultdict(int)
    current_windows: dict[tuple[str, str, str, str], list[str]] = {}
    rows = 0

    for path in args.logs:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                message = message_for(line)
                if message is None or (message_re is not None and not message_re.search(message)):
                    continue
                fields = {match.group("key"): parse_value(match.group("value")) for match in FIELD_RE.finditer(line)}
                original_idx = (
                    fields.get("original_idx") or fields.get("conjunct_idx") or fields.get("child_idx") or "?"
                )
                conjunct = fields.get("conjunct", "")
                groups[(message, original_idx, conjunct)].append(fields)
                rows += 1

                window_key = (
                    fields.get("scan_label", ""),
                    fields.get("coord_start", ""),
                    fields.get("coord_end", ""),
                    fields.get("output_coord_hash", ""),
                )
                order = current_windows.setdefault(window_key, [])
                if not order:
                    order_input_rows[tuple(order)] += 0
                order.append(original_idx)
                if as_int(fields, "output_rows") == 0:
                    order_tuple = tuple(order)
                    order_counts[order_tuple] += 1
                    order_input_rows[order_tuple] += as_int(fields, "input_rows")
                    current_windows.pop(window_key, None)

    for order in current_windows.values():
        if order:
            order_tuple = tuple(order)
            order_counts[order_tuple] += 1

    if rows == 0:
        print("No conjunct debug rows found.")
        return 1

    print(f"rows={rows:,}")
    print(
        "message\tconjunct\tindex\tevents\tinput_rows\toutput_rows\tcompute_input_rows\tcompute_output_rows\telapsed_ms\tcompute_per_input"
    )
    for (message, idx, conjunct), entries in sorted(groups.items(), key=lambda item: item[0]):
        input_rows = sum(as_int(e, "input_rows") for e in entries)
        output_rows = sum(as_int(e, "output_rows") for e in entries)
        compute_input = sum(as_int(e, "compute_input_rows") or as_int(e, "input_rows") for e in entries)
        compute_output = sum(as_int(e, "compute_output_rows") or as_int(e, "output_rows") for e in entries)
        elapsed = sum(as_float(e, "elapsed_ms") for e in entries)
        ratio = compute_input / input_rows if input_rows else 0.0
        print(
            f"{message}\t{conjunct}\t{idx}\t{len(entries):,}\t{input_rows:,}\t{output_rows:,}\t"
            f"{compute_input:,}\t{compute_output:,}\t{elapsed:,.3f}\t{ratio:.3f}"
        )

    print("\norders:")
    for order, count in order_counts.most_common(args.top_orders):
        print(f"  {order}: count={count:,} input_rows={order_input_rows[order]:,}")

    print("\nevents_per_window:")
    lengths = [len(order) for order, count in order_counts.items() for _ in range(count)]
    if lengths:
        print(f"  median={statistics.median(lengths):.0f} p90={quantile(lengths, 0.90):.0f} max={max(lengths)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
