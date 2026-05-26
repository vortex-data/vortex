#!/usr/bin/env python3
"""Summarize Vortex mask-style tracing lines from benchmark logs."""

from __future__ import annotations

import argparse
import collections
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

FIELD_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>\"(?:[^\"\\]|\\.)*\"|Some\([^)]+\)|None|[^\s]+)"
)
MESSAGE_RE = re.compile(
    r"\b(?P<message>v[0-9]+\s+(?:"
    r"filter batch projected|"
    r"filter conjunct evaluated|"
    r"filtered flat batch projected|"
    r"filtered flat batch skipped|"
    r"scan batch projected|"
    r"scan batch skipped|"
    r"conjunct mask evaluated|"
    r"pruning conjunct evaluated"
    r"))\b"
)


@dataclass
class Row:
    path: Path
    line_no: int
    message: str
    fields: dict[str, str]


def parse_value(raw: str) -> str:
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("Some(") and raw.endswith(")"):
        return raw[5:-1]
    return raw


def parse_int(fields: dict[str, str], key: str) -> int | None:
    value = fields.get(key)
    if value is None or value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(fields: dict[str, str], key: str) -> float | None:
    value = fields.get(key)
    if value is None or value == "None":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def iter_rows(paths: list[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                message_match = MESSAGE_RE.search(line)
                if not message_match:
                    continue
                fields = {match.group("key"): parse_value(match.group("value")) for match in FIELD_RE.finditer(line)}
                if not has_row_fields(fields):
                    continue
                rows.append(
                    Row(
                        path=path,
                        line_no=line_no,
                        message=message_match.group("message").strip(),
                        fields=fields,
                    )
                )
    return rows


def has_row_fields(fields: dict[str, str]) -> bool:
    return ("batch_input_rows" in fields and "batch_output_rows" in fields) or (
        "input_rows" in fields and "output_rows" in fields
    )


def row_count(fields: dict[str, str], primary: str, fallback: str) -> int:
    return parse_int(fields, primary) or parse_int(fields, fallback) or 0


def quantile(values: list[int | float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def fmt_num(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if math.isfinite(value):
        return f"{value:,.3f}"
    return str(value)


def summarize_group(message: str, rows: list[Row], top: int) -> None:
    input_rows = [row_count(row.fields, "batch_input_rows", "input_rows") for row in rows]
    output_rows = [row_count(row.fields, "batch_output_rows", "output_rows") for row in rows]
    extra_rows = [parse_int(row.fields, "batch_extra_rows") for row in rows]
    elapsed_ms = [value for row in rows if (value := parse_float(row.fields, "elapsed_ms")) is not None]

    total_input = sum(input_rows)
    total_output = sum(output_rows)
    zero = sum(1 for value in output_rows if value == 0)
    density = (total_output / total_input) if total_input else 0.0
    print(f"\n{message}")
    print(
        f"  batches={len(rows):,} zero_output={zero:,} ({(zero * 100.0 / len(rows)) if rows else 0.0:.1f}%) "
        f"input_rows={total_input:,} output_rows={total_output:,} density={density:.6%}"
    )
    if input_rows:
        print(
            "  input_rows_per_batch "
            f"min={min(input_rows):,} median={statistics.median(input_rows):,.0f} "
            f"p90={quantile(input_rows, 0.90):,.0f} p99={quantile(input_rows, 0.99):,.0f} "
            f"max={max(input_rows):,}"
        )
    if elapsed_ms:
        print(
            "  elapsed_ms "
            f"sum={sum(elapsed_ms):,.3f} median={statistics.median(elapsed_ms):,.3f} "
            f"p90={quantile(elapsed_ms, 0.90):,.3f} p99={quantile(elapsed_ms, 0.99):,.3f} "
            f"max={max(elapsed_ms):,.3f}"
        )
    if any(value is not None for value in extra_rows):
        extra_sum = sum(value or 0 for value in extra_rows)
        print(f"  extra_rows_sum={extra_sum:,}")

    print("  largest input batches:")
    for row, input_count, output_count in sorted(
        zip(rows, input_rows, output_rows),
        key=lambda item: item[1],
        reverse=True,
    )[:top]:
        label = row.fields.get("scan_label", "")
        row_start = row.fields.get("row_start", "?")
        row_end = row.fields.get("row_end", "?")
        print(
            f"    {input_count:>10,} -> {output_count:<8,} "
            f"{Path(label).name if label else '-'} rows={row_start}..{row_end} "
            f"{row.path}:{row.line_no}"
        )


def duplicate_keys(rows: list[Row]) -> list[tuple[tuple[str, str, str, str, str], int, int]]:
    counts: collections.Counter[tuple[str, str, str, str, str]] = collections.Counter()
    input_totals: collections.defaultdict[tuple[str, str, str, str, str], int] = collections.defaultdict(int)
    for row in rows:
        key = (
            row.message,
            row.fields.get("scan_label", ""),
            row.fields.get("coord_start") or row.fields.get("row_start", ""),
            row.fields.get("coord_end") or row.fields.get("row_end", ""),
            row.fields.get("coord_hash") or row.fields.get("output_coord_hash", ""),
        )
        counts[key] += 1
        input_totals[key] += row_count(row.fields, "batch_input_rows", "input_rows")
    return [(key, count, input_totals[key]) for key, count in counts.items() if count > 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--message-regex", help="Only include messages matching this regex")
    parser.add_argument("--top", type=int, default=8, help="Largest batches to print per message")
    parser.add_argument("--duplicates", action="store_true", help="Report duplicate coordinate masks")
    args = parser.parse_args()

    rows = iter_rows(args.logs)
    if args.message_regex:
        message_re = re.compile(args.message_regex)
        rows = [row for row in rows if message_re.search(row.message)]

    if not rows:
        print("No mask-style rows found.")
        return 1

    groups: collections.defaultdict[str, list[Row]] = collections.defaultdict(list)
    for row in rows:
        groups[row.message].append(row)

    print(f"rows={len(rows):,} files={len(args.logs)}")
    for message in sorted(groups):
        summarize_group(message, groups[message], args.top)

    if args.duplicates:
        duplicates = duplicate_keys(rows)
        if duplicates:
            print("\nduplicate coordinate masks:")
            for key, count, input_total in sorted(duplicates, key=lambda item: item[1], reverse=True)[: args.top]:
                message, label, start, end, coord_hash = key
                print(
                    f"  count={count:,} input_rows={input_total:,} "
                    f"message={message!r} scan={Path(label).name if label else '-'} "
                    f"coords={start}..{end} hash={coord_hash}"
                )
        else:
            print("\nduplicate coordinate masks: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
