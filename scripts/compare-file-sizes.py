#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare file sizes between base and HEAD and generate markdown report."""

import argparse
import json
import sys
from collections import defaultdict


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


def format_change(change_bytes: int) -> str:
    """Format byte change with sign."""
    sign = "+" if change_bytes > 0 else ""
    return f"{sign}{format_size(abs(change_bytes))}"


def format_pct_change(pct: float) -> str:
    """Format percentage change with sign."""
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Compare file sizes between base and HEAD")
    parser.add_argument("base_file", help="Base JSONL file")
    parser.add_argument("head_file", help="HEAD JSONL file")
    args = parser.parse_args()

    # Load base and head data
    base_data = {}
    try:
        with open(args.base_file) as f:
            for line in f:
                record = json.loads(line)
                # Support old records without scale_factor (default to "1.0")
                scale_factor = record.get("scale_factor", "1.0")
                key = (record["benchmark"], scale_factor, record["format"], record["file"])
                base_data[key] = record["size_bytes"]
    except FileNotFoundError:
        print("_Base file sizes not found._")
        sys.exit(0)

    head_data = {}
    try:
        with open(args.head_file) as f:
            for line in f:
                record = json.loads(line)
                scale_factor = record.get("scale_factor", "1.0")
                key = (record["benchmark"], scale_factor, record["format"], record["file"])
                head_data[key] = record["size_bytes"]
    except FileNotFoundError:
        print("_HEAD file sizes not found._")
        sys.exit(0)

    # Compare sizes
    comparisons = []
    format_totals = defaultdict(lambda: {"base": 0, "head": 0})

    all_keys = set(base_data.keys()) | set(head_data.keys())
    for key in all_keys:
        benchmark, scale_factor, fmt, file_name = key
        base_size = base_data.get(key, 0)
        head_size = head_data.get(key, 0)

        format_totals[fmt]["base"] += base_size
        format_totals[fmt]["head"] += head_size

        change = head_size - base_size
        if change == 0:
            continue

        if base_size > 0:
            pct_change = (head_size / base_size - 1) * 100
        elif head_size > 0:
            pct_change = float("inf")
        else:
            pct_change = 0

        comparisons.append(
            {
                "file": file_name,
                "scale_factor": scale_factor,
                "format": fmt,
                "base_size": base_size,
                "head_size": head_size,
                "change": change,
                "pct_change": pct_change,
            }
        )

    if not comparisons:
        print("_No file size changes detected._")
        return

    # Sort by pct_change descending (largest increases first)
    comparisons.sort(key=lambda x: x["pct_change"], reverse=True)

    # Output markdown table
    print("| File | Scale | Format | Base | HEAD | Change | % |")
    print("|------|-------|--------|------|------|--------|---|")

    for comp in comparisons:
        pct_str = format_pct_change(comp["pct_change"]) if comp["pct_change"] != float("inf") else "new"
        base_str = format_size(comp["base_size"]) if comp["base_size"] > 0 else "-"
        print(
            f"| {comp['file']} | {comp['scale_factor']} | {comp['format']} | {base_str} | "
            f"{format_size(comp['head_size'])} | {format_change(comp['change'])} | {pct_str} |"
        )

    # Output totals
    print("")
    print("**Totals:**")
    for fmt in sorted(format_totals.keys()):
        totals = format_totals[fmt]
        base_total = totals["base"]
        head_total = totals["head"]
        if base_total > 0:
            total_pct = (head_total / base_total - 1) * 100
            pct_str = f" ({format_pct_change(total_pct)})"
        else:
            pct_str = ""
        print(f"- {fmt}: {format_size(base_total)} \u2192 {format_size(head_total)}{pct_str}")


if __name__ == "__main__":
    main()
