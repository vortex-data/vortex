#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render per-crate lines-of-code as a collapsible markdown comment.

Takes the JSON produced by ``crate-loc.py`` for HEAD and (optionally) a base
revision, and prints a single ``<details>`` block: the ``<summary>`` is a
one-line total so the comment stays compact, and expanding it reveals the full
per-crate table with deltas against the base.
"""

import argparse
import json


def fmt_delta(delta: int) -> str:
    """Format a signed line-count delta, or an em dash when unchanged."""
    if delta == 0:
        return "—"
    return f"{'+' if delta > 0 else '−'}{abs(delta):,}"


def fmt_pct(base: int, head: int) -> str:
    """Format a percentage change, handling newly added crates."""
    if base == 0:
        return "new" if head > 0 else "—"
    if head == base:
        return "—"
    pct = (head / base - 1) * 100
    return f"{'+' if pct > 0 else '−'}{abs(pct):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-crate LOC as a markdown comment")
    parser.add_argument("head_file", help="LOC JSON for HEAD")
    parser.add_argument("--base-file", help="LOC JSON for the base revision", default=None)
    args = parser.parse_args()

    with open(args.head_file) as f:
        head = json.load(f)

    base = {}
    if args.base_file:
        try:
            with open(args.base_file) as f:
                base = json.load(f)
        except FileNotFoundError:
            base = {}
    have_base = bool(base)

    crates = sorted(set(head) | set(base))
    rows = []
    for crate in crates:
        h = head.get(crate, 0)
        b = base.get(crate, 0)
        rows.append((crate, b, h, h - b))

    total_head = sum(h for _, _, h, _ in rows)
    total_base = sum(b for _, b, _, _ in rows)
    total_delta = total_head - total_base
    n_crates = sum(1 for _, _, h, _ in rows if h > 0)

    # Largest movers first, then largest crates.
    rows.sort(key=lambda r: (abs(r[3]), r[2]), reverse=True)

    if have_base and total_delta != 0:
        summary = (
            f"Code size: {total_head:,} lines of Rust across {n_crates} crates "
            f"({fmt_delta(total_delta)}, {fmt_pct(total_base, total_head)})"
        )
    else:
        summary = f"Code size: {total_head:,} lines of Rust across {n_crates} crates"

    print("<details>")
    print(f"<summary>{summary}</summary>")
    print("")
    print("<br>")
    print("")

    if have_base:
        print("| Crate | Lines | Δ | % |")
        print("|-------|------:|--:|--:|")
        for crate, b, h, d in rows:
            print(f"| `{crate}` | {h:,} | {fmt_delta(d)} | {fmt_pct(b, h)} |")
        print("")
        print(f"**Total:** {total_base:,} → {total_head:,} ({fmt_delta(total_delta)})")
    else:
        print("| Crate | Lines |")
        print("|-------|------:|")
        for crate, _, h, _ in rows:
            print(f"| `{crate}` | {h:,} |")
        print("")
        print(f"**Total:** {total_head:,} lines")

    print("")
    print("</details>")


if __name__ == "__main__":
    main()
