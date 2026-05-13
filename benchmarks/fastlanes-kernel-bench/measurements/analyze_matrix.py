#!/usr/bin/env python3
"""Summarize /tmp/full_matrix.csv into the tables embedded in README.md.

Reads the 720-row CSV produced by run_matrix.py and emits, for each unsigned
type, a Markdown table with:

  * one row per bit width W;
  * six numeric columns: sse2_bare, sse2_fused, ymm_bare, ymm_fused,
    zmm_bare, zmm_fused (best-of-3 medians in ns);
  * a 'best' column naming the (simd, variant) that has the lowest fused-FoR
    runtime, since FoR is what production code uses.

After the per-type tables it prints aggregate summaries:

  * Per-type fused-FoR speedup of (best of {sse2,ymm,zmm}) vs sse2 at each W.
  * Per-ISA, per-type geomean ratio of fused / bare (fusing overhead).
  * Per-ISA, per-type geomean ratio of zmm/ymm and ymm/sse2 (ISA speedup).

The script writes to stdout. Pipe it into the README or copy interesting
tables in by hand.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path("/tmp/full_matrix.csv")
TYPES = ("u8", "u16", "u32", "u64")
MAXW = {"u8": 8, "u16": 16, "u32": 32, "u64": 64}
SIMD_ORDER = ("sse2", "ymm", "zmm")
VARIANTS = ("bare_unpack", "fused_for")


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def fmt_ns(x: float) -> str:
    if not math.isfinite(x):
        return "  -  "
    if x < 100:
        return f"{x:5.2f}"
    return f"{x:5.1f}"


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        return 1

    # data[(T, W, simd, variant)] = ns
    data: dict[tuple[str, int, str, str], float] = {}
    with CSV_PATH.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ns = float(row["best_median_ns"])
            except ValueError:
                ns = float("nan")
            data[(row["T"], int(row["W"]), row["simd"], row["variant"])] = ns

    print("# Full (T, W, SIMD, variant) matrix")
    print()
    print(f"Source: `{CSV_PATH}` ({len(data)} cells).")
    print()

    for t in TYPES:
        print(f"## `{t}`")
        print()
        print("| W | sse2 bare | sse2 fused | ymm bare | ymm fused | zmm bare | zmm fused | best fused (simd) |")
        print("|--:|---------:|----------:|--------:|---------:|--------:|---------:|:------------------|")
        for w in range(1, MAXW[t] + 1):
            cells = {}
            for simd in SIMD_ORDER:
                for v in VARIANTS:
                    cells[(simd, v)] = data.get((t, w, simd, v), float("nan"))
            # Best fused over SIMDs.
            fused = [(simd, cells[(simd, "fused_for")]) for simd in SIMD_ORDER]
            fused_valid = [(s, v) for s, v in fused if math.isfinite(v)]
            if fused_valid:
                best_simd, best_ns = min(fused_valid, key=lambda x: x[1])
                best_str = f"{best_simd} ({best_ns:.1f} ns)"
            else:
                best_str = " - "
            print(
                f"| {w} | {fmt_ns(cells[('sse2','bare_unpack')])} "
                f"| {fmt_ns(cells[('sse2','fused_for')])} "
                f"| {fmt_ns(cells[('ymm','bare_unpack')])} "
                f"| {fmt_ns(cells[('ymm','fused_for')])} "
                f"| {fmt_ns(cells[('zmm','bare_unpack')])} "
                f"| {fmt_ns(cells[('zmm','fused_for')])} "
                f"| {best_str} |"
            )
        print()

    # --- Aggregate: per-(T, ISA) geomean of fused/bare. ---
    print("## Fused-vs-bare overhead by ISA and type (geomean across all W)")
    print()
    print("| type | sse2 fused/bare | ymm fused/bare | zmm fused/bare |")
    print("|------|----------------:|---------------:|---------------:|")
    for t in TYPES:
        row = [f"`{t}`"]
        for simd in SIMD_ORDER:
            ratios = []
            for w in range(1, MAXW[t] + 1):
                b = data.get((t, w, simd, "bare_unpack"), float("nan"))
                f = data.get((t, w, simd, "fused_for"), float("nan"))
                if math.isfinite(b) and math.isfinite(f) and b > 0:
                    ratios.append(f / b)
            g = geomean(ratios)
            pct = (g - 1.0) * 100 if math.isfinite(g) else float("nan")
            row.append(f"{g:.3f} ({pct:+.1f}%)")
        print("| " + " | ".join(row) + " |")
    print()

    # --- Aggregate: per-(T, comparison) geomean of ISA speedups. ---
    print("## ISA speedup by type (geomean across all W, bare_unpack only)")
    print()
    print("| type | ymm/sse2 | zmm/ymm | zmm/sse2 |")
    print("|------|---------:|--------:|---------:|")
    for t in TYPES:
        ymm_sse2, zmm_ymm, zmm_sse2 = [], [], []
        for w in range(1, MAXW[t] + 1):
            s = data.get((t, w, "sse2", "bare_unpack"), float("nan"))
            y = data.get((t, w, "ymm", "bare_unpack"), float("nan"))
            z = data.get((t, w, "zmm", "bare_unpack"), float("nan"))
            if math.isfinite(s) and math.isfinite(y) and s > 0:
                ymm_sse2.append(y / s)
            if math.isfinite(y) and math.isfinite(z) and y > 0:
                zmm_ymm.append(z / y)
            if math.isfinite(s) and math.isfinite(z) and s > 0:
                zmm_sse2.append(z / s)
        print(
            f"| `{t}` | {geomean(ymm_sse2):.3f} "
            f"| {geomean(zmm_ymm):.3f} "
            f"| {geomean(zmm_sse2):.3f} |"
        )
    print()

    # --- Aggregate: per-T count of which ISA wins (bare_unpack). ---
    print("## How often does each ISA win bare_unpack at a given W?")
    print()
    print("| type | sse2 wins | ymm wins | zmm wins |")
    print("|------|----------:|---------:|---------:|")
    for t in TYPES:
        wins = {"sse2": 0, "ymm": 0, "zmm": 0}
        for w in range(1, MAXW[t] + 1):
            cells = []
            for simd in SIMD_ORDER:
                v = data.get((t, w, simd, "bare_unpack"), float("nan"))
                if math.isfinite(v):
                    cells.append((simd, v))
            if cells:
                winner = min(cells, key=lambda x: x[1])[0]
                wins[winner] += 1
        print(f"| `{t}` | {wins['sse2']} | {wins['ymm']} | {wins['zmm']} |")
    print()

    # --- Same for fused_for, which is what production runs. ---
    print("## How often does each ISA win fused_for at a given W?")
    print()
    print("| type | sse2 wins | ymm wins | zmm wins |")
    print("|------|----------:|---------:|---------:|")
    for t in TYPES:
        wins = {"sse2": 0, "ymm": 0, "zmm": 0}
        for w in range(1, MAXW[t] + 1):
            cells = []
            for simd in SIMD_ORDER:
                v = data.get((t, w, simd, "fused_for"), float("nan"))
                if math.isfinite(v):
                    cells.append((simd, v))
            if cells:
                winner = min(cells, key=lambda x: x[1])[0]
                wins[winner] += 1
        print(f"| `{t}` | {wins['sse2']} | {wins['ymm']} | {wins['zmm']} |")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
