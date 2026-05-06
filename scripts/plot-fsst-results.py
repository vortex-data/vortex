#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Convert datafusion-bench gh-json output to CSV, markdown, and a bar chart.

Reads JSONL produced by `datafusion-bench --display-format gh-json -o <file>`,
where each line is a `QueryMeasurementJson` (see vortex-bench/src/measurements.rs).

Outputs into --output-dir:
  results.csv   one row per (query, format)
  summary.md    markdown table for pasting into PRs / docs
  plot.png      bar chart (skipped if matplotlib is missing)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def parse_query_id(name: str) -> str:
    # name format: "tpch_q01/datafusion:vortex-file-compressed"
    if "_q" not in name:
        return name
    after = name.split("_q", 1)[1]
    return after.split("/", 1)[0]


def load(input_path: Path) -> list[dict]:
    rows = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            runtimes_ns = rec.get("all_runtimes") or []
            target = rec.get("target") or {}
            rows.append(
                {
                    "query": parse_query_id(rec.get("name", "")),
                    "engine": target.get("engine", ""),
                    "format": target.get("format", ""),
                    "median_ms": rec["value"] / 1e6,
                    "min_ms": (min(runtimes_ns) / 1e6) if runtimes_ns else None,
                    "max_ms": (max(runtimes_ns) / 1e6) if runtimes_ns else None,
                    "mean_ms": (statistics.mean(runtimes_ns) / 1e6) if runtimes_ns else None,
                    "stdev_ms": (statistics.pstdev(runtimes_ns) / 1e6) if len(runtimes_ns) > 1 else 0.0,
                    "n_runs": len(runtimes_ns),
                    "commit": rec.get("commit_id", ""),
                    "name": rec.get("name", ""),
                }
            )
    rows.sort(key=lambda r: (r["query"], r["format"]))
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        out.write_text("")
        return
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict], out: Path) -> None:
    lines = [
        "| query | format | median (ms) | min (ms) | max (ms) | stdev (ms) | runs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['query']} | {r['format']} | "
            f"{r['median_ms']:.2f} | "
            f"{(r['min_ms'] or 0):.2f} | "
            f"{(r['max_ms'] or 0):.2f} | "
            f"{r['stdev_ms']:.2f} | "
            f"{r['n_runs']} |"
        )
    out.write_text("\n".join(lines) + "\n")


def write_plot(rows: list[dict], out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed; skipping plot.png "
            "(install with `pip install matplotlib`)",
            file=sys.stderr,
        )
        return False

    labels = [r["query"] for r in rows]
    values = [r["median_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.4), 5))
    ax.bar(labels, values)
    ax.set_xlabel("query")
    ax.set_ylabel("median runtime (ms)")
    ax.set_title("TPC-H — Vortex / DataFusion")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="raw.jsonl path")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load(args.input)
    if not rows:
        print(f"no measurements found in {args.input}", file=sys.stderr)
        return 1

    csv_path = args.output_dir / "results.csv"
    md_path = args.output_dir / "summary.md"
    plot_path = args.output_dir / "plot.png"

    write_csv(rows, csv_path)
    print(f"wrote {csv_path}")

    write_markdown(rows, md_path)
    print(f"wrote {md_path}")
    print()
    print(md_path.read_text())

    if write_plot(rows, plot_path):
        print(f"wrote {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
