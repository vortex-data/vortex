#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Convert datafusion-bench gh-json output to CSV, markdown, and bar charts.

Reads JSONL produced by `datafusion-bench --display-format gh-json -o <file>`,
where each line is a `QueryMeasurementJson` (see vortex-bench/src/measurements.rs).

Two modes:
  --input <file>      single JSONL -> CSV/MD/PNG in --output-dir
  --input-dir <dir>   walks <dir>/<bench>/raw.jsonl, writes per-benchmark
                      outputs alongside each raw.jsonl, plus aggregated
                      results.csv / summary.md / plot.png at <dir>'s top level
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def parse_query_id(name: str) -> str:
    # "tpch_q01/datafusion:vortex-file-compressed" -> "01"
    if "_q" not in name:
        return name
    return name.split("_q", 1)[1].split("/", 1)[0]


def parse_dataset(name: str, fallback: str) -> str:
    # "tpch_q01/..." -> "tpch"
    if "_q" not in name:
        return fallback
    return name.split("_q", 1)[0]


def load(input_path: Path, dataset_fallback: str = "") -> list[dict]:
    rows: list[dict] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            runtimes_ns = rec.get("all_runtimes") or []
            target = rec.get("target") or {}
            name = rec.get("name", "")
            rows.append(
                {
                    "dataset": parse_dataset(name, dataset_fallback),
                    "query": parse_query_id(name),
                    "engine": target.get("engine", ""),
                    "format": target.get("format", ""),
                    "median_ms": rec["value"] / 1e6,
                    "min_ms": (min(runtimes_ns) / 1e6) if runtimes_ns else None,
                    "max_ms": (max(runtimes_ns) / 1e6) if runtimes_ns else None,
                    "mean_ms": (
                        statistics.mean(runtimes_ns) / 1e6 if runtimes_ns else None
                    ),
                    "stdev_ms": (
                        statistics.pstdev(runtimes_ns) / 1e6
                        if len(runtimes_ns) > 1
                        else 0.0
                    ),
                    "n_runs": len(runtimes_ns),
                    "commit": rec.get("commit_id", ""),
                    "name": name,
                }
            )
    rows.sort(key=lambda r: (r["dataset"], r["query"], r["format"]))
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        out.write_text("")
        return
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict], out: Path, include_dataset: bool) -> None:
    if include_dataset:
        header = (
            "| dataset | query | format | median (ms) | min (ms) | max (ms) | "
            "stdev (ms) | runs |"
        )
        sep = "|---|---|---|---:|---:|---:|---:|---:|"
    else:
        header = (
            "| query | format | median (ms) | min (ms) | max (ms) | "
            "stdev (ms) | runs |"
        )
        sep = "|---|---|---:|---:|---:|---:|---:|"

    lines = [header, sep]
    for r in rows:
        common = (
            f"{r['median_ms']:.2f} | "
            f"{(r['min_ms'] or 0):.2f} | "
            f"{(r['max_ms'] or 0):.2f} | "
            f"{r['stdev_ms']:.2f} | "
            f"{r['n_runs']}"
        )
        if include_dataset:
            lines.append(
                f"| {r['dataset']} | {r['query']} | {r['format']} | {common} |"
            )
        else:
            lines.append(f"| {r['query']} | {r['format']} | {common} |")
    out.write_text("\n".join(lines) + "\n")


def write_plot(rows: list[dict], out: Path, title: str) -> bool:
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

    datasets = sorted({r["dataset"] for r in rows})
    cmap = plt.get_cmap("tab10")
    color_for = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}

    labels = [
        f"{r['dataset']}/q{r['query']}" if len(datasets) > 1 else f"q{r['query']}"
        for r in rows
    ]
    values = [r["median_ms"] for r in rows]
    colors = [color_for[r["dataset"]] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.35), 5))
    ax.bar(labels, values, color=colors)
    ax.set_xlabel("query")
    ax.set_ylabel("median runtime (ms)")
    ax.set_title(title)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    if len(datasets) > 1:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=color_for[d], label=d) for d in datasets]
        ax.legend(handles=handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def process_single(input_path: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = input_path.parent.name
    rows = load(input_path, dataset_fallback=dataset)
    if not rows:
        print(f"no measurements in {input_path}", file=sys.stderr)
        return []

    csv_path = output_dir / "results.csv"
    md_path = output_dir / "summary.md"
    plot_path = output_dir / "plot.png"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, include_dataset=False)
    title = (
        f"{rows[0]['dataset'] or dataset} — "
        f"{rows[0]['engine']} / {rows[0]['format']}"
    )
    write_plot(rows, plot_path, title)
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")
    if plot_path.exists():
        print(f"  wrote {plot_path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="single raw.jsonl")
    g.add_argument(
        "--input-dir",
        type=Path,
        help="dir containing <bench>/raw.jsonl subdirs",
    )
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        rows = process_single(args.input, args.output_dir)
        if rows:
            print()
            print((args.output_dir / "summary.md").read_text())
        return 0 if rows else 1

    # --input-dir: walk each <bench>/raw.jsonl
    raw_files = sorted(args.input_dir.glob("*/raw.jsonl"))
    if not raw_files:
        print(f"no */raw.jsonl found under {args.input_dir}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for raw in raw_files:
        bench_name = raw.parent.name
        print(f"[{bench_name}]")
        rows = process_single(raw, raw.parent)
        all_rows.extend(rows)

    if not all_rows:
        print("no measurements collected", file=sys.stderr)
        return 1

    agg_csv = args.output_dir / "results.csv"
    agg_md = args.output_dir / "summary.md"
    agg_png = args.output_dir / "plot.png"
    write_csv(all_rows, agg_csv)
    write_markdown(all_rows, agg_md, include_dataset=True)
    write_plot(all_rows, agg_png, "All benchmarks — Vortex / DataFusion")
    print()
    print(f"aggregate -> {agg_csv}")
    print(f"aggregate -> {agg_md}")
    if agg_png.exists():
        print(f"aggregate -> {agg_png}")
    print()
    print(agg_md.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
