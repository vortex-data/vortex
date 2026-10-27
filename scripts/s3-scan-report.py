#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render `s3-latency-bench scan` results as per-table runtime tables, real against simulated S3.

Usage: s3-scan-report.py RESULTS_DIR [--report report.md]

RESULTS_DIR is searched recursively for the JSON files `scan` writes, one per dataset and store.
Each table's runtime is the median over iterations; simulated stores also show their error against
real S3.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

MIB = 1 << 20
STORE_ORDER = {"real": 0, "simulated-model": 1, "simulated-throttle": 2}


def median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2] if s else float("nan")


def load(results: Path) -> dict[str, dict[str, dict]]:
    """dataset -> store -> scan result."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(results.rglob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and {"dataset", "store", "tables"} <= data.keys():
            out[data["dataset"]][data["store"]] = data
    return out


def render(datasets: dict[str, dict[str, dict]]) -> str:
    md = ["# Full scans: real vs simulated S3", ""]
    md.append(
        "Every table of each dataset scanned end to end with `vortex-file` and decoded to canonical "
        "arrays, timed per table (median over iterations). Simulated stores serve an in-memory copy "
        "of the same files with the latency model fitted in this run."
    )
    for dataset in sorted(datasets):
        stores = sorted(datasets[dataset], key=lambda s: (STORE_ORDER.get(s, 9), s))
        by_table: dict[str, dict[str, dict]] = defaultdict(dict)
        for store in stores:
            for t in datasets[dataset][store]["tables"]:
                by_table[t["table"]][store] = t
        sims = [s for s in stores if s != "real"]
        header = ["table", "files", "MiB", "rows"] + [f"{s} s" for s in stores]
        header += [f"{s} error" for s in sims if "real" in stores]
        rows = []
        totals = dict.fromkeys(stores, 0.0)
        bytes_total = 0
        for table in sorted(by_table):
            cells = by_table[table]
            any_cell = next(iter(cells.values()))
            bytes_total += any_cell["bytes"]
            times = {s: median(cells[s]["runs_s"]) for s in stores if s in cells}
            for s, t in times.items():
                totals[s] += t
            row = [table, str(any_cell["files"]), f"{any_cell['bytes'] / MIB:.0f}", f"{any_cell['rows']:,}"]
            row += [f"{times[s]:.2f}" if s in times else "-" for s in stores]
            if "real" in times:
                row += [f"{(times[s] - times['real']) / times['real']:+.0%}" if s in times else "-" for s in sims]
            rows.append(row)
        total = ["**total**", "", f"{bytes_total / MIB:.0f}", ""] + [f"**{totals[s]:.2f}**" for s in stores]
        if "real" in stores and totals["real"]:
            total += [f"**{(totals[s] - totals['real']) / totals['real']:+.0%}**" for s in sims]
        rows.append(total)
        iterations = max(datasets[dataset][s]["iterations"] for s in stores)
        md += ["", f"## {dataset}", ""]
        if "real" in stores and totals["real"]:
            md.append(
                f"{len(by_table)} tables, {bytes_total / MIB / 1024:.1f} GiB, {iterations} iterations. "
                f"Real S3 full scan: {totals['real']:.1f} s ({bytes_total / MIB / totals['real']:.0f} MiB/s)."
            )
            md.append("")
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        md += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(md) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    datasets = load(args.results)
    if not datasets:
        print(f"no scan results under {args.results}", file=sys.stderr)
        return 1
    report = render(datasets)
    print(report)
    if args.report:
        args.report.write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
