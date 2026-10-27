#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render `datafusion-bench --display-format gh-json` results as per-query runtimes, real S3
against simulated S3.

Usage: s3-query-report.py RESULTS_DIR [--report report.md]

RESULTS_DIR is searched recursively for `*.json` files holding one JSON measurement per line. The
`runner` field names the store each run used (`real`, `simulated-model`, `simulated-throttle`).
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

STORE_ORDER = {"real": 0, "simulated-model": 1, "simulated-throttle": 2}


def load(results: Path) -> dict[str, dict[int, dict[str, float]]]:
    """dataset -> query -> store -> median seconds."""
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(results.rglob("*.json")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            m = json.loads(line)
            name = m.get("name", "")
            if "_q" not in name or "runner" not in m or m.get("unit") != "ns":
                continue
            dataset, rest = name.split("_q", 1)
            query = int(rest.split("/", 1)[0])
            out[dataset][query][m["runner"]] = m["value"] / 1e9
    return out


def render(datasets: dict[str, dict[int, dict[str, float]]]) -> str:
    md = ["# Queries: real vs simulated S3", ""]
    md.append(
        "DataFusion over Vortex files, median runtime per query. Simulated stores serve an in-memory "
        "copy of the same files with the latency model fitted in this run."
    )
    for dataset in sorted(datasets):
        queries = datasets[dataset]
        stores = sorted({s for q in queries.values() for s in q}, key=lambda s: (STORE_ORDER.get(s, 9), s))
        sims = [s for s in stores if s != "real"]
        header = ["query"] + [f"{s} s" for s in stores] + [f"{s} error" for s in sims if "real" in stores]
        rows, totals = [], dict.fromkeys(stores, 0.0)
        ratios: dict[str, list[float]] = defaultdict(list)
        # Totals and ratios only over queries that every store completed, so they compare alike.
        complete = [q for q in sorted(queries) if all(s in queries[q] for s in stores)]
        for q in sorted(queries):
            times = queries[q]
            row = [f"q{q:02}"] + [f"{times[s]:.2f}" if s in times else "failed" for s in stores]
            if "real" in stores:
                row += [
                    f"{(times[s] - times['real']) / times['real']:+.0%}" if s in times and "real" in times else "-"
                    for s in sims
                ]
            rows.append(row)
            if q in complete:
                for s in stores:
                    totals[s] += times[s]
                    if "real" in stores and s != "real" and times["real"] > 0:
                        ratios[s].append(times[s] / times["real"])
        total = ["**total**"] + [f"**{totals[s]:.2f}**" for s in stores]
        if "real" in stores and totals["real"]:
            total += [f"**{(totals[s] - totals['real']) / totals['real']:+.0%}**" for s in sims]
        rows.append(total)
        md += ["", f"## {dataset}", ""]
        summary = f"{len(queries)} queries"
        if len(complete) != len(queries):
            summary += f", {len(complete)} completed on every store (totals cover those)"
        for s in sims:
            if ratios[s]:
                geomean = math.exp(sum(math.log(r) for r in ratios[s]) / len(ratios[s]))
                summary += f". {s}: geometric mean {geomean:.2f}x real"
        md.append(summary + ".")
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
        print(f"no query results under {args.results}", file=sys.stderr)
        return 1
    report = render(datasets)
    print(report)
    if args.report:
        args.report.write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
