#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Build the consolidated per-query tables from results.md, results-warmfirst.md and prof/*.json."""

import glob
import json
import math
import os
import re
from collections.abc import Iterable
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))


def gmean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x and x > 0 and not math.isnan(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def sections(path: str) -> dict[tuple[str, str, str, str], list[dict[int, list[float]]]]:
    """Return {(kind, engine, suite, sf): [rows_by_query, ...]} in file order, skipping superseded."""
    out = {}
    if not os.path.exists(path):
        return out
    for s in re.split(r"\n(?=## )", open(path).read()):
        m = re.match(r"## (HOT|COLD) (\w+) (\w+) sf=(\S+)", s)
        if not m or "SUPERSEDED" in s.split("\n")[0]:
            continue
        rows = {}
        for line in s.split("\n"):
            mm = re.match(r"\| \w+?_q(\d+)/\S+ \| (.*) \|$", line)
            if mm:
                rows[int(mm.group(1))] = [float(x) if x.strip() else float("nan") for x in mm.group(2).split("|")]
        out.setdefault(m.groups(), []).append(rows)
    return out


def prof(engine: str, suite: str, sf: str, prefix: str = "") -> dict[tuple[int, str], dict[str, Any]]:
    d = os.path.join(HERE, "prof", f"{prefix}{engine}-{suite}-{sf}")
    out = {}
    for f in glob.glob(os.path.join(d, "q*-V?.json")):
        m = re.search(r"q(\d+)-(V\d)\.json$", f)
        out[(int(m.group(1)), m.group(2))] = json.load(open(f))
    return out


def main():
    res = sections(os.path.join(HERE, "results.md"))
    first = sections(os.path.join(HERE, "results-warmfirst.md"))
    suites = (("tpch", "1.0"), ("tpch", "10.0"), ("clickbench", "-"))
    print("# Consolidated V1 vs V2 results (binaries of 9c6392e5aa)\n")
    print("## Geomeans\n")
    print(
        "| engine | suite | hot V1 ms | hot V2 ms | hot V2/V1 (median) | hot V2/V1 (min) | hot pass-1 V1 / V2 / ratio "
        "| cold V1 ms | cold V2 ms | cold V2/V1 | warm-first V1 ms | warm-first V2 ms | MB paged in V1 | V2 |"
    )
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for eng in ("duckdb", "datafusion"):
        for suite, sf in suites:
            hots = res.get(("HOT", eng, suite, sf), [])
            h, h1 = hots[-1], hots[0]
            c = res[("COLD", eng, suite, sf)][-1]
            w = first.get(("COLD", eng, suite, sf), [{}])[-1]
            qs = sorted(h)
            p1 = (
                f"{gmean(h1[q][0] for q in qs):.1f} / {gmean(h1[q][1] for q in qs):.1f} "
                f"/ {gmean(h1[q][1] / h1[q][0] for q in qs):.3f}"
            )
            print(
                f"| {eng} | {suite} {sf} | {gmean(h[q][0] for q in qs):.1f} | {gmean(h[q][1] for q in qs):.1f} "
                f"| {gmean(h[q][1] / h[q][0] for q in qs):.3f} | {gmean(h[q][4] / h[q][3] for q in qs):.3f} | {p1} "
                f"| {gmean(c[q][0] for q in qs):.1f} | {gmean(c[q][1] for q in qs):.1f} "
                f"| {gmean(c[q][1] / c[q][0] for q in qs):.3f} "
                f"| {gmean(w[q][0] for q in qs) if w else float('nan'):.1f} "
                f"| {gmean(w[q][1] for q in qs) if w else float('nan'):.1f} "
                f"| {sum(c[q][3] for q in qs):.0f} | {sum(c[q][4] for q in qs):.0f} |"
            )
    for suite, sf in suites:
        print(
            f"\n## Per query: {suite} {sf} (ms; hot = median steady-state of the second pass; "
            "cold = median of 3 evicted fresh processes; wf = warm page cache, fresh process, first iteration; "
            "scan% = scan share of wall from the V1/V2 hot profile)\n"
        )
        print(
            "| q | DuckDB hot V1 | hot V2 | V2/V1 | cold V1 | cold V2 | V2/V1 | wf V1 | MB V1 | MB V2 "
            "| scan% V1 | scan% V2 | DataFusion hot V1 | hot V2 | V2/V1 | cold V1 | cold V2 | V2/V1 | wf V1 "
            "| MB V1 | MB V2 | scan% V1 | scan% V2 |"
        )
        print("|---|" + "---:|" * 22)
        data = {}
        for eng in ("duckdb", "datafusion"):
            data[eng] = (
                res[("HOT", eng, suite, sf)][-1],
                res[("COLD", eng, suite, sf)][-1],
                first.get(("COLD", eng, suite, sf), [{}])[-1],
                prof(eng, suite, sf),
            )
        qs = sorted(data["duckdb"][0])
        for q in qs:
            cells = []
            for eng in ("duckdb", "datafusion"):
                h, c, w, p = data[eng]
                s1 = p.get((q, "V1"), {}).get("scan_share_wall", float("nan"))
                s2 = p.get((q, "V2"), {}).get("scan_share_wall", float("nan"))
                wf = w[q][0] if w and q in w else float("nan")
                cells.append(
                    f"{h[q][0]:.1f} | {h[q][1]:.1f} | {h[q][1] / h[q][0]:.2f} | {c[q][0]:.1f} | {c[q][1]:.1f} "
                    f"| {c[q][1] / c[q][0]:.2f} | {wf:.1f} | {c[q][3]:.0f} | {c[q][4]:.0f} "
                    f"| {100 * s1:.0f} | {100 * s2:.0f}"
                )
            print(f"| q{q} | " + " | ".join(cells) + " |")
        cells = []
        for eng in ("duckdb", "datafusion"):
            h, c, w, p = data[eng]
            cells.append(
                f"**{gmean(h[q][0] for q in qs):.1f}** | **{gmean(h[q][1] for q in qs):.1f}** "
                f"| **{gmean(h[q][1] / h[q][0] for q in qs):.3f}** | **{gmean(c[q][0] for q in qs):.1f}** "
                f"| **{gmean(c[q][1] for q in qs):.1f}** | **{gmean(c[q][1] / c[q][0] for q in qs):.3f}** "
                f"| **{gmean(w[q][0] for q in qs) if w else float('nan'):.1f}** | {sum(c[q][3] for q in qs):.0f} "
                f"| {sum(c[q][4] for q in qs):.0f} | | "
            )
        print("| **geomean (MB: sum)** | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
