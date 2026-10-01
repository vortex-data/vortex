#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Summarise prof/<engine>-<suite>-<sf>/q*-V?.json into per-suite scan-share tables.

usage: sweep_summary.py [--per-query]
"""

import glob
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def gmean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "q*-V?.json")):
        m = re.search(r"q(\d+)-(V\d)\.json$", f)
        out[(int(m.group(1)), m.group(2))] = json.load(open(f))
    return out


IO_KINDS = ("io: pread (kernel copy + in-kernel wait)", "io: open/close/stat syscalls", "io: dispatch + segment source")
GROUPS = [
    ("io read + dispatch", lambda k: k.startswith("io:") and "spin" not in k),
    ("decode", lambda k: k.startswith("decode:")),
    ("filter/expr kernels", lambda k: k.startswith("filter:")),
    ("pruning", lambda k: k.startswith("pruning")),
    ("sched/layout/dispatch", lambda k: k.startswith("scan:") or k.startswith("array executor")),
    ("convert to engine", lambda k: k.startswith("convert") or k.startswith("duckdb C API")),
]


def main():
    per_query = "--per-query" in sys.argv
    for d in sorted(glob.glob(os.path.join(HERE, "prof", "*-*-*"))):
        if not os.path.isdir(d):
            continue
        data = load(d)
        if not data:
            continue
        name = os.path.basename(d)
        print(f"\n## {name}")
        for cfg in ("V1", "V2"):
            qs = sorted(q for (q, c) in data if c == cfg)
            if not qs:
                continue
            rows = [data[(q, cfg)] for q in qs]
            wall = [r["scan_share_wall"] for r in rows]
            cpu = [r["scan_share_cpu"] for r in rows]
            blocked = [r["scan_wait"] / max(r["scan_cpu"] + r["scan_wait"] + r["engine_cpu"], 1) for r in rows]
            print(f"\n### {name} {cfg}: {len(qs)} queries")
            print(f"scan share of wall (per-round): median {100 * statistics.median(wall):.0f}%, "
                  f"min {100 * min(wall):.0f}%, max {100 * max(wall):.0f}%, mean {100 * statistics.mean(wall):.0f}%")
            print(f"scan share of on-CPU: median {100 * statistics.median(cpu):.0f}%, mean {100 * statistics.mean(cpu):.0f}%")
            print(f"threads blocked inside scan, share of busy thread-time: median {100 * statistics.median(blocked):.0f}%, "
                  f"mean {100 * statistics.mean(blocked):.0f}%")
            for f in (1.0, 0.5, 0.3):
                g = gmean(1 - f * w for w in wall)
                n30 = sum(1 for w in wall if 1 - f * w <= 0.70)
                print(f"  if scan cost falls by {int(f * 100)}%: geomean time ratio {g:.3f} "
                      f"({100 * (1 - g):.0f}% faster); queries reaching >=30% faster: {n30}/{len(wall)}")
            # aggregate composition of scan thread-time, query-weighted mean of shares of busy thread-time
            comp = {g: [] for g, _ in GROUPS}
            comp["blocked in scan (IO wait/handoff)"] = []
            comp["engine on-CPU"] = []
            for r in rows:
                busy = max(r["scan_cpu"] + r["scan_wait"] + r["engine_cpu"], 1)
                for g, pred in GROUPS:
                    comp[g].append(sum(v for k, v in r["kinds"].items() if pred(k)) / busy)
                comp["blocked in scan (IO wait/handoff)"].append(r["scan_wait"] / busy)
                comp["engine on-CPU"].append(r["engine_cpu"] / busy)
            print("  mean share of busy thread-time per query: " + "; ".join(
                f"{g} {100 * statistics.mean(v):.1f}%" for g, v in comp.items()))
            spin = [r["spin_cpu_ms"] / max(r["cpu_delta_ms"], 1) for r in rows]
            print(f"  spin/yield CPU (threadCPUDelta) as share of all CPU: mean {100 * statistics.mean(spin):.1f}%, "
                  f"max {100 * max(spin):.1f}%; io-pool threads: median {statistics.median(r.get('io_threads', 0) for r in rows):.0f}, "
                  f"max {max(r.get('io_threads', 0) for r in rows)}")
            if per_query:
                print("| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |")
                print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for i, q in enumerate(qs):
                    r = rows[i]
                    print(f"| q{q} | {r.get('hot_ms') or float('nan'):.1f} | {r['profiled_ms']:.1f} | {100 * wall[i]:.0f} | {100 * cpu[i]:.0f} "
                          f"| {100 * blocked[i]:.0f} | {100 * comp['io read + dispatch'][i]:.0f} | {100 * comp['decode'][i]:.0f} "
                          f"| {100 * comp['filter/expr kernels'][i]:.0f} | {100 * comp['sched/layout/dispatch'][i]:.0f} "
                          f"| {100 * comp['convert to engine'][i]:.0f} | {100 * comp['engine on-CPU'][i]:.0f} |")


if __name__ == "__main__":
    main()
