#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Attribution table (% of busy thread-time) for chosen queries from prof/*/q*-V?.json.
usage: attr_table.py <engine>-<suite>-<sf>:<q,q,...> [...]
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
print("| engine suite | q | cfg | ms under profile | scan share of wall | blocked in scan (IO wait/handoff) | pread | open/close | IO dispatch "
      "| decode total | decode detail | filter/expr kernels | pruning | layout/plan/async/dispatch | convert to engine | engine on-CPU | spin CPU share | IO-pool threads |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
for arg in sys.argv[1:]:
    name, qs = arg.split(":")
    for q in qs.split(","):
        for cfg in ("V1", "V2"):
            r = json.load(open(os.path.join(HERE, "prof", name, f"q{q}-{cfg}.json")))
            k = r["kinds"]
            busy = max(r["scan_cpu"] + r["scan_wait"] + r["engine_cpu"], 1)
            pct = lambda v: f"{100 * v / busy:.1f}"
            dec = {a.replace("decode: ", ""): v for a, v in k.items() if a.startswith("decode:")}
            detail = ", ".join(f"{a} {100 * v / busy:.1f}" for a, v in sorted(dec.items(), key=lambda kv: -kv[1])[:4])
            g = lambda pred: sum(v for a, v in k.items() if pred(a))
            sched = g(lambda a: a.startswith("scan:") or a.startswith("array executor"))
            conv = g(lambda a: a.startswith("convert") or a.startswith("duckdb C API"))
            print(f"| {name} | q{q} | {cfg} | {r['profiled_ms']:.1f} | {100 * r['scan_share_wall']:.0f}% | {pct(r['scan_wait'])} "
                  f"| {pct(g(lambda a: a.startswith('io: pread')))} | {pct(g(lambda a: a.startswith('io: open')))} "
                  f"| {pct(g(lambda a: a.startswith('io: dispatch')))} | {pct(sum(dec.values()))} | {detail} "
                  f"| {pct(g(lambda a: a.startswith('filter:')))} | {pct(g(lambda a: a.startswith('pruning')))} | {pct(sched)} | {pct(conv)} "
                  f"| {pct(r['engine_cpu'])} | {100 * r['spin_cpu_ms'] / max(r['cpu_delta_ms'], 1):.1f}% | {r.get('io_threads', 0)} |")
