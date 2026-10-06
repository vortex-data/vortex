#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Profile every query of a suite with Samply (hot) and attribute it with prof_attr.py.

usage: prof_sweep.py <engine> <suite> <sf|-> <configs: V1,V2> [queries] [--rate N] [--secs S] [--keep]

Iteration count per query is chosen from the hot timing results in raw/ so each profile covers
about <secs> seconds of query execution. One JSON summary per (query, config) is written to
prof/<engine>-<suite>-<sf>/q<N>-<cfg>.json; profiles are deleted unless --keep is given.
"""

import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_ab  # noqa: E402
import prof_attr  # noqa: E402


def hot_medians(engine, suite, sf):
    med = {}
    for path in glob.glob(os.path.join(HERE, "raw", f"hot-{engine}-{suite}-{sf}-*-p*-V1.jsonl")):
        for line in open(path):
            if line.startswith("{"):
                r = json.loads(line)
                q = bench_ab.qkey(r["name"])
                med.setdefault(q, []).extend(x / 1e6 for x in r["all_runtimes"][1:])
    return {q: statistics.median(v) for q, v in med.items()}


def main():
    args = sys.argv[1:]
    engine, suite, sf, cfgs = args[:4]
    rest = args[4:]
    rate = int(rest[rest.index("--rate") + 1]) if "--rate" in rest else 1000
    secs = float(rest[rest.index("--secs") + 1]) if "--secs" in rest else 2.0
    keep = "--keep" in rest
    queries = rest[0] if rest and not rest[0].startswith("--") else None
    qs = bench_ab.list_queries(engine, suite, sf, queries)
    med = hot_medians(engine, suite, sf)
    outdir = os.path.join(HERE, "prof", f"{engine}-{suite}-{sf}")
    os.makedirs(outdir, exist_ok=True)
    for q in qs:
        iters = max(3, min(400, math.ceil(secs * 1000 / med.get(q, 100.0))))
        for cfg_name in cfgs.split(","):
            cfg = 0 if cfg_name == "V1" else 1
            name, overrides = bench_ab.CONFIGS[cfg]
            base = os.path.join(outdir, f"q{q}-{name}")
            prof = base + ".json.gz"
            env = dict(os.environ)
            env.pop("VORTEX_SCAN_V2", None)
            env.update(overrides)
            cmd = ["samply", "record", "--save-only", "--unstable-presymbolicate", "--rate", str(rate),
                   "--output", prof, "--"] + bench_ab.base_cmd(engine, suite, sf, iters, q)[2:] + \
                  ["-o", base + ".jsonl"]
            if engine == "duckdb":
                cmd.append("--reuse")
            p = subprocess.run(cmd, env=env, cwd=bench_ab.ROOT, capture_output=True, text=True)
            if p.returncode != 0 or not os.path.exists(prof):
                print(f"FAILED q{q} {name}: {p.stderr[-500:]}", flush=True)
                continue
            try:
                a = prof_attr.analyse(prof)
                s = prof_attr.summarise(a)
            except Exception as e:  # noqa: BLE001
                print(f"ANALYSIS FAILED q{q} {name}: {e}", flush=True)
                continue
            rts = [json.loads(line) for line in open(base + ".jsonl") if line.startswith("{")]
            prof_ms = statistics.median(x / 1e6 for x in rts[0]["all_runtimes"][1:]) if rts else float("nan")
            s.update(query=q, config=name, iters=iters, rate=rate, hot_ms=med.get(q), profiled_ms=prof_ms,
                     load=os.getloadavg()[0])
            with open(base + ".json", "w") as f:
                json.dump(s, f)
            print(f"{engine} {suite} {sf} q{q} {name}: iters={iters} hot={med.get(q, float('nan')):.1f}ms "
                  f"profiled={prof_ms:.1f}ms scan_wall={100 * s['scan_share_wall']:.0f}% "
                  f"scan_cpu={100 * s['scan_share_cpu']:.0f}% scan_busy={100 * s['scan_share_busy']:.0f}% "
                  f"load={s['load']:.1f}", flush=True)
            if not keep:
                for ext in (".json.gz", ".json.syms.json"):
                    try:
                        os.remove(base + ext)
                    except OSError:
                        pass


if __name__ == "__main__":
    main()
