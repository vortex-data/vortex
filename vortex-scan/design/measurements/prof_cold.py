#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Cold-run attribution: evict the suite's data files, profile a fresh process running one
iteration of one query, repeat, and sum the per-run attributions.

usage: prof_cold.py <engine> <suite> <sf|-> <configs: V1,V2> <queries> <reps>
Writes prof/cold-<engine>-<suite>-<sf>/q<N>-<cfg>.json and prints one line per (query, config).
"""

import collections
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_ab  # noqa: E402
import prof_attr  # noqa: E402


def main():
    engine, suite, sf, cfgs, queries, reps = sys.argv[1:7]
    outdir = os.path.join(HERE, "prof", f"cold-{engine}-{suite}-{sf}")
    os.makedirs(outdir, exist_ok=True)
    for q in [int(x) for x in queries.split(",")]:
        for cfg_name in cfgs.split(","):
            cfg = 0 if cfg_name == "V1" else 1
            name, overrides = bench_ab.CONFIGS[cfg]
            res, kinds, phases = collections.Counter(), collections.Counter(), collections.Counter()
            times, wall_scan, wall_rounds = [], 0.0, 0.0
            for rep in range(int(reps)):
                bench_ab.evict_all(suite, sf)
                base = os.path.join(outdir, f"q{q}-{name}-r{rep}")
                env = dict(os.environ)
                env.pop("VORTEX_SCAN_V2", None)
                env.update(overrides)
                cmd = ["samply", "record", "--save-only", "--unstable-presymbolicate", "--rate", "1000",
                       "--output", base + ".json.gz", "--"] + bench_ab.base_cmd(engine, suite, sf, 1, q)[2:] + \
                      ["-o", base + ".jsonl"]
                p = subprocess.run(cmd, env=env, cwd=bench_ab.ROOT, capture_output=True, text=True)
                if p.returncode != 0:
                    print(f"FAILED q{q} {name}: {p.stderr[-300:]}", flush=True)
                    continue
                a = prof_attr.analyse(base + ".json.gz")
                res.update(a["res"])
                kinds.update(a["kinds"])
                phases.update(a["phases"])
                wall_scan += a["wall_scan"]
                wall_rounds += a["wall_rounds"]
                rts = [json.loads(line) for line in open(base + ".jsonl") if line.startswith("{")]
                times.append(rts[0]["all_runtimes"][0] / 1e6)
                if rep > 0:
                    for ext in (".json.gz", ".json.syms.json"):
                        os.remove(base + ext)
            a = dict(res=res, kinds=kinds, phases=phases, wall_scan=wall_scan, wall_rounds=wall_rounds,
                     total_cpu_ms=0.0, cpu_us=collections.Counter(), io_threads=0, threads=0)
            s = prof_attr.summarise(a)
            s.update(query=q, config=name, cold_ms=statistics.median(times), reps=int(reps))
            with open(os.path.join(outdir, f"q{q}-{name}.json"), "w") as f:
                json.dump(s, f)
            busy = max(s["scan_cpu"] + s["scan_wait"] + s["engine_cpu"], 1)
            io = sum(v for k, v in kinds.items() if k.startswith("io: pread"))
            print(f"COLD {engine} {suite} {sf} q{q} {name}: {s['cold_ms']:.0f} ms; scan wall {100 * s['scan_share_wall']:.0f}%; "
                  f"of busy thread-time: blocked-in-scan {100 * s['scan_wait'] / busy:.0f}%, in pread {100 * io / busy:.0f}%, "
                  f"other scan on-CPU {100 * (s['scan_cpu'] - io) / busy:.0f}%, engine {100 * s['engine_cpu'] / busy:.0f}%",
                  flush=True)


if __name__ == "__main__":
    main()
