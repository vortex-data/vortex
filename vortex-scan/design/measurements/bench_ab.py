#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Interleaved V1 (VORTEX_SCAN_V2 unset) vs V2 (VORTEX_SCAN_V2=1) runs, hot and cold.

usage:
  bench_ab.py hot  <engine> <suite> <sf|-> <pairs> <iters> [queries]
  bench_ab.py cold <engine> <suite> <sf|-> <reps> [queries]

env:
  ROOT     worktree whose vortex-bench/data is used (the binaries are run with cwd=ROOT)
  BIN_DIR  directory holding duckdb-bench and datafusion-bench
  A_ENV / B_ENV  optional "K=V K=V" overrides for the two configurations (default: A = V1 with
           VORTEX_SCAN_V2 unset, B = V2 with VORTEX_SCAN_V2=1)

hot:  one discarded warm-up process per config, then <pairs> alternating A/B processes, each
      running every query <iters> times. Steady state = all iterations except the first of each
      process. Per-query median and min; geomean over queries. Also CPU seconds and instructions
      retired per process from /usr/bin/time -l.
cold: per query, rep and config: evict every data file of the suite from the page cache
      (evict.py: msync(MS_INVALIDATE), residency verified by mincore, abort if >0.5% stays),
      run a fresh process with -i 1, then count the pages the run brought back into the cache
      (= bytes physically read, at 16 KiB page granularity, including OS read-ahead).
      A/B order alternates. Per-query median over reps; geomean over queries.
Raw gh-json and /usr/bin/time -l output are kept under raw/; tables are appended to results.md.
"""

import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evict as ev  # noqa: E402

RAW = os.path.join(HERE, "raw")
RESULTS = os.path.join(HERE, "results.md")
# The worktree this file is checked out in: vortex-scan/design/measurements is three levels down.
ROOT = os.environ.get("ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
BIN_DIR = os.environ.get("BIN_DIR", os.path.join(ROOT, "target", "bench-bins", "9c6392e5aa"))
os.makedirs(RAW, exist_ok=True)


def parse_env(s: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in s.split()) if s else {}


CONFIGS = [
    ("V1", parse_env(os.environ.get("A_ENV", ""))),
    ("V2", parse_env(os.environ.get("B_ENV", "VORTEX_SCAN_V2=1"))),
]


def emit(text: str):
    print(text, flush=True)
    with open(RESULTS, "a") as f:
        f.write(text + "\n")


def data_dir(suite: str, sf: str) -> str:
    if suite == "tpch":
        return f"{ROOT}/vortex-bench/data/tpch/{sf}/vortex-file-compressed"
    return f"{ROOT}/vortex-bench/data/clickbench_partitioned/vortex-file-compressed"


def base_cmd(engine: str, suite: str, sf: str, iters: int, q: int | str | None) -> list[str]:
    cmd = [
        "/usr/bin/time",
        "-l",
        f"{BIN_DIR}/{engine}-bench",
        suite,
        "--formats",
        "vortex",
        "-i",
        str(iters),
        "--hide-progress-bar",
        "-d",
        "gh-json",
    ]
    if q is not None:
        cmd += ["-q", str(q)]
    if suite == "tpch":
        cmd += ["--opt", f"scale-factor={sf}"]
    return cmd


def run(engine: str, suite: str, sf: str, iters: int, q: int | str | None, cfg: int, tag: str) -> dict[str, Any]:
    name, overrides = CONFIGS[cfg]
    env = dict(os.environ)
    for k in ("VORTEX_SCAN_V2", "A_ENV", "B_ENV"):
        env.pop(k, None)
    env.update(overrides)
    out = os.path.join(RAW, f"{tag}.jsonl")
    cmd = base_cmd(engine, suite, sf, iters, q) + ["-o", out]
    load = os.getloadavg()
    p = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    with open(os.path.join(RAW, f"{tag}.log"), "w") as f:
        f.write(f"# cwd={ROOT} load={load[0]:.2f} config={name}\n")
        f.write(" ".join(f"{k}={v}" for k, v in overrides.items()) + " " + " ".join(cmd) + "\n")
        f.write(p.stdout)
        f.write(p.stderr)
    if p.returncode != 0:
        sys.exit(f"FAILED {tag}: {p.stderr[-3000:]}")
    m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", p.stderr)
    ins = re.search(r"(\d+)\s+instructions retired", p.stderr)
    rows = [json.loads(line) for line in open(out) if line.strip().startswith("{")]
    rts = {r["name"]: [x / 1e6 for x in r["all_runtimes"]] for r in rows}
    return dict(
        real=float(m[1]),
        user=float(m[2]),
        sys=float(m[3]),
        cpu=float(m[2]) + float(m[3]),
        ins=int(ins[1]) if ins else 0,
        rts=rts,
        load=load[0],
    )


def gmean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def qkey(n: str) -> int:
    return int(re.findall(r"\d+", n.split("/")[0])[-1])


def hot(engine: str, suite: str, sf: str, pairs: int, iters: int, queries: str | None):
    stamp = f"hot-{engine}-{suite}-{sf}-{int(time.time())}"
    for cfg in (0, 1):
        run(engine, suite, sf, 1, queries, cfg, f"{stamp}-warm-{CONFIGS[cfg][0]}")
    res = {0: [], 1: []}
    for i in range(pairs):
        for cfg in (0, 1) if i % 2 == 0 else (1, 0):
            r = run(engine, suite, sf, iters, queries, cfg, f"{stamp}-p{i}-{CONFIGS[cfg][0]}")
            res[cfg].append(r)
            print(
                f"pair {i} {CONFIGS[cfg][0]} real={r['real']:.2f}s cpu={r['cpu']:.2f}s "
                f"ins={r['ins'] / 1e9:.2f}G load={r['load']:.1f}",
                flush=True,
            )
    med, mn = {}, {}
    for cfg in (0, 1):
        med[cfg], mn[cfg] = {}, {}
        for n in res[cfg][0]["rts"]:
            xs = [x for r in res[cfg] for x in r["rts"][n][1:]] or [x for r in res[cfg] for x in r["rts"][n]]
            med[cfg][n] = statistics.median(xs)
            mn[cfg][n] = min(xs)
    names = sorted(med[0], key=qkey)
    loads = [r["load"] for cfg in (0, 1) for r in res[cfg]]
    emit(
        f"\n## HOT {engine} {suite} sf={sf} — {pairs} interleaved pairs x {iters} iters, "
        f"first iter of each process dropped (load avg {min(loads):.1f}-{max(loads):.1f}) [{stamp}]"
    )
    emit("| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    for n in names:
        emit(
            f"| {n} | {med[0][n]:.2f} | {med[1][n]:.2f} | {med[1][n] / med[0][n]:.3f} "
            f"| {mn[0][n]:.2f} | {mn[1][n]:.2f} | {mn[1][n] / mn[0][n]:.3f} |"
        )
    emit(
        f"| **geomean** | **{gmean(med[0].values()):.2f}** | **{gmean(med[1].values()):.2f}** "
        f"| **{gmean(med[1][n] / med[0][n] for n in names):.3f}** "
        f"| **{gmean(mn[0].values()):.2f}** | **{gmean(mn[1].values()):.2f}** "
        f"| **{gmean(mn[1][n] / mn[0][n] for n in names):.3f}** |"
    )
    emit(
        f"| **sum** | {sum(med[0].values()):.0f} | {sum(med[1].values()):.0f} "
        f"| {sum(med[1].values()) / sum(med[0].values()):.3f} | {sum(mn[0].values()):.0f} "
        f"| {sum(mn[1].values()):.0f} | {sum(mn[1].values()) / sum(mn[0].values()):.3f} |"
    )
    agg = {}
    for cfg in (0, 1):
        agg[cfg] = {k: statistics.median(r[k] for r in res[cfg]) for k in ("cpu", "user", "sys", "ins", "real")}
        emit(
            f"\n{CONFIGS[cfg][0]} per process (median of {pairs}): wall {agg[cfg]['real']:.2f}s, "
            f"cpu {agg[cfg]['cpu']:.2f}s (user {agg[cfg]['user']:.2f} sys {agg[cfg]['sys']:.2f}), "
            f"instructions {agg[cfg]['ins'] / 1e9:.2f}G"
        )
    emit(
        f"\nV2/V1 per process: instructions {agg[1]['ins'] / agg[0]['ins']:.3f}, "
        f"cpu {agg[1]['cpu'] / agg[0]['cpu']:.3f}, wall {agg[1]['real'] / agg[0]['real']:.3f}"
    )


def list_queries(engine: str, suite: str, sf: str, queries: str | None) -> list[int]:
    if queries:
        return [int(x) for x in queries.split(",")]
    cmd = base_cmd(engine, suite, sf, 1, None)[2:] + ["--print-queries"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [int(x) for x in p.stdout.split() if x.isdigit()]


def evict_all(suite: str, sf: str) -> tuple[int, int, int]:
    tb, ta, tp, _ = ev.sweep([data_dir(suite, sf)], True)
    if tp and ta / tp > 0.005:
        sys.exit(f"eviction failed: {ta}/{tp} pages still resident")
    return tb, ta, tp


def cold(engine: str, suite: str, sf: str, reps: int, queries: str | None):
    stamp = f"cold-{engine}-{suite}-{sf}-{int(time.time())}"
    qs = list_queries(engine, suite, sf, queries)
    times = {0: {}, 1: {}}
    read_mb = {0: {}, 1: {}}
    ins = {0: {}, 1: {}}
    loads = []
    total_mb = 0.0
    for rep in range(reps):
        for q in qs:
            for cfg in (0, 1) if (rep + q) % 2 == 0 else (1, 0):
                evict_all(suite, sf)
                tag = f"{stamp}-r{rep}-q{q}-{CONFIGS[cfg][0]}"
                r = run(engine, suite, sf, 1, q, cfg, tag)
                tb, _, tp, per = ev.sweep([data_dir(suite, sf)], False)
                total_mb = tp * ev.PAGE / 1e6
                mb = tb * ev.PAGE / 1e6
                with open(os.path.join(RAW, f"{tag}.log"), "a") as f:
                    f.write(f"# resident after run: {mb:.1f} MB of {total_mb:.1f} MB\n")
                    for path, (b, _, n) in per.items():
                        if b:
                            f.write(
                                f"#   {os.path.basename(path)}: {b * ev.PAGE / 1e6:.2f} MB of "
                                f"{n * ev.PAGE / 1e6:.2f} MB\n"
                            )
                ((name, (t,)),) = r["rts"].items()
                times[cfg].setdefault(name, []).append(t)
                read_mb[cfg].setdefault(name, []).append(mb)
                ins[cfg].setdefault(name, []).append(r["ins"])
                loads.append(r["load"])
                print(
                    f"rep {rep} q{q} {CONFIGS[cfg][0]} {t:.1f}ms read={mb:.1f}MB "
                    f"ins={r['ins'] / 1e9:.2f}G load={r['load']:.1f}",
                    flush=True,
                )
    med = {c: {n: statistics.median(ts) for n, ts in times[c].items()} for c in (0, 1)}
    rd = {c: {n: statistics.median(ts) for n, ts in read_mb[c].items()} for c in (0, 1)}
    names = sorted(med[0], key=qkey)
    emit(
        f"\n## COLD {engine} {suite} sf={sf} — {reps} reps, fresh process per query, data files "
        f"({total_mb:.0f} MB) evicted from the page cache before every process "
        f"(load avg {min(loads):.1f}-{max(loads):.1f}) [{stamp}]"
    )
    emit("| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    for n in names:
        ratio_mb = rd[1][n] / rd[0][n] if rd[0][n] else float("nan")
        emit(
            f"| {n} | {med[0][n]:.1f} | {med[1][n]:.1f} | {med[1][n] / med[0][n]:.3f} "
            f"| {rd[0][n]:.1f} | {rd[1][n]:.1f} | {ratio_mb:.3f} |"
        )
    emit(
        f"| **geomean** | **{gmean(med[0].values()):.1f}** | **{gmean(med[1].values()):.1f}** "
        f"| **{gmean(med[1][n] / med[0][n] for n in names):.3f}** "
        f"| {sum(rd[0].values()):.0f} (sum) | {sum(rd[1].values()):.0f} (sum) "
        f"| {sum(rd[1].values()) / max(sum(rd[0].values()), 1e-9):.3f} |"
    )
    emit(
        f"| **sum** | {sum(med[0].values()):.0f} | {sum(med[1].values()):.0f} "
        f"| {sum(med[1].values()) / sum(med[0].values()):.3f} | | | |"
    )


def main():
    mode, engine, suite, sf = sys.argv[1:5]
    rest = sys.argv[5:]
    if mode == "hot":
        hot(engine, suite, sf, int(rest[0]), int(rest[1]), rest[2] if len(rest) > 2 else None)
    else:
        cold(engine, suite, sf, int(rest[0]), rest[1] if len(rest) > 1 else None)


if __name__ == "__main__":
    main()
