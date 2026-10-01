#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Distribution of the scan's share of query time per engine/suite/config, from prof/*/q*-V?.json.

Two estimates per query:
  wall   = per-sampling-round share of active threads that are in the scan (on-CPU in scan code, in an
           IO-pool thread, or blocked inside the scan), averaged over rounds. IO-pool threads count.
  worker = thread-time share on engine worker threads only: (scan on-CPU outside IO threads + blocked
           inside scan) / (that + engine on-CPU). IO-pool threads are excluded, so a DataFusion worker
           that yields while its read is outstanding contributes nothing: a lower bound there.
"""
import glob, json, math, os, re, statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def gmean(xs):
    xs = [max(x, 1e-9) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


print("| engine-suite | cfg | n | scan share of wall: median | mean | time-weighted | >50% | >80% | <20% | worker-only: median | time-weighted "
      "| blocked-in-scan mean | geomean ratio if scan -100% | -50% | -30% | scan cut needed for 0.70 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for d in sorted(glob.glob(os.path.join(HERE, "prof", "*-*-*"))):
    name = os.path.basename(d)
    if name.startswith("cold-") or not os.path.isdir(d):
        continue
    data = {}
    for f in glob.glob(os.path.join(d, "q*-V?.json")):
        m = re.search(r"q(\d+)-(V\d)\.json$", f)
        data[(int(m.group(1)), m.group(2))] = json.load(open(f))
    for cfg in ("V1", "V2"):
        rows = [data[k] for k in sorted(data) if k[1] == cfg]
        if not rows:
            continue
        wall = [r["scan_share_wall"] for r in rows]
        t = [r["profiled_ms"] for r in rows]
        worker = []
        for r in rows:
            io_thread = r["phases"].get("io thread", 0)
            sw = r["scan_cpu"] - io_thread + r["scan_wait"]
            worker.append(sw / max(sw + r["engine_cpu"], 1))
        blocked = [r["scan_wait"] / max(r["scan_cpu"] + r["scan_wait"] + r["engine_cpu"], 1) for r in rows]
        tw = sum(w * x for w, x in zip(wall, t)) / sum(t)
        tww = sum(w * x for w, x in zip(worker, t)) / sum(t)
        # smallest uniform fractional cut f of scan cost giving geomean ratio <= 0.70
        need = next((f / 100 for f in range(1, 101) if gmean(1 - f / 100 * w for w in wall) <= 0.70), None)
        print(f"| {name} | {cfg} | {len(rows)} | {100 * statistics.median(wall):.0f}% | {100 * statistics.mean(wall):.0f}% | {100 * tw:.0f}% "
              f"| {sum(w > 0.5 for w in wall)} | {sum(w > 0.8 for w in wall)} | {sum(w < 0.2 for w in wall)} "
              f"| {100 * statistics.median(worker):.0f}% | {100 * tww:.0f}% | {100 * statistics.mean(blocked):.0f}% "
              f"| {gmean(1 - w for w in wall):.2f} | {gmean(1 - 0.5 * w for w in wall):.2f} | {gmean(1 - 0.3 * w for w in wall):.2f} "
              f"| {'%d%%' % round(need * 100) if need else 'unreachable'} |")
