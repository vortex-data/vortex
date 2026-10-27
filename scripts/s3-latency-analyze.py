#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Aggregate s3-latency-bench runs into a Markdown report and a latency model.

Usage: s3-latency-analyze.py RESULTS_DIR [--report report.md] [--model model.json]

RESULTS_DIR is searched recursively for `requests.csv`, `scans.json` and `meta.json`, one set per
run. Only the standard library is used, so this runs on any CI image.

The model (`--model`, schema `vortex-s3-latency-model/1`) describes one request as

    latency = first_byte + bytes / min(per_connection_bytes_per_s, nic_bytes_per_s / active_bodies)

with `first_byte` drawn from the measured inverse CDF. It is fitted only from the microbenchmark
phases; the scan phase is then replayed through a simulator driven by the model, and the predicted
scan wall times are reported next to the measured ones as a validation of the model.
"""

import argparse
import csv
import heapq
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "vortex-s3-latency-model/1"
QUANTILES = (0.5, 0.9, 0.99)
# Points of the inverse CDFs stored in the model.
CDF_POINTS = sorted([i / 100 for i in range(101)] + [0.995, 0.999])
# Runs that start this soon after the dataset upload see slower, freshly written objects (small GETs
# took ~90 ms instead of ~26 ms until the objects were about two minutes old). They are reported
# separately rather than mixed into the model.
FRESH_AGE_S = 150
MIB = 1 << 20


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def fmt_bytes(n: float) -> str:
    for unit, scale in (("MiB", MIB), ("KiB", 1 << 10)):
        if n >= scale:
            v = n / scale
            return f"{v:.0f} {unit}" if v == int(v) else f"{v:.1f} {unit}"
    return f"{n:.0f} B"


def bucket(n: int) -> int:
    """Power-of-two bucket floor, so log-uniform sizes group with their neighbours."""
    return 1 << max(0, int(n).bit_length() - 1)


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for small dense systems."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            return [float("nan")] * n
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def least_squares(xs: list[list[float]], ys: list[float]) -> tuple[list[float], float]:
    """Return coefficients and R^2 of ys ~ xs (xs rows include the intercept column)."""
    k = len(xs[0])
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(k)] for i in range(k)]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(k)]
    coef = solve(ata, aty)
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys) or 1.0
    ss_res = sum((y - sum(c * v for c, v in zip(coef, x))) ** 2 for x, y in zip(xs, ys))
    return coef, 1 - ss_res / ss_tot


def load(results: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rows, scans, metas = [], [], []
    for path in sorted(results.rglob("requests.csv")):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                for key in (
                    "scan_id",
                    "file",
                    "offset",
                    "length",
                    "segments",
                    "useful_bytes",
                    "concurrency",
                    "inflight_at_start",
                    "seq",
                    "start_us",
                    "headers_us",
                    "first_byte_us",
                    "end_us",
                    "bytes",
                    "chunks",
                ):
                    r[key] = int(r[key])
                r["total_ms"] = (r["end_us"] - r["start_us"]) / 1e3
                r["ttfb_ms"] = (r["first_byte_us"] - r["start_us"]) / 1e3
                r["headers_ms"] = (r["headers_us"] - r["start_us"]) / 1e3
                r["body_ms"] = (r["end_us"] - r["first_byte_us"]) / 1e3
                rows.append(r)
        scans_path = path.parent / "scans.json"
        if scans_path.exists():
            run = rows[-1]["run"] if rows else path.parent.name
            for s in json.loads(scans_path.read_text()):
                s["run"] = run
                scans.append(s)
        meta_path = path.parent / "meta.json"
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text()))
    return rows, scans, metas


def table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


def ms(v: float) -> str:
    return "-" if math.isnan(v) else f"{v:.1f}"


def fresh_section(fresh: list[dict], warm: list[dict], fresh_runs: list[str], ages: dict[str, int]) -> list[str]:
    md = ["", "## Freshly uploaded objects", ""]
    md.append(
        "Excluded from every other section: "
        + ", ".join(f"{run} (dataset {ages[run]}s old)" for run in fresh_runs)
        + ". Serial read p50 ms:"
    )
    body = []
    for b in (1, 1 << 10, 64 << 10, 1 << 20, 8 << 20):
        cells = [fmt_bytes(b)]
        for rows in (fresh, warm):
            vals = [r["total_ms"] for r in rows if r["phase"] == "size_sweep" and bucket(r["length"]) == b]
            cells.append(ms(quantile(vals, 0.5)))
        body.append(cells)
    md += ["", table(["size >=", "fresh runs p50", "other runs p50"], body), ""]
    md += ["GET latency in the fresh runs as the dataset ages:", ""]
    small = [r for r in fresh if r["kind"] != "head" and r["length"] <= 64 << 10]
    windows: dict[int, list[float]] = defaultdict(list)
    for r in small:
        age = ages[r["run"]] + r["start_us"] / 1e6
        windows[int(age // 20)].append(r["total_ms"])
    md.append(
        table(
            ["object age s", "n", "p50 ms (<= 64 KiB GETs)"],
            [[f"{w * 20}-{w * 20 + 20}", str(len(v)), ms(quantile(v, 0.5))] for w, v in sorted(windows.items())],
        )
    )
    return md


def inverse_cdf(values: list[float]) -> list[list[float]]:
    return [[q, quantile(values, q)] for q in CDF_POINTS]


def sample(cdf: list[list[float]], rng: random.Random) -> float:
    """Draw from an inverse CDF stored as `[[quantile, value], ...]`, interpolating linearly."""
    u = rng.random()
    for (q0, v0), (q1, v1) in zip(cdf, cdf[1:]):
        if u <= q1:
            return v0 + (v1 - v0) * (u - q0) / ((q1 - q0) or 1.0)
    return cdf[-1][1]


def fit_request_model(ok: list[dict], fresh: list[dict], ages: dict[str, int]) -> dict:
    """Fit the per-request model from the microbenchmark phases only."""
    serial = [r for r in ok if r["phase"] == "size_sweep"]
    # Requests of 1-3 bytes are measurably slower; keep them out of the base distribution.
    ttfb = [r["ttfb_ms"] / 1e3 for r in serial if r["length"] >= 4]
    heads = [r["total_ms"] / 1e3 for r in ok if r["phase"] == "suffix" and r["kind"] == "head"]
    big = [r for r in serial if r["length"] >= 4 * MIB and r["body_ms"] > 0]
    if big:
        per_connection = quantile([r["bytes"] / (r["body_ms"] / 1e3) for r in big], 0.5)
    else:
        # Stores that deliver the whole body at once (like `ThrottledStore`) leave no body time
        # to measure; fall back to the slope of serial latency against size.
        (_, ms_per_mib), _ = least_squares([[1.0, r["length"] / MIB] for r in serial], [r["total_ms"] for r in serial])
        per_connection = MIB / (ms_per_mib / 1e3)

    # The instance's network cap: the best aggregate throughput of 8 MiB reads in each run, then
    # the median over runs.
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in ok:
        if r["phase"] == "concurrency" and r["length"] >= 4 * MIB:
            cells[(r["run"], r["concurrency"])].append(r)
    best: dict[str, float] = defaultdict(float)
    for (run, _), rs in cells.items():
        span = (max(r["end_us"] for r in rs) - min(r["start_us"] for r in rs)) / 1e6
        if span > 0:
            best[run] = max(best[run], sum(r["bytes"] for r in rs) / span)
    # Aggregate throughput includes first-byte waits, so a run that never saturates the network
    # underestimates the cap; it is never below what one connection achieved.
    nic = max(quantile(list(best.values()), 0.5), per_connection) if best else per_connection

    base = quantile(ttfb, 0.5)
    small_extra = {}
    for n in (1, 2, 3):
        vals = [r["ttfb_ms"] / 1e3 for r in serial if r["length"] == n]
        if vals:
            small_extra[str(n)] = max(0.0, quantile(vals, 0.5) - base)

    model = {
        "formula": "latency = first_byte + bytes / min(per_connection_bytes_per_s, nic_bytes_per_s / active_bodies)",
        "first_byte_s": {"inverse_cdf": inverse_cdf(ttfb), "n": len(ttfb)},
        "head_s": {"inverse_cdf": inverse_cdf(heads), "n": len(heads)},
        "per_connection_bytes_per_s": per_connection,
        "nic_bytes_per_s": nic,
        "small_read_extra_first_byte_s": small_extra,
    }
    fresh_object = fit_fresh_object(fresh, ages, base)
    if fresh_object:
        model["fresh_object"] = fresh_object
    return model


def fit_fresh_object(fresh: list[dict], ages: dict[str, int], base: float) -> dict | None:
    """Fit how much slower objects are right after upload, and for how long.

    The slowdown ends abruptly, so the end is found from the data: the first 10 s window of small
    GETs whose median is back within 1.5x of the warm first-byte median.
    """
    small = sorted(
        (ages[r["run"]] + r["start_us"] / 1e6, r["ttfb_ms"] / 1e3)
        for r in fresh
        if r["kind"] != "head" and r["length"] <= 64 << 10
    )
    windows: dict[int, list[float]] = defaultdict(list)
    for age, ttfb in small:
        windows[int(age // 10)].append(ttfb)
    warm_after = next((w * 10 for w, v in sorted(windows.items()) if quantile(v, 0.5) < 1.5 * base), None)
    if warm_after is None:
        return None
    slow = [ttfb for age, ttfb in small if age < warm_after]
    if not slow:
        return None
    return {
        "warm_after_age_s": warm_after,
        "first_byte_multiplier_p50": quantile(slow, 0.5) / base,
        "n": len(slow),
    }


# Requests within a file are issued stage by stage, as vortex-file does: open the file, then read
# the filter column, then the projection. Stages after the footer use the per-file concurrency.
STAGES = ("head", "footer", "footer_rest", "zonemap", "filter", "segment", "project")
SERIAL_STAGES = ("head", "footer", "footer_rest", "zonemap")


def scan_tasks(reqs: list[dict]) -> list[list[tuple[int, list[tuple[str, int]]]]]:
    """Rebuild one scan's per-file tasks (stages of `(kind, bytes)` requests) from its records."""
    reqs = sorted(reqs, key=lambda r: r["seq"])
    per_file_concurrency = reqs[0]["concurrency"]
    by_file: dict[int, list[dict]] = defaultdict(list)
    for r in reqs:
        by_file[r["file"]].append(r)
    tasks = []
    for frs in by_file.values():
        # Point lookups open the same file several times; each HEAD starts a new open.
        opens: list[list[dict]] = [[]]
        for r in frs:
            if r["kind"] == "head" and opens[-1]:
                opens.append([])
            opens[-1].append(r)
        for rs in opens:
            stages = []
            for kind in STAGES:
                batch = [(r["kind"], 0 if kind == "head" else r["length"]) for r in rs if r["kind"] == kind]
                if batch:
                    stages.append((1 if kind in SERIAL_STAGES else per_file_concurrency, batch))
            tasks.append(stages)
    return tasks


def simulate(
    model: dict,
    scans: list[list[list[tuple[int, list[tuple[str, int]]]]]],
    file_concurrency: list[int],
    delays: list[float],
    rng: random.Random,
) -> tuple[list[float], list[tuple[str, float]]]:
    """Discrete-event simulation of concurrent scans under the request model.

    Returns each scan's wall time and every request's `(kind, latency)`.
    """
    per_conn = model["per_connection_bytes_per_s"]
    nic = model["nic_bytes_per_s"]
    if not (math.isfinite(per_conn) and math.isfinite(nic) and per_conn > 0 and nic > 0):
        raise ValueError(f"model transfer rates must be positive, got {per_conn} and {nic}")
    t = 0.0
    timers: list[tuple[float, int]] = []
    bodies: dict[int, float] = {}
    reqs: dict[int, tuple[int, int, float, str, int]] = {}
    next_id = 0
    pending = [list(range(len(tasks))) for tasks in scans]
    active = [0] * len(scans)
    state: dict[tuple[int, int], list] = {}
    started = [False] * len(scans)
    done: list[float | None] = [None] * len(scans)
    latencies: list[tuple[str, float]] = []

    def issue(scan: int, task: int) -> None:
        nonlocal next_id
        stage, queue, inflight = state[(scan, task)]
        limit = scans[scan][task][stage][0]
        while queue and inflight < limit:
            kind, n = queue.pop(0)
            inflight += 1
            cdf = model["head_s"] if kind == "head" else model["first_byte_s"]
            next_id += 1
            reqs[next_id] = (scan, task, t, kind, n)
            heapq.heappush(timers, (t + sample(cdf["inverse_cdf"], rng), next_id))
        state[(scan, task)][1:] = [queue, inflight]

    def open_files(scan: int) -> None:
        while pending[scan] and active[scan] < file_concurrency[scan]:
            task = pending[scan].pop(0)
            active[scan] += 1
            state[(scan, task)] = [0, list(scans[scan][task][0][1]), 0]
            issue(scan, task)

    def finish(i: int) -> None:
        scan, task, start, kind, _ = reqs.pop(i)
        latencies.append((kind, t - start))
        s = state[(scan, task)]
        s[2] -= 1
        if s[1]:
            issue(scan, task)
        elif s[2] == 0:
            s[0] += 1
            if s[0] < len(scans[scan][task]):
                s[1] = list(scans[scan][task][s[0]][1])
                issue(scan, task)
            else:
                active[scan] -= 1
                open_files(scan)
                if not pending[scan] and active[scan] == 0:
                    done[scan] = t

    while any(d is None for d in done):
        for s in range(len(scans)):
            if not started[s] and delays[s] <= t + 1e-12:
                started[s] = True
                open_files(s)
                if not scans[s]:
                    done[s] = t
        rate = min(per_conn, nic / len(bodies)) if bodies else 0.0
        first = min(bodies, key=bodies.__getitem__) if bodies else None
        next_body = t + bodies[first] / rate if first is not None else math.inf
        next_timer = timers[0][0] if timers else math.inf
        next_start = min((delays[s] for s in range(len(scans)) if not started[s]), default=math.inf)
        now = min(next_body, next_timer, next_start)
        if now == math.inf:
            break
        for i in bodies:
            bodies[i] -= rate * (now - t)
        t = now
        if first is not None and now == next_body:
            # Finish the body this step waited for even if rounding left a sliver of it, or a
            # step too small to advance `t` would repeat forever.
            bodies[first] = 0.0
        for i in [i for i, left in bodies.items() if left <= 1e-6]:
            del bodies[i]
            finish(i)
        while timers and timers[0][0] <= t + 1e-12:
            _, i = heapq.heappop(timers)
            if reqs[i][4] > 0:
                bodies[i] = reqs[i][4]
            else:
                finish(i)
    return [(d if d is not None else t) - delay for d, delay in zip(done, delays)], latencies


def validate(model: dict, ok: list[dict], scans: list[dict]) -> dict:
    """Replay every measured scan through the model and compare wall times."""
    rng = random.Random(0)
    by_scan: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in ok:
        if r["phase"] == "scan":
            by_scan[(r["run"], r["scan_id"])].append(r)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in scans:
        groups[(s["run"], s["scenario"])].append(s)
    measured: dict[str, list[float]] = defaultdict(list)
    simulated: dict[str, list[float]] = defaultdict(list)
    seg_sim: dict[str, list[float]] = defaultdict(list)
    for (run, scenario), ss in sorted(groups.items()):
        ss.sort(key=lambda s: s["scan_id"])
        start = min(s["start_us"] for s in ss)
        tasks = [scan_tasks(by_scan[(run, s["scan_id"])]) for s in ss if by_scan[(run, s["scan_id"])]]
        if len(tasks) != len(ss):
            continue
        delays = [(s["start_us"] - start) / 1e6 for s in ss]
        walls, lat = simulate(model, tasks, [s["file_concurrency"] for s in ss], delays, rng)
        for s, w in zip(ss, walls):
            measured[scenario].append((s["end_us"] - s["start_us"]) / 1e6)
            simulated[scenario].append(w)
        seg_sim[scenario] += [x for k, x in lat if k in ("segment", "filter", "project")]
    seg_meas: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        if r["phase"] == "scan" and r["kind"] in ("segment", "filter", "project"):
            seg_meas[r["scenario"]].append(r["total_ms"] / 1e3)
    out = {}
    for scenario in measured:
        m, p = quantile(measured[scenario], 0.5), quantile(simulated[scenario], 0.5)
        out[scenario] = {
            "measured_wall_s_p50": m,
            "simulated_wall_s_p50": p,
            "relative_error": (p - m) / m,
            "measured_segment_s": {f"p{int(q * 100)}": quantile(seg_meas[scenario], q) for q in QUANTILES},
            "simulated_segment_s": {f"p{int(q * 100)}": quantile(seg_sim[scenario], q) for q in QUANTILES},
        }
    return out


def model_section(request_model: dict, validation: dict) -> list[str]:
    fb = request_model["first_byte_s"]["inverse_cdf"]
    at = {q: v for q, v in fb}
    md = ["", "## Latency model", ""]
    md.append(f"`{request_model['formula']}`, fitted from the microbenchmarks only:")
    md.append("")
    md.append(
        f"- first byte: p50 {at[0.5] * 1e3:.1f} ms, p90 {at[0.9] * 1e3:.1f} ms, p99 {at[0.99] * 1e3:.1f} ms "
        f"(inverse CDF in `model.json`)"
    )
    md.append(f"- per connection: {request_model['per_connection_bytes_per_s'] / MIB:.0f} MiB/s")
    md.append(f"- instance network cap: {request_model['nic_bytes_per_s'] / MIB:.0f} MiB/s")
    extra = request_model["small_read_extra_first_byte_s"]
    if extra:
        md.append(
            "- extra first-byte latency for tiny reads: "
            + ", ".join(f"{n} B +{v * 1e3:.1f} ms" for n, v in extra.items())
        )
    if "fresh_object" in request_model:
        fresh = request_model["fresh_object"]
        md.append(
            f"- objects younger than {fresh['warm_after_age_s']} s: "
            f"first byte x{fresh['first_byte_multiplier_p50']:.1f}"
        )
    if validation:
        md += ["", "Scans replayed through the model against their measured wall time:", ""]
        body = [
            [
                name,
                f"{v['measured_wall_s_p50']:.2f}",
                f"{v['simulated_wall_s_p50']:.2f}",
                f"{v['relative_error']:+.0%}",
                f"{v['measured_segment_s']['p50'] * 1e3:.0f} / {v['measured_segment_s']['p99'] * 1e3:.0f}",
                f"{v['simulated_segment_s']['p50'] * 1e3:.0f} / {v['simulated_segment_s']['p99'] * 1e3:.0f}",
            ]
            for name, v in validation.items()
        ]
        md.append(
            table(
                ["scenario", "measured s", "simulated s", "error", "segment p50/p99 ms", "simulated p50/p99 ms"],
                body,
            )
        )
    return md


def analyze(rows: list[dict], scans: list[dict], metas: list[dict]) -> tuple[str, dict]:
    ages = {m["run_label"]: m["dataset_age_s"] for m in metas if "dataset_age_s" in m}
    all_runs = sorted({r["run"] for r in rows})
    fresh_runs = [run for run in all_runs if ages.get(run, FRESH_AGE_S) < FRESH_AGE_S]
    if len(fresh_runs) == len(all_runs):
        fresh_runs = []
    fresh = [r for r in rows if r["run"] in fresh_runs and not r["error"]]
    rows = [r for r in rows if r["run"] not in fresh_runs]
    scans = [s for s in scans if s["run"] not in fresh_runs]
    metas = [m for m in metas if m.get("run_label") not in fresh_runs]

    ok = [r for r in rows if not r["error"]]
    errors = [r for r in rows if r["error"]]
    runs = sorted({r["run"] for r in rows})
    model: dict = {
        "runs": runs,
        "fresh_runs_excluded": fresh_runs,
        "dataset_age_s": ages,
        "requests": len(rows),
        "errors": len(errors),
    }
    md = ["# S3 latency report", ""]
    md.append(
        f"{len(runs)} runs, {len(rows):,} requests, {len(errors)} errors, "
        f"{sum(r['bytes'] for r in ok) / (1 << 30):.1f} GiB read."
    )
    instance_types = sorted({m.get("meta", {}).get("instance_type", "?") for m in metas})
    md.append(f"Instance types: {', '.join(instance_types)}.")
    if errors:
        kinds = defaultdict(int)
        for r in errors:
            kinds[r["error"][:120]] += 1
        md += ["", "### Errors", ""]
        md += [f"- {n} x `{e}`" for e, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:10]]

    # Serial size sweep: the base latency-vs-bytes curve.
    serial = [r for r in ok if r["phase"] == "size_sweep"]
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for r in serial:
        by_bucket[bucket(r["length"])].append(r)
    body = []
    fit_points: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for b in sorted(by_bucket):
        rs = by_bucket[b]
        total = [r["total_ms"] for r in rs]
        ttfb = [r["ttfb_ms"] for r in rs]
        mean_len = sum(r["length"] for r in rs) / len(rs)
        tput = [r["bytes"] / MIB / (r["total_ms"] / 1e3) for r in rs if r["total_ms"] > 0]
        for q in QUANTILES:
            fit_points[q].append((mean_len, quantile(total, q)))
        body.append(
            [
                fmt_bytes(b),
                str(len(rs)),
                ms(quantile(ttfb, 0.5)),
                ms(quantile(ttfb, 0.99)),
                ms(quantile(total, 0.5)),
                ms(quantile(total, 0.9)),
                ms(quantile(total, 0.99)),
                ms(max(total)),
                f"{quantile(tput, 0.5):.1f}",
            ]
        )
    md += ["", "## Serial reads (concurrency 1)", ""]
    md.append(
        table(
            [
                "size >=",
                "n",
                "TTFB p50",
                "TTFB p99",
                "p50 ms",
                "p90 ms",
                "p99 ms",
                "max ms",
                "MiB/s p50",
            ],
            body,
        )
    )

    serial_model = {}
    md += ["", "### Fitted serial model: `latency_ms = a + MiB / bandwidth`", ""]
    fit_body = []
    for q in QUANTILES:
        pts = fit_points[q]
        if len(pts) < 2:
            continue
        (a, slope), r2 = least_squares([[1.0, x / MIB] for x, _ in pts], [y for _, y in pts])
        bw = 1e3 / slope if slope > 0 else float("inf")
        serial_model[f"p{int(q * 100)}"] = {"a_ms": a, "ms_per_mib": slope, "mib_per_s": bw, "r2": r2}
        fit_body.append([f"p{int(q * 100)}", f"{a:.2f}", f"{slope:.2f}", f"{bw:.0f}", f"{r2:.3f}"])
    if serial:
        (a, slope), r2 = least_squares([[1.0, r["length"] / MIB] for r in serial], [r["total_ms"] for r in serial])
        serial_model["mean_ols"] = {"a_ms": a, "ms_per_mib": slope, "mib_per_s": 1e3 / slope, "r2": r2}
        fit_body.append(["mean (OLS, all points)", f"{a:.2f}", f"{slope:.2f}", f"{1e3 / slope:.0f}", f"{r2:.3f}"])
    md.append(table(["quantile", "a (ms)", "ms per MiB", "MiB/s per stream", "R^2"], fit_body))
    model["serial"] = serial_model

    # Run-to-run variation of the serial curve.
    md += ["", "### Run-to-run variation (serial p50 ms)", ""]
    probe = [1 << 10, 64 << 10, 1 << 20, 8 << 20]
    var_body = []
    for run in runs:
        cells = [run]
        for p in probe:
            vals = [r["total_ms"] for r in serial if r["run"] == run and bucket(r["length"]) == p]
            cells.append(ms(quantile(vals, 0.5)))
        var_body.append(cells)
    md.append(table(["run"] + [fmt_bytes(p) for p in probe], var_body))

    # Cold start vs warm.
    cold = [r for r in ok if r["phase"] == "cold"]
    if cold:
        first = quantile([r["total_ms"] for r in cold if r["seq"] == 0], 0.5)
        heads = quantile([r["total_ms"] for r in cold if r["kind"] == "head" and r["seq"] > 0], 0.5)
        footers = quantile([r["total_ms"] for r in cold if r["kind"] == "footer"], 0.5)
        md += ["", "## Cold start", ""]
        md.append(
            f"First request of each run (DNS, TCP, TLS): p50 {ms(first)} ms. "
            f"Later cold-phase HEAD p50 {ms(heads)} ms, footer (64 KiB) p50 {ms(footers)} ms."
        )
        model["cold_first_request_ms_p50"] = first

    # Suffix vs bounded tail reads, and HEAD.
    suffix = [r for r in ok if r["phase"] == "suffix"]
    if suffix:
        md += ["", "## Suffix (`bytes=-N`) vs bounded tail reads", ""]
        groups: dict[tuple[int, str], list[float]] = defaultdict(list)
        for r in suffix:
            groups[(r["useful_bytes"], r["kind"])].append(r["total_ms"])
        body = []
        for length in sorted({k[0] for k in groups if k[1] != "head"}):
            s, t = groups.get((length, "suffix"), []), groups.get((length, "tail_bounded"), [])
            body.append(
                [
                    fmt_bytes(length),
                    ms(quantile(s, 0.5)),
                    ms(quantile(t, 0.5)),
                    ms(quantile(s, 0.99)),
                    ms(quantile(t, 0.99)),
                ]
            )
        md.append(table(["length", "suffix p50", "bounded p50", "suffix p99", "bounded p99"], body))
        heads = groups.get((0, "head"), [])
        md.append("")
        md.append(
            f"HEAD: p50 {ms(quantile(heads, 0.5))} ms, p90 {ms(quantile(heads, 0.9))} ms, "
            f"p99 {ms(quantile(heads, 0.99))} ms."
        )
        model["head_ms"] = {f"p{int(q * 100)}": quantile(heads, q) for q in QUANTILES}

    # Concurrency sweep.
    conc = [r for r in ok if r["phase"] == "concurrency"]
    if conc:
        md += ["", "## Latency under concurrency", ""]
        cells: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
        for r in conc:
            cells[(r["length"], r["concurrency"], r["run"])].append(r)
        agg: dict[tuple[int, int], dict] = defaultdict(lambda: {"lat": [], "tput": [], "rps": []})
        for (length, c, _run), rs in cells.items():
            span = (max(r["end_us"] for r in rs) - min(r["start_us"] for r in rs)) / 1e6
            a = agg[(length, c)]
            a["lat"] += [r["total_ms"] for r in rs]
            if span > 0:
                a["tput"].append(sum(r["bytes"] for r in rs) / MIB / span)
                a["rps"].append(len(rs) / span)
        body = []
        conc_model = defaultdict(dict)
        for length, c in sorted(agg):
            a = agg[(length, c)]
            q = {f"p{int(x * 100)}": quantile(a["lat"], x) for x in QUANTILES}
            conc_model[str(length)][str(c)] = {
                **q,
                "agg_mib_s": quantile(a["tput"], 0.5),
                "req_s": quantile(a["rps"], 0.5),
            }
            body.append(
                [
                    fmt_bytes(length),
                    str(c),
                    str(len(a["lat"])),
                    ms(q["p50"]),
                    ms(q["p90"]),
                    ms(q["p99"]),
                    f"{quantile(a['tput'], 0.5):.0f}",
                    f"{quantile(a['rps'], 0.5):.0f}",
                ]
            )
        md.append(table(["size", "concurrency", "n", "p50 ms", "p90 ms", "p99 ms", "agg MiB/s", "req/s"], body))
        model["concurrency"] = conc_model

    # Multivariate fit over every GET.
    gets = [r for r in ok if r["kind"] != "head" and r["bytes"] > 0]
    if gets:
        xs = [[1.0, r["bytes"] / MIB, r["inflight_at_start"], r["bytes"] / MIB * r["inflight_at_start"]] for r in gets]
        coef, r2 = least_squares(xs, [r["total_ms"] for r in gets])
        model["multivariate_ols"] = {
            "terms": ["intercept_ms", "ms_per_mib", "ms_per_inflight", "ms_per_mib_per_inflight"],
            "coef": coef,
            "r2": r2,
            "n": len(gets),
        }
        md += ["", "## Multivariate fit over all GETs", ""]
        md.append(
            f"`latency_ms = {coef[0]:.2f} + {coef[1]:.2f}*MiB + {coef[2]:.4f}*inflight + {coef[3]:.4f}*MiB*inflight` "
            f"(R^2 {r2:.3f}, n {len(gets):,})"
        )

    # Scan simulations.
    if scans:
        md += ["", "## Scan simulation", ""]
        scan_reqs: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in ok:
            if r["phase"] == "scan":
                scan_reqs[(r["run"], r["scenario"])].append(r)
        by_scenario: dict[str, list[dict]] = defaultdict(list)
        for s in scans:
            by_scenario[s["scenario"]].append(s)
        body = []
        scan_model = {}
        for name in by_scenario:
            ss = by_scenario[name]
            secs = [(s["end_us"] - s["start_us"]) / 1e6 for s in ss]
            mibs = [s["bytes"] / MIB / max(t, 1e-9) for s, t in zip(ss, secs)]
            reqs = [r for (run, sc), rs in scan_reqs.items() if sc == name for r in rs]
            segs = [r for r in reqs if r["kind"] in ("segment", "filter", "project")]
            seg_lat = [r["total_ms"] for r in segs]
            useful = sum(r["useful_bytes"] for r in segs)
            read = sum(r["bytes"] for r in segs) or 1
            per_run_reqs = len(reqs) / max(1, len({s["run"] for s in ss}))
            scan_model[name] = {
                "wall_s_p50": quantile(secs, 0.5),
                "mib_s_p50": quantile(mibs, 0.5),
                "requests_per_run": per_run_reqs,
                "segments_per_request": sum(r["segments"] for r in segs) / max(1, len(segs)),
                "useful_fraction": useful / read,
                "segment_latency_ms": {f"p{int(q * 100)}": quantile(seg_lat, q) for q in QUANTILES},
            }
            body.append(
                [
                    name,
                    str(ss[0]["files"]),
                    f"{quantile(secs, 0.5):.2f}",
                    f"{min(secs):.2f}-{max(secs):.2f}",
                    f"{quantile(mibs, 0.5):.0f}",
                    f"{per_run_reqs:.0f}",
                    f"{scan_model[name]['segments_per_request']:.1f}",
                    f"{useful / read:.0%}",
                    ms(quantile(seg_lat, 0.5)),
                    ms(quantile(seg_lat, 0.99)),
                ]
            )
        md.append(
            table(
                [
                    "scenario",
                    "files",
                    "wall s p50",
                    "wall s range",
                    "MiB/s p50",
                    "requests/run",
                    "segments/req",
                    "useful bytes",
                    "seg p50 ms",
                    "seg p99 ms",
                ],
                body,
            )
        )
        model["scans"] = scan_model

    if fresh_runs:
        md += fresh_section(fresh, ok, fresh_runs, ages)
        model["fresh_serial_p50_ms"] = {
            fmt_bytes(b): quantile(
                [r["total_ms"] for r in fresh if r["phase"] == "size_sweep" and bucket(r["length"]) == b], 0.5
            )
            for b in (1, 1 << 10, 64 << 10, 1 << 20, 8 << 20)
        }

    request_model = fit_request_model(ok, fresh, ages)
    validation = validate(request_model, ok, scans) if scans else {}
    md += model_section(request_model, validation)
    md += ["", "Raw per-request CSVs and the latency model (`model.json`) are in the workflow artifacts."]

    environment = {
        key: model.pop(key) for key in ("runs", "fresh_runs_excluded", "dataset_age_s", "requests", "errors")
    }
    environment["instance_types"] = instance_types
    environment["store"] = ",".join(sorted({m.get("store", "real") for m in metas})) or "real"
    environment["availability_zones"] = sorted({m.get("meta", {}).get("availability_zone", "?") for m in metas})
    fits = {key: model.pop(key) for key in ("serial", "multivariate_ols") if key in model}
    return "\n".join(md) + "\n", {
        "schema": SCHEMA,
        "environment": environment,
        "request_model": request_model,
        "validation": validation,
        "fits": fits,
        "measurements": model,
    }


def compare(real: dict, sim: dict) -> tuple[list[str], dict]:
    """Put a simulated run's measurements next to the real ones they should reproduce."""
    rows: list[tuple[str, str, float, float]] = []

    def add(group: str, name: str, r: float | None, s: float | None) -> None:
        if r is not None and s is not None and not math.isnan(r) and not math.isnan(s):
            rows.append((group, name, r, s))

    for q in ("p50", "p90", "p99"):
        r, s = real["fits"].get("serial", {}).get(q), sim["fits"].get("serial", {}).get(q)
        if r and s:
            add("serial", f"{q} first byte ms", r["a_ms"], s["a_ms"])
            add("serial", f"{q} MiB/s per connection", r["mib_per_s"], s["mib_per_s"])
    for q in ("p50", "p99"):
        add(
            "HEAD",
            f"{q} ms",
            real["measurements"].get("head_ms", {}).get(q),
            sim["measurements"].get("head_ms", {}).get(q),
        )
    real_conc, sim_conc = real["measurements"].get("concurrency", {}), sim["measurements"].get("concurrency", {})
    for length, cells in real_conc.items():
        for c, cell in cells.items():
            other = sim_conc.get(length, {}).get(c)
            if other:
                name = f"{fmt_bytes(int(length))} x{c}"
                add("concurrency p50 ms", name, cell["p50"], other["p50"])
                add("concurrency MiB/s", name, cell["agg_mib_s"], other["agg_mib_s"])
    real_scans, sim_scans = real["measurements"].get("scans", {}), sim["measurements"].get("scans", {})
    for name, cell in real_scans.items():
        other = sim_scans.get(name)
        if other:
            add("scan wall s", name, cell["wall_s_p50"], other["wall_s_p50"])
            for q in ("p50", "p99"):
                add(f"scan segment {q} ms", name, cell["segment_latency_ms"][q], other["segment_latency_ms"][q])

    def error(r: float, s: float) -> float:
        return (s - r) / r if r else math.nan

    groups: dict[str, list[float]] = defaultdict(list)
    for group, _, r, s in rows:
        groups[group].append(abs(error(r, s)))
    md = ["", "## Simulated vs real", ""]
    md.append(f"Simulated store: `{sim['environment'].get('store', '?')}`. Median absolute error by group:")
    md.append("")
    md.append(
        table(
            ["group", "n", "median abs error", "worst"],
            [[g, str(len(v)), f"{quantile(v, 0.5):.0%}", f"{max(v):.0%}"] for g, v in groups.items()],
        )
    )
    md += ["", "<details><summary>Every compared value</summary>", ""]
    md.append(
        table(
            ["group", "value", "real", "simulated", "error"],
            [[g, n, f"{r:.2f}", f"{s:.2f}", f"{error(r, s):+.0%}"] for g, n, r, s in rows],
        )
    )
    md += ["", "</details>"]
    summary = {
        g: {"n": len(v), "median_abs_error": quantile(v, 0.5), "max_abs_error": max(v)} for g, v in groups.items()
    }
    return md, {"store": sim["environment"].get("store"), "groups": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--compare",
        type=Path,
        help="model.json of real S3 runs; adds a section comparing these (simulated) runs with them",
    )
    args = parser.parse_args()

    rows, scans, metas = load(args.results)
    if not rows:
        print(f"no requests.csv found under {args.results}", file=sys.stderr)
        return 1
    report, model = analyze(rows, scans, metas)
    if args.compare:
        section, comparison = compare(json.loads(args.compare.read_text()), model)
        report += "\n".join(section) + "\n"
        model["comparison"] = comparison
    print(report)
    if args.report:
        args.report.write_text(report)
    if args.model:
        args.model.write_text(json.dumps(model, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
