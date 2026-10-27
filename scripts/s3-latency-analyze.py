#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Aggregate s3-latency-bench runs into a Markdown report and a fitted latency model.

Usage: s3-latency-analyze.py RESULTS_DIR [--report report.md] [--model model.json]

RESULTS_DIR is searched recursively for `requests.csv`, `scans.json` and `meta.json`, one set per
run. Only the standard library is used, so this runs on any CI image.

The model fitted for serial requests is `latency_ms(bytes) = a + bytes / bandwidth`, per quantile
(fitted to per-size-bucket quantiles) and as an ordinary least-squares mean. The concurrency phase
adds how the quantiles shift with requests in flight, and a multivariate least-squares fit over
every GET: `latency_ms ~ 1 + MiB + inflight + MiB * inflight`.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

QUANTILES = (0.5, 0.9, 0.99)
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


def analyze(rows: list[dict], scans: list[dict], metas: list[dict]) -> tuple[str, dict]:
    ok = [r for r in rows if not r["error"]]
    errors = [r for r in rows if r["error"]]
    runs = sorted({r["run"] for r in rows})
    model: dict = {"runs": runs, "requests": len(rows), "errors": len(errors)}
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

    md += ["", "Raw per-request CSVs and `model.json` are in the workflow artifacts."]
    return "\n".join(md) + "\n", model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    rows, scans, metas = load(args.results)
    if not rows:
        print(f"no requests.csv found under {args.results}", file=sys.stderr)
        return 1
    report, model = analyze(rows, scans, metas)
    print(report)
    if args.report:
        args.report.write_text(report)
    if args.model:
        args.model.write_text(json.dumps(model, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
