#!/usr/bin/env python3
"""Compare matrix_run1.csv and matrix_run2.csv. Write variance.md."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/user/vortex/benchmarks/fastlanes-kernel-bench/measurements")
R1 = ROOT / "matrix_run1.csv"
R2 = ROOT / "matrix_run2.csv"
OUT = ROOT / "variance.md"


def load(path: Path) -> dict[tuple[str, int, str, str], float]:
    d = {}
    with path.open() as fh:
        next(fh)
        for line in fh:
            t, w, simd, variant, ns = line.strip().split(",")
            try:
                d[(t, int(w), simd, variant)] = float(ns)
            except ValueError:
                d[(t, int(w), simd, variant)] = float("nan")
    return d


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    r1 = load(R1)
    r2 = load(R2)
    keys = sorted(set(r1) & set(r2))
    variances = []  # (var_pct, t, w, simd, variant, ns1, ns2)
    for k in keys:
        ns1, ns2 = r1[k], r2[k]
        if ns1 != ns1 or ns2 != ns2:  # NaN
            continue
        mn = min(ns1, ns2)
        if mn <= 0:
            continue
        var = abs(ns2 - ns1) / mn * 100
        variances.append((var, *k, ns1, ns2))

    var_pcts = [v[0] for v in variances]
    noisy = [v for v in variances if v[0] > 15]
    p50 = percentile(var_pcts, 50)
    p75 = percentile(var_pcts, 75)
    p90 = percentile(var_pcts, 90)
    p99 = percentile(var_pcts, 99)
    pmax = max(var_pcts) if var_pcts else float("nan")

    lines = []
    lines.append("# Run-to-run variance (matrix_run1 vs matrix_run2)\n")
    lines.append(f"Compared **{len(variances)}** cells (full T x W x SIMD x variant grid).\n")
    lines.append("## Distribution\n")
    lines.append("| percentile | variance % |")
    lines.append("|---:|---:|")
    lines.append(f"| p50 | {p50:.2f} |")
    lines.append(f"| p75 | {p75:.2f} |")
    lines.append(f"| p90 | {p90:.2f} |")
    lines.append(f"| p99 | {p99:.2f} |")
    lines.append(f"| max | {pmax:.2f} |")
    lines.append("")
    lines.append(f"**{len(noisy)} cells (= {100*len(noisy)/len(variances):.1f}%) exceed 15% variance "
                 "and are flagged as noisy.** These cells should not be cited in conclusions.\n")
    if noisy:
        lines.append("## Noisy cells (>15% variance)\n")
        lines.append("| T | W | SIMD | variant | run1 ns | run2 ns | variance % |")
        lines.append("|---|---:|---|---|---:|---:|---:|")
        for v, t, w, simd, variant, ns1, ns2 in sorted(noisy, key=lambda x: -x[0]):
            lines.append(f"| {t} | {w} | {simd} | {variant} | {ns1:.1f} | {ns2:.1f} | {v:.1f} |")
        lines.append("")
    lines.append("## All cells\n")
    lines.append("Sorted by descending variance %. First 50 rows only; full grid is in the CSVs.\n")
    lines.append("| T | W | SIMD | variant | run1 ns | run2 ns | variance % |")
    lines.append("|---|---:|---|---|---:|---:|---:|")
    for v, t, w, simd, variant, ns1, ns2 in sorted(variances, key=lambda x: -x[0])[:50]:
        lines.append(f"| {t} | {w} | {simd} | {variant} | {ns1:.1f} | {ns2:.1f} | {v:.1f} |")
    lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
