#!/usr/bin/env python3
"""Produce the four per-T matrix tables in markdown format."""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/user/vortex/benchmarks/fastlanes-kernel-bench/measurements")
INP = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "matrix_run1.csv")

def main():
    table = defaultdict(dict)
    with INP.open() as fh:
        next(fh)
        for line in fh:
            t, w, simd, variant, ns = line.strip().split(",")
            try:
                table[(t, int(w))][(simd, variant)] = float(ns)
            except ValueError:
                pass
    out = []
    for T in ("u8", "u16", "u32", "u64"):
        out.append(f"\n### `{T}`\n")
        out.append("| W | sse2 bare | sse2 fused | overhead % | ymm bare | ymm fused | overhead % | zmm bare | zmm fused | overhead % |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        ws = sorted(w for (t, w) in table if t == T)
        for w in ws:
            cells = table[(T, w)]
            def g(s, v): return cells.get((s, v), float('nan'))
            def ov(s):
                b, f = g(s, "bare_unpack"), g(s, "fused_for")
                if not (b == b and f == f): return "n/a"
                return f"{(f - b) / b * 100:+.0f}%"
            out.append(f"| {w} | {g('sse2','bare_unpack'):.1f} | {g('sse2','fused_for'):.1f} | {ov('sse2')} | {g('ymm','bare_unpack'):.1f} | {g('ymm','fused_for'):.1f} | {ov('ymm')} | {g('zmm','bare_unpack'):.1f} | {g('zmm','fused_for'):.1f} | {ov('zmm')} |")
    print("\n".join(out))

if __name__ == "__main__":
    main()
