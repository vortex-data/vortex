#!/usr/bin/env python3
"""Collect best-of-3 medians for every (T, W, SIMD, variant) cell.

Writes /tmp/full_matrix.csv with columns:
    T,W,simd,variant,best_median_ns

Resumable: if /tmp/full_matrix.csv exists and RESUME=1 in the env, already-
measured rows are skipped.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/user/vortex")
BINS = {
    "sse2":  ROOT / "target/release/deps/unpack_vs_fused-ea96ed7a581f4adc",
    "ymm":   ROOT / "target/release/deps/unpack_vs_fused-16e476fea4eedc3d",
    "zmm":   ROOT / "target/release/deps/unpack_vs_fused-afe3f8c643e9a77f",
}
OUT_PATH = Path("/tmp/full_matrix.csv")
MAXW = {"u8": 8, "u16": 16, "u32": 32, "u64": 64}
VARIANTS = ["bare_unpack", "fused_for"]


# Divan output line example:
#   ├─ bare_unpack__u32__w10  63.93 ns      │ 5.547 µs      │ 64.09 ns      │ 69.16 ns      │ 192420  │ 6157440
# Columns (after the tree prefix + name): fastest, slowest, median, mean, samples, iters.
LINE_RE = re.compile(
    r"^[├╰]─\s+(\S+)\s+"
    r"([\d.]+)\s+(\S+)\s+│\s+"   # fastest val unit
    r"([\d.]+)\s+(\S+)\s+│\s+"   # slowest val unit
    r"([\d.]+)\s+(\S+)\s+│\s+"   # median  val unit
    r"([\d.]+)\s+(\S+)\s+│\s+"   # mean    val unit
    r"(\d+)\s+│\s+(\d+)\s*$"
)


UNIT_TO_NS = {
    "ns": 1.0,
    "µs": 1e3,
    "us": 1e3,
    "ms": 1e6,
    "s":  1e9,
}


def to_ns(val: str, unit: str) -> float:
    if unit not in UNIT_TO_NS:
        raise ValueError(f"unknown unit {unit!r}")
    return float(val) * UNIT_TO_NS[unit]


def run_bench(binary: Path, pattern: str) -> str:
    res = subprocess.run(
        [str(binary), pattern, "--min-time", "0.5", "--bench"],
        check=False,
        capture_output=True,
        text=True,
    )
    return res.stdout + res.stderr


def parse_median(output: str, fname: str) -> float | None:
    for line in output.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name != fname:
            continue
        med_val, med_unit = m.group(6), m.group(7)
        return to_ns(med_val, med_unit)
    return None


def measure_cell(binary: Path, t: str, w: int, variant: str) -> float | None:
    fname = f"{variant}__{t}__w{w}"
    pattern = fname + "$"
    best: float | None = None
    for _ in range(3):
        out = run_bench(binary, pattern)
        ns = parse_median(out, fname)
        if ns is None:
            sys.stderr.write(f"WARN: empty parse for {fname}\n")
            continue
        if best is None or ns < best:
            best = ns
    return best


def main() -> int:
    resume = os.environ.get("RESUME", "0") == "1"
    done: set[tuple[str, int, str, str]] = set()
    if resume and OUT_PATH.exists():
        with OUT_PATH.open() as fh:
            first = True
            for line in fh:
                if first:
                    first = False
                    continue
                parts = line.strip().split(",")
                if len(parts) != 5:
                    continue
                t, w_s, simd, variant, _ns = parts
                try:
                    done.add((t, int(w_s), simd, variant))
                except ValueError:
                    pass
        mode = "a"
    else:
        mode = "w"

    out_fh = OUT_PATH.open(mode, buffering=1)  # line-buffered
    if mode == "w":
        out_fh.write("T,W,simd,variant,best_median_ns\n")

    total = 0
    for t in ("u8", "u16", "u32", "u64"):
        total += MAXW[t] * len(BINS) * len(VARIANTS)
    counter = 0
    start = time.monotonic()

    for t in ("u8", "u16", "u32", "u64"):
        for w in range(1, MAXW[t] + 1):
            for simd, binary in BINS.items():
                for variant in VARIANTS:
                    counter += 1
                    key = (t, w, simd, variant)
                    if key in done:
                        continue
                    ns = measure_cell(binary, t, w, variant)
                    if ns is None:
                        sys.stderr.write(f"ERROR: no median for {key}\n")
                        ns_str = "NaN"
                    else:
                        ns_str = f"{ns:.6f}"
                    out_fh.write(f"{t},{w},{simd},{variant},{ns_str}\n")
                    elapsed = time.monotonic() - start
                    if counter > 0:
                        rate = elapsed / counter
                        eta = rate * (total - counter)
                    else:
                        eta = 0.0
                    print(
                        f"[{counter:4d}/{total:4d} elapsed={elapsed:6.0f}s eta={eta:6.0f}s] "
                        f"{t} W={w:2d} {simd:>4s} {variant:>11s} -> {ns_str} ns",
                        flush=True,
                    )

    out_fh.close()
    print(f"\nDone. Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
