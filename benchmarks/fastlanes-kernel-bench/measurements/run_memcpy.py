#!/usr/bin/env python3
"""Run the memcpy_baseline bench for every (T, W) cell and save best-of-3 median ns.

Writes memcpy_baseline.csv with columns: T,W,bytes,best_median_ns
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/user/vortex")
BIN = ROOT / "target/release/deps/memcpy_baseline-08efe142971a934d"
OUT = ROOT / "benchmarks/fastlanes-kernel-bench/measurements/memcpy_baseline.csv"
MAXW = {"u8": 8, "u16": 16, "u32": 32, "u64": 64}

LINE_RE = re.compile(
    r"^[├╰]─\s+(\S+)\s+"
    r"([\d.]+)\s+(\S+)\s+│\s+"
    r"([\d.]+)\s+(\S+)\s+│\s+"
    r"([\d.]+)\s+(\S+)\s+│\s+"
    r"([\d.]+)\s+(\S+)\s+│\s+"
    r"(\d+)\s+│\s+(\d+)\s*$"
)

UNIT_TO_NS = {"ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}


def to_ns(v: str, u: str) -> float:
    return float(v) * UNIT_TO_NS[u]


def run(t: str, w: int) -> float | None:
    fname = f"memcpy__{t}__w{w}"
    pat = fname + "$"
    best = None
    for _ in range(3):
        res = subprocess.run(
            [str(BIN), pat, "--min-time", "0.5", "--bench"],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in (res.stdout + res.stderr).splitlines():
            m = LINE_RE.match(line)
            if not m or m.group(1) != fname:
                continue
            ns = to_ns(m.group(6), m.group(7))
            if best is None or ns < best:
                best = ns
            break
    return best


def main() -> int:
    out_fh = OUT.open("w", buffering=1)
    out_fh.write("T,W,bytes,best_median_ns\n")
    total = sum(MAXW.values())
    counter = 0
    start = time.monotonic()
    for t, maxw in MAXW.items():
        bits = {"u8": 8, "u16": 16, "u32": 32, "u64": 64}[t]
        for w in range(1, maxw + 1):
            counter += 1
            byts = (1024 * w // 8) + (1024 * bits // 8)
            ns = run(t, w)
            ns_str = f"{ns:.6f}" if ns is not None else "NaN"
            out_fh.write(f"{t},{w},{byts},{ns_str}\n")
            elapsed = time.monotonic() - start
            rate = elapsed / counter
            eta = rate * (total - counter)
            print(f"[{counter:3d}/{total}] {t} W={w:2d} bytes={byts:5d} -> {ns_str} ns  (eta={eta:.0f}s)", flush=True)
    out_fh.close()
    print(f"Done. Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
