#!/usr/bin/env python3
"""Run the multi_block (N=8) bench for the chosen (T, W) cells."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/user/vortex")
BIN = ROOT / "target/release/deps/multi_block-49bbef7bc78a32b9"
OUT = ROOT / "benchmarks/fastlanes-kernel-bench/measurements/multi_block.csv"

CELLS = [
    # (T, W)
    ("u8", 1), ("u8", 3), ("u8", 5), ("u8", 8),
    ("u16", 1), ("u16", 4), ("u16", 7), ("u16", 11), ("u16", 15), ("u16", 16),
    ("u32", 1), ("u32", 5), ("u32", 8), ("u32", 10), ("u32", 17),
    ("u32", 24), ("u32", 25), ("u32", 32),
    ("u64", 1), ("u64", 8), ("u64", 11), ("u64", 33), ("u64", 55), ("u64", 64),
]
VARIANTS = ["bare_unpack_n8", "fused_for_n8"]

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


def run(t: str, w: int, variant: str) -> float | None:
    fname = f"{variant}__{t}__w{w}"
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
    out_fh.write("T,W,variant,total_8block_ns,per_block_ns\n")
    total = len(CELLS) * len(VARIANTS)
    counter = 0
    start = time.monotonic()
    for t, w in CELLS:
        for variant in VARIANTS:
            counter += 1
            ns = run(t, w, variant)
            ns_str = f"{ns:.6f}" if ns is not None else "NaN"
            per_block = f"{ns/8.0:.6f}" if ns is not None else "NaN"
            out_fh.write(f"{t},{w},{variant},{ns_str},{per_block}\n")
            elapsed = time.monotonic() - start
            rate = elapsed / counter
            eta = rate * (total - counter)
            print(f"[{counter:3d}/{total}] {t} W={w:2d} {variant} -> {ns_str} ns (per-block={per_block})  eta={eta:.0f}s", flush=True)
    out_fh.close()
    print(f"Done. Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
