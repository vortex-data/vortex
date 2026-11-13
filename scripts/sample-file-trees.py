#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Dump `vx` file stats and layout trees for a sample of a benchmark's Vortex files.

A benchmark dataset can hold hundreds of files, and the full tree of every one is far too large
to keep. Every file is dumped first, so the sample is chosen from the real line counts rather than
an estimate. Per Vortex format directory, the files kept are:

1. One file of every table. Partitions of a table (`lineitem_0001.vortex`, `lineitem_0002.vortex`,
   `hits_17.vortex`) share a table name; one partition per table is drawn at random, weighted by
   file size in bytes. These are always kept, even past the line budget.
2. Further partitions in a random order weighted by file size in bytes, each kept if its dump
   still fits in the remaining line budget, so the larger partitions that dominate the dataset are
   more likely to be inspected.

The draw is seeded (by default with the commit), so reruns of a commit keep the same files.
`manifest.json` lists every file with its line count and whether its dump was kept.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

FORMAT_DIRS = ("vortex-file-compressed", "vortex-compact")

# Trailing partition index: `lineitem_0001`, `hits_17`, `part-3`.
PARTITION_SUFFIX = re.compile(r"[_-]\d+$")


@dataclass(frozen=True)
class VortexFile:
    path: Path
    size: int
    lines: int = 0

    @property
    def table(self) -> str:
        return table_name(self.path)


def table_name(path: Path) -> str:
    """The table a file belongs to: its stem without a trailing partition index."""
    return PARTITION_SUFFIX.sub("", path.stem)


def weighted_order(files: list[VortexFile], rng: random.Random) -> list[VortexFile]:
    """A random order of `files` in which each next file is drawn with probability proportional
    to its size, i.e. a weighted sample without replacement of all of them.

    Uses Efraimidis-Spirakis keys `u ** (1 / w)`, compared as `log(u) / w` so that weights of
    many gigabytes keep their precision. Zero-byte files get a minimal weight so they still come
    last rather than never.
    """
    keyed = [(math.log(1.0 - rng.random()) / max(f.size, 1), f) for f in files]
    keyed.sort(key=lambda kf: kf[0], reverse=True)
    return [f for _, f in keyed]


def sample_files(files: list[VortexFile], max_lines: int, rng: random.Random) -> list[VortexFile]:
    """One file per table, then size-weighted further files while they fit in `max_lines`."""
    by_table: dict[str, list[VortexFile]] = {}
    for f in sorted(files, key=lambda f: f.path):
        by_table.setdefault(f.table, []).append(f)

    chosen = [weighted_order(by_table[table], rng)[0] for table in sorted(by_table)]
    budget = max_lines - sum(f.lines for f in chosen)

    chosen_paths = {f.path for f in chosen}
    remaining = [f for f in sorted(files, key=lambda f: f.path) if f.path not in chosen_paths]
    for f in weighted_order(remaining, rng):
        if f.lines <= budget:
            chosen.append(f)
            budget -= f.lines
    return sorted(chosen, key=lambda f: f.path)


def run_vx(vx: str, args: list[str]) -> str:
    result = subprocess.run([vx, *args], capture_output=True, text=True)
    if result.returncode != 0:
        return f"`vx {' '.join(args)}` failed with exit code {result.returncode}:\n{result.stderr}"
    return result.stdout


def dump_file(vx: str, file: Path, output: Path) -> int:
    """Write the stats and trees of one file to `output`, returning the number of lines."""
    sections = [
        ("vx inspect", ["inspect", str(file)]),
        ("vx tree layout --verbose", ["tree", "layout", "--verbose", str(file)]),
    ]
    text = "".join(f"===== {title} {file} =====\n{run_vx(vx, args)}\n" for title, args in sections)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return text.count("\n")


def find_format_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(d for d in dataset_dir.rglob("*") if d.is_dir() and d.name in FORMAT_DIRS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Benchmark data directory, e.g. vortex-bench/data/tpch")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Directory to write the kept dumps to")
    parser.add_argument("--vx", default="vx", help="Path to the vx binary")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=20_000,
        help="Line budget per format directory; one file per table is kept even past it (default: 20000)",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Files to dump in parallel")
    parser.add_argument("--seed", default="0", help="Seed for the draw, e.g. the commit SHA")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = []
    with tempfile.TemporaryDirectory() as staging_dir:
        staging = Path(staging_dir)
        for format_dir in find_format_dirs(args.dataset_dir):
            paths = sorted(p for p in format_dir.rglob("*.vortex") if p.is_file())
            if not paths:
                continue

            def dump(path: Path) -> VortexFile:
                rel = path.relative_to(args.dataset_dir)
                lines = dump_file(args.vx, path, staging / rel.with_suffix(".txt"))
                return VortexFile(path, path.stat().st_size, lines)

            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                files = list(pool.map(dump, paths))

            rng = random.Random(f"{args.seed}:{format_dir.relative_to(args.dataset_dir)}")
            kept = {f.path for f in sample_files(files, args.max_lines, rng)}
            print(
                f"{format_dir}: kept {len(kept)} of {len(files)} files "
                f"({len({f.table for f in files})} tables, {sum(f.size for f in files):,} bytes), "
                f"{sum(f.lines for f in files if f.path in kept):,} of {sum(f.lines for f in files):,} lines"
            )
            for f in files:
                rel = f.path.relative_to(args.dataset_dir)
                if f.path in kept:
                    dest = args.output / rel.with_suffix(".txt")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(staging / rel.with_suffix(".txt"), dest)
                    print(f"  {rel}  {f.size:>15,} bytes  {f.lines:>8,} lines")
                manifest.append(
                    {"file": str(rel), "table": f.table, "bytes": f.size, "lines": f.lines, "kept": f.path in kept}
                )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
