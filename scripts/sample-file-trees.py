#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Dump `vx` file stats and layout trees for a sample of a benchmark's Vortex files.

A benchmark dataset can hold hundreds of files, and the full tree of every one is far too large
to keep. Per Vortex format directory, the sample is:

1. One file of every table. Partitions of a table (`lineitem_0001.vortex`, `lineitem_0002.vortex`,
   `hits_17.vortex`) share a table name; one partition per table is drawn at random, weighted by
   file size in bytes.
2. A fixed number of further files drawn from the remaining partitions, without replacement and
   weighted by file size in bytes, so the larger partitions that dominate the dataset are more
   likely to be inspected.

The draw is seeded (by default with the commit), so reruns of a commit inspect the same files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FORMAT_DIRS = ("vortex-file-compressed", "vortex-compact")

# Trailing partition index: `lineitem_0001`, `hits_17`, `part-3`.
PARTITION_SUFFIX = re.compile(r"[_-]\d+$")


@dataclass(frozen=True)
class VortexFile:
    path: Path
    size: int

    @property
    def table(self) -> str:
        return table_name(self.path)


def table_name(path: Path) -> str:
    """The table a file belongs to: its stem without a trailing partition index."""
    return PARTITION_SUFFIX.sub("", path.stem)


def weighted_sample(files: list[VortexFile], k: int, rng: random.Random) -> list[VortexFile]:
    """Draw `k` files without replacement, each with probability proportional to its size.

    Uses Efraimidis-Spirakis keys `u ** (1 / w)`, compared as `log(u) / w` so that weights of
    many gigabytes keep their precision, and keeps the `k` largest. Zero-byte files get a minimal
    weight so they can still be drawn once larger files run out.
    """
    if k <= 0:
        return []
    keyed = [(math.log(1.0 - rng.random()) / max(f.size, 1), f) for f in files]
    keyed.sort(key=lambda kf: kf[0], reverse=True)
    return [f for _, f in keyed[:k]]


def sample_files(files: list[VortexFile], extra: int, rng: random.Random) -> list[VortexFile]:
    """One file per table, plus `extra` more weighted by size from the files left over."""
    by_table: dict[str, list[VortexFile]] = {}
    for f in sorted(files, key=lambda f: f.path):
        by_table.setdefault(f.table, []).append(f)

    chosen: list[VortexFile] = []
    for table in sorted(by_table):
        chosen.extend(weighted_sample(by_table[table], 1, rng))

    chosen_paths = {f.path for f in chosen}
    remaining = [f for f in sorted(files, key=lambda f: f.path) if f.path not in chosen_paths]
    chosen.extend(weighted_sample(remaining, extra, rng))
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
    parser.add_argument("-o", "--output", type=Path, required=True, help="Directory to write the dumps to")
    parser.add_argument("--vx", default="vx", help="Path to the vx binary")
    parser.add_argument(
        "--extra",
        type=int,
        default=8,
        help="Files to draw per format directory after one file of every table (default: 8)",
    )
    parser.add_argument("--seed", default="0", help="Seed for the draw, e.g. the commit SHA")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = []
    for format_dir in find_format_dirs(args.dataset_dir):
        files = [VortexFile(p, p.stat().st_size) for p in format_dir.rglob("*.vortex") if p.is_file()]
        if not files:
            continue
        rng = random.Random(f"{args.seed}:{format_dir.relative_to(args.dataset_dir)}")
        sample = sample_files(files, args.extra, rng)
        total_bytes = sum(f.size for f in files)
        print(
            f"{format_dir}: {len(sample)} of {len(files)} files "
            f"({len({f.table for f in files})} tables, {total_bytes:,} bytes)"
        )
        for f in sample:
            rel = f.path.relative_to(args.dataset_dir)
            lines = dump_file(args.vx, f.path, args.output / rel.with_suffix(".txt"))
            print(f"  {rel}  {f.size:>15,} bytes  {lines:>8,} lines")
            manifest.append({"file": str(rel), "table": f.table, "bytes": f.size, "lines": lines})

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
