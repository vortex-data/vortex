#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Plan the CodSpeed simulation shards.

CodSpeed rejects an upload with too many benchmarks and recommends fewer than 1,000 per upload, so
the simulation benchmarks are split into shards of at most `CAP` benchmarks each. The plan only
reads `Cargo.toml` files and `codspeed_bench_counts.json`, so it needs no Rust toolchain and runs
in a few seconds before the shard jobs start.

    python3 scripts/codspeed_shards.py plan >> "$GITHUB_OUTPUT"
    python3 scripts/codspeed_shards.py counts job-log.txt [...] > scripts/codspeed_bench_counts.json

The second form regenerates the counts from the logs of CodSpeed simulation jobs.
"""

import argparse
import collections
import json
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTS_PATH = REPO_ROOT / "scripts" / "codspeed_bench_counts.json"

# Most benchmarks per shard, leaving headroom under CodSpeed's recommended 1,000.
CAP = 800

# Shard slots in the `bench-codspeed` matrix of `.github/workflows/codspeed.yml`. The slots are fixed
# so that each one can be a required check; slots the plan does not use succeed without running.
MAX_SHARDS = 5

# Benchmarks measured for a target missing from the counts file, until the file is regenerated.
DEFAULT_COUNT = 100

# The packages whose benchmarks run in simulation, with the features each one builds with.
PACKAGES: dict[str, tuple[str, ...]] = {
    "vortex": (),
    "vortex-alp": (),
    "vortex-array": ("_test-harness",),
    "vortex-buffer": (),
    "vortex-compressor": (),
    "vortex-compute": (),
    "vortex-datetime-parts": (),
    "vortex-decimal-byte-parts": (),
    "vortex-elias-fano": (),
    "vortex-fastlanes": ("_test-harness",),
    "vortex-file": (),
    "vortex-fsst": ("_test-harness",),
    "vortex-layout": (),
    "vortex-mask": (),
    "vortex-pco": (),
    "vortex-row": (),
    "vortex-sparse": (),
    "vortex-spatial": ("_test-harness",),
    "vortex-tensor": (),
}

# vortex-array targets measured without allocator time: deferred allocator cleanup dominates them.
# CodSpeed only compares a benchmark against a base measured with the same setting, so these must
# never share a shard with other benchmarks.
EXCLUDE_ALLOCATIONS_PREFIXES = ("dict_", "chunked_dict_", "patches_", "take_", "filter_", "piecewise_sequence_")

Target = tuple[str, str]  # (package, bench target)

# One entry of the GitHub Actions matrix.
Shard = TypedDict(
    "Shard",
    {"shard": int, "benchmarks": int, "packages": str, "build-args": str, "exclude-allocations": bool},
)


def excludes_allocations(target: Target) -> bool:
    package, name = target
    return package == "vortex-array" and name.startswith(EXCLUDE_ALLOCATIONS_PREFIXES)


def discover_targets(root: Path = REPO_ROOT) -> list[Target]:
    """Return the `[[bench]]` targets of every package in `PACKAGES`."""
    members = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["members"]
    targets = []
    for member in members:
        for manifest_dir in sorted(root.glob(member)):
            manifest_path = manifest_dir / "Cargo.toml"
            if not manifest_path.is_file():
                continue
            manifest = tomllib.loads(manifest_path.read_text())
            package = manifest.get("package", {}).get("name")
            if package in PACKAGES:
                targets.extend((package, bench["name"]) for bench in manifest.get("bench", []))
    missing = set(PACKAGES) - {package for package, _ in targets}
    if missing:
        raise SystemExit(f"no benchmark targets found for {sorted(missing)}")
    return sorted(targets)


def load_counts(path: Path = COUNTS_PATH) -> dict[Target, int]:
    raw = json.loads(path.read_text())
    return {(package, name): count for package, benches in raw.items() for name, count in benches.items()}


def pack(items: list[tuple[str, list[Target], int]], cap: int) -> list[list[tuple[str, list[Target], int]]]:
    """Pack `(key, targets, count)` items into as few bins of at most `cap` as greedy allows.

    Largest item first, each into the currently lightest bin; a new bin opens only when the
    lightest one cannot take the item.
    """
    total = sum(count for _, _, count in items)
    bins: list[list[tuple[str, list[Target], int]]] = [[] for _ in range(max(1, math.ceil(total / cap)))]
    for item in sorted(items, key=lambda item: (-item[2], item[0])):
        lightest = min(bins, key=lambda b: sum(count for _, _, count in b))
        if sum(count for _, _, count in lightest) + item[2] > cap:
            lightest = []
            bins.append(lightest)
        lightest.append(item)
    return [b for b in bins if b]


def plan(counts: dict[Target, int], cap: int = CAP) -> list[Shard]:
    """Split targets into shards of at most `cap` benchmarks.

    Targets that exclude allocations and the rest are packed separately. Within each pool, a
    package's targets stay together unless the package alone exceeds `cap`, in which case its
    targets are packed one by one.
    """
    for target, count in counts.items():
        if count > cap:
            raise SystemExit(f"{target[0]}::{target[1]} alone has {count} benchmarks; split it below {cap}")

    pools: dict[bool, dict[str, list[Target]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for target in sorted(counts):
        pools[excludes_allocations(target)][target[0]].append(target)

    shards: list[Shard] = []
    for exclude in sorted(pools):
        items = []
        for package, targets in sorted(pools[exclude].items()):
            total = sum(counts[t] for t in targets)
            if total <= cap:
                items.append((package, targets, total))
            else:
                items.extend((f"{t[0]}::{t[1]}", [t], counts[t]) for t in targets)
        for shard_items in pack(items, cap):
            targets = sorted(t for _, ts, _ in shard_items for t in ts)
            packages = sorted({package for package, _ in targets})
            features = sorted(f"{package}/{feature}" for package in packages for feature in PACKAGES[package])
            args = [f"--features {','.join(features)}"] if features else []
            args += [f"-p {package}" for package in packages]
            args += [f"--bench {name}" for name in sorted({name for _, name in targets})]
            shards.append(
                {
                    "shard": len(shards) + 1,
                    "benchmarks": sum(counts[t] for t in targets),
                    "packages": " ".join(packages),
                    "build-args": " ".join(args),
                    "exclude-allocations": exclude,
                }
            )
    if len(shards) > MAX_SHARDS:
        raise SystemExit(
            f"{len(shards)} shards needed but the workflow has {MAX_SHARDS} slots; add slots to the "
            "`bench-codspeed` matrix and MAX_SHARDS together"
        )
    return shards


def github_outputs(shards: list[Shard]) -> dict[str, str]:
    """Return the `plan-codspeed` job outputs: the used slots, and each slot's shard by number."""
    by_slot = {str(shard["shard"]): shard for shard in shards}
    return {
        "active": json.dumps([shard["shard"] for shard in shards]),
        "shards": json.dumps(by_slot, separators=(",", ":")),
    }


def counts_from_logs(paths: list[Path]) -> dict[str, dict[str, int]]:
    """Count the `Measured:` lines per target in CodSpeed simulation job logs."""
    counts: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for path in paths:
        current = None
        for line in path.read_text(errors="replace").splitlines():
            if match := re.search(r"\bRunning (\S+) (\S+)$", line):
                current = (match.group(1), match.group(2))
                if current[0] in PACKAGES:
                    counts[current[0]].setdefault(current[1], 0)
                else:
                    current = None
            elif "Done running" in line:
                current = None
            elif current and "Measured: " in line:
                counts[current[0]][current[1]] += 1
    return {package: dict(sorted(benches.items())) for package, benches in sorted(counts.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print the shard plan as GITHUB_OUTPUT lines")
    counts_parser = sub.add_parser("counts", help="regenerate benchmark counts from CodSpeed job logs")
    counts_parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "counts":
        json.dump(counts_from_logs(args.logs), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    known = load_counts()
    counts = {}
    for target in discover_targets():
        if target not in known:
            print(
                f"::warning::{target[0]}::{target[1]} is missing from {COUNTS_PATH.name}; assuming "
                f"{DEFAULT_COUNT} benchmarks. Regenerate the counts with `codspeed_shards.py counts`.",
                file=sys.stderr,
            )
        counts[target] = known.get(target, DEFAULT_COUNT)
    shards = plan(counts)
    for shard in shards:
        print(f"shard {shard['shard']}: {shard['benchmarks']} benchmarks, {shard['packages']}", file=sys.stderr)
    for name, value in github_outputs(shards).items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
