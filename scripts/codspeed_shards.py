#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Plan the CodSpeed simulation shards.

CodSpeed rejects an upload with too many benchmarks and recommends fewer than 1,000 per upload, so
the simulation benchmarks are split into shards of at most `CAP` benchmarks each. Use the available
slots to balance recorded target runtimes, including setup inside each benchmark binary. The plan
reads manifests, benchmark counts, and durations, so it needs no Rust toolchain.

    python3 scripts/codspeed_shards.py plan >> "$GITHUB_OUTPUT"
    python3 scripts/codspeed_shards.py counts job-log.txt [...] > scripts/codspeed_bench_counts.json
    python3 scripts/codspeed_shards.py durations --source <run-url> --commit <sha> job-log.txt [...]

The latter commands regenerate counts and durations from completed simulation job logs. Save the
durations output to `scripts/codspeed_bench_durations.json`.
"""

import argparse
import collections
import datetime
import itertools
import json
import math
import re
import statistics
import sys
import tomllib
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTS_PATH = REPO_ROOT / "scripts" / "codspeed_bench_counts.json"
DURATIONS_PATH = REPO_ROOT / "scripts" / "codspeed_bench_durations.json"

# Most benchmarks per shard, leaving headroom under CodSpeed's recommended 1,000.
CAP = 800

# Shard slots in the `bench-codspeed` matrix of `.github/workflows/codspeed.yml`. The slots are fixed
# so that each one can be a required check; slots the plan does not use succeed without running.
MAX_SHARDS = 5

# Benchmarks measured for a target missing from the counts file, until the file is regenerated.
DEFAULT_COUNT = 100

# A new target starts with a five-second estimate until simulation logs record its runtime.
DEFAULT_SECONDS = 5.0

# Every workspace package with `[[bench]]` targets runs in simulation, except these. Each entry gives
# the reason, so that a new benchmark package is measured by default instead of skipped silently.
EXCLUDED_PACKAGES: dict[str, str] = {
    "vortex-btrblocks": "its benchmarks are compiled out under `cfg(codspeed)`",
    "vortex-cuda": "its benchmarks need a GPU and run in the `bench-codspeed-cuda` jobs",
    "vortex-duckdb": "its build script downloads or builds DuckDB",
}

# The features that a package builds its benchmarks with, if it needs any.
PACKAGE_FEATURES: dict[str, tuple[str, ...]] = {
    "vortex-array": ("_test-harness",),
    "vortex-fastlanes": ("_test-harness",),
    "vortex-fsst": ("_test-harness",),
    "vortex-spatial": ("_test-harness",),
}

# vortex-array targets measured without allocator time: deferred allocator cleanup dominates them.
# CodSpeed only compares a benchmark against a base measured with the same setting, so these must
# never share a shard with other benchmarks.
EXCLUDE_ALLOCATIONS_PREFIXES = ("dict_", "chunked_dict_", "patches_", "take_", "filter_", "piecewise_sequence_")

Target = tuple[str, str]  # (package, bench target)

# One entry of the GitHub Actions matrix.
Shard = TypedDict(
    "Shard",
    {
        "shard": int,
        "benchmarks": int,
        "seconds": float,
        "packages": str,
        "build-args": str,
        "exclude-allocations": bool,
    },
)


def excludes_allocations(target: Target) -> bool:
    package, name = target
    return package == "vortex-array" and name.startswith(EXCLUDE_ALLOCATIONS_PREFIXES)


def discover_targets(root: Path = REPO_ROOT) -> list[Target]:
    """Return the `[[bench]]` targets of every workspace package not in `EXCLUDED_PACKAGES`."""
    members = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["members"]
    targets = []
    for member in members:
        for manifest_dir in sorted(root.glob(member)):
            manifest_path = manifest_dir / "Cargo.toml"
            if not manifest_path.is_file():
                continue
            manifest = tomllib.loads(manifest_path.read_text())
            package = manifest.get("package", {}).get("name")
            targets.extend((package, bench["name"]) for bench in manifest.get("bench", []))

    benchmarked = {package for package, _ in targets}
    stale = (set(EXCLUDED_PACKAGES) | set(PACKAGE_FEATURES)) - benchmarked
    if stale:
        raise SystemExit(f"no benchmark targets found for {sorted(stale)}")

    excluded_with_features = set(EXCLUDED_PACKAGES) & set(PACKAGE_FEATURES)
    if excluded_with_features:
        raise SystemExit(f"excluded packages must not set features: {sorted(excluded_with_features)}")

    simulated = sorted(target for target in targets if target[0] not in EXCLUDED_PACKAGES)

    # A shard selects targets with `--bench <name>`, which Cargo matches in every selected package.
    packages_by_name = collections.defaultdict(list)
    for package, name in simulated:
        packages_by_name[name].append(package)
    shared = {name: packages for name, packages in packages_by_name.items() if len(packages) > 1}
    if shared:
        raise SystemExit(f"benchmark target names must be unique across packages: {shared}")

    return simulated


def simulated_packages() -> set[str]:
    """Return the packages whose targets `discover_targets` returns."""
    return {package for package, _ in discover_targets()}


def load_counts(path: Path = COUNTS_PATH) -> dict[Target, int]:
    raw = json.loads(path.read_text())
    return {(package, name): count for package, benches in raw.items() for name, count in benches.items()}


def load_durations(path: Path = DURATIONS_PATH) -> dict[Target, float]:
    raw = json.loads(path.read_text())["seconds"]
    return {(package, name): seconds for package, benches in raw.items() for name, seconds in benches.items()}


def plan(
    counts: dict[Target, int],
    cap: int = CAP,
    durations: dict[Target, float] | None = None,
    max_shards: int = MAX_SHARDS,
) -> list[Shard]:
    """Balance target runtimes across available slots, with at most `cap` benchmarks per shard.

    Allocation modes stay separate. Keep small packages together to limit repeated builds, but
    split a package above the per-shard runtime budget or count cap at target boundaries. Keep
    each target intact because its benchmarks share process state.
    """
    for target, count in counts.items():
        if count > cap:
            raise SystemExit(f"{target[0]}::{target[1]} alone has {count} benchmarks; split it below {cap}")

    durations = durations or {}
    seconds = {target: durations.get(target, DEFAULT_SECONDS) for target in counts}
    pools: dict[bool, list[Target]] = collections.defaultdict(list)
    for target in sorted(counts):
        pools[excludes_allocations(target)].append(target)

    slots = {exclude: max(1, math.ceil(sum(counts[t] for t in targets) / cap)) for exclude, targets in pools.items()}
    if sum(slots.values()) > max_shards:
        raise SystemExit(f"benchmark counts require more than {max_shards} shard slots")
    while sum(slots.values()) < max_shards:
        candidates = [exclude for exclude in sorted(pools) if slots[exclude] < len(pools[exclude])]
        if not candidates:
            break
        exclude = max(candidates, key=lambda mode: sum(seconds[t] for t in pools[mode]) / slots[mode])
        slots[exclude] += 1

    shards: list[Shard] = []
    for exclude in sorted(pools):
        budget = sum(seconds[t] for t in pools[exclude]) / slots[exclude]
        items = []
        for _, group in itertools.groupby(pools[exclude], key=lambda t: t[0]):
            targets = list(group)
            if sum(seconds[t] for t in targets) <= budget and sum(counts[t] for t in targets) <= cap:
                items.append(targets)
            else:
                items.extend([target] for target in targets)
        bins: list[list[Target]] = [[] for _ in range(slots[exclude])]
        for item in sorted(items, key=lambda ts: (-sum(seconds[t] for t in ts), -sum(counts[t] for t in ts), ts)):
            count = sum(counts[t] for t in item)
            candidates = [b for b in bins if sum(counts[t] for t in b) + count <= cap]
            if not candidates:
                raise SystemExit(f"cannot pack benchmarks into {max_shards} shard slots under the {cap}-benchmark cap")
            lightest = min(candidates, key=lambda b: sum(seconds[t] for t in b))
            lightest.extend(item)
        for targets in bins:
            if not targets:
                continue
            targets = sorted(targets)
            packages = sorted({package for package, _ in targets})
            features = sorted(
                f"{package}/{feature}" for package in packages for feature in PACKAGE_FEATURES.get(package, ())
            )
            args = [f"--features {','.join(features)}"] if features else []
            args += [f"-p {package}" for package in packages]
            args += [f"--bench {name}" for name in sorted({name for _, name in targets})]
            shards.append(
                {
                    "shard": len(shards) + 1,
                    "benchmarks": sum(counts[t] for t in targets),
                    "seconds": round(sum(seconds[t] for t in targets), 3),
                    "packages": " ".join(packages),
                    "build-args": " ".join(args),
                    "exclude-allocations": exclude,
                }
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
    packages = simulated_packages()
    counts: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for path in paths:
        current = None
        for line in path.read_text(errors="replace").splitlines():
            if match := re.search(r"\bRunning (\S+) (\S+)$", line):
                current = (match.group(1), match.group(2))
                if current[0] in packages:
                    counts[current[0]].setdefault(current[1], 0)
                else:
                    current = None
            elif "Done running" in line:
                current = None
            elif current and "Measured: " in line:
                counts[current[0]][current[1]] += 1
    return {package: dict(sorted(benches.items())) for package, benches in sorted(counts.items())}


def durations_from_logs(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Measure complete target executions, including startup and untimed fixture construction."""
    packages = simulated_packages()
    samples: dict[Target, list[float]] = collections.defaultdict(list)
    for path in paths:
        current = None
        started = None
        for line in path.read_text(errors="replace").splitlines():
            if match := re.search(r"\bRunning (\S+) (\S+)$", line):
                current = (match.group(1), match.group(2)) if match.group(1) in packages else None
                if current:
                    started = datetime.datetime.fromisoformat(line.split()[0])
            elif current and started is not None and re.search(rf"\bDone running {re.escape(current[1])}$", line):
                ended = datetime.datetime.fromisoformat(line.split()[0])
                samples[current].append((ended - started).total_seconds())
                current = None
    durations: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for (package, target), values in sorted(samples.items()):
        durations[package][target] = round(statistics.median(values), 3)
    return dict(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print the shard plan as GITHUB_OUTPUT lines")
    counts_parser = sub.add_parser("counts", help="regenerate benchmark counts from CodSpeed job logs")
    counts_parser.add_argument("logs", nargs="+", type=Path)
    durations_parser = sub.add_parser("durations", help="record target runtimes from completed simulation job logs")
    durations_parser.add_argument("--source", required=True, help="workflow run URL")
    durations_parser.add_argument("--commit", required=True, help="measured commit SHA")
    durations_parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    if args.command == "counts":
        json.dump(counts_from_logs(args.logs), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if args.command == "durations":
        json.dump(
            {"source": args.source, "commit": args.commit, "seconds": durations_from_logs(args.logs)},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return

    known = load_counts()
    durations = load_durations()
    counts = {}
    for target in discover_targets():
        if target not in known:
            print(
                f"::warning::{target[0]}::{target[1]} is missing from {COUNTS_PATH.name}; assuming "
                f"{DEFAULT_COUNT} benchmarks. Regenerate the counts with `codspeed_shards.py counts`.",
                file=sys.stderr,
            )
        counts[target] = known.get(target, DEFAULT_COUNT)
        if target not in durations:
            print(
                f"::warning::{target[0]}::{target[1]} is missing from {DURATIONS_PATH.name}; "
                f"assuming {DEFAULT_SECONDS}s. Regenerate durations from simulation job logs.",
                file=sys.stderr,
            )
    shards = plan(counts, durations=durations)
    for shard in shards:
        print(
            f"shard {shard['shard']}: {shard['benchmarks']} benchmarks, "
            f"{shard['seconds']:.1f}s estimated execution, {shard['packages']}",
            file=sys.stderr,
        )
    for name, value in github_outputs(shards).items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
