# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Run random-access-bench once per (dataset, format, pattern, open-mode)
then merge the per-combination outputs
"""

import argparse
import glob
import json
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/random-access-bench"
PARTS_DIR = Path("parts")

DATASETS = ["taxi", "feature-vectors", "nested-lists", "nested-structs"]
# EXPERIMENT (not for merge): same-machine A/B of read-path variants, Vortex only.
FORMATS = ["vortex"]
XP_MODES = ["base", "dev", "conc8", "conc128", "coal64", "nocoal", "batch", "batch4", "inflight16"]
# Rounds per open mode; every mode runs once per round, in a shuffled order.
XP_ROUNDS = {"cached": 1, "reopen": 2}
# Only this mode's first round feeds the results the PR comment compares.
XP_REPORTED_MODE = "base"
# Formats only benchmarked from local disk. A remote run would silently read them from local disk.
LOCAL_ONLY_FORMATS = {"arrow-ipc", "litdata"}
PATTERNS = ["correlated", "uniform"]
OPEN_MODES = ["cached", "reopen"]
# Seconds each measurement runs for, passed as --time-limit.
TIME_LIMIT_SECS = 3


def drop_os_caches() -> None:
    try:
        subprocess.run(["sync"], check=True)
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pass


@contextmanager
def log_group(title: str) -> Iterator[None]:
    """Fold the output printed inside into a collapsible GitHub Actions log group."""
    print(f"::group::{title}", flush=True)
    try:
        yield
    finally:
        print("::endgroup::", flush=True)


def benchmark_formats(remote_data_dir: str | None) -> list[str]:
    return [f for f in FORMATS if not (remote_data_dir and f in LOCAL_ONLY_FORMATS)]


def prepare_data(remote_data_dir: str | None) -> None:
    """Generate every dataset concurrently, one process per dataset.

    Files that already exist, such as those CI restores from its cache, are not regenerated.
    """
    joined = ",".join(benchmark_formats(remote_data_dir))
    procs = []
    for dataset in DATASETS:
        args = [BINARY, "--prepare-data", "--datasets", dataset, "--formats", joined]
        print("+", " ".join(args), flush=True)
        procs.append((dataset, subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)))

    failed = []
    for dataset, proc in procs:
        output, _ = proc.communicate()
        with log_group(f"Generate {dataset} ({joined})"):
            print(output, end="", flush=True)
        if proc.returncode != 0:
            failed.append(dataset)
    if failed:
        raise SystemExit(f"data generation failed for: {', '.join(failed)}")


def run_combinations(emit_ingest_records: bool, remote_data_dir: str | None) -> None:
    import os
    import random

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    formats = benchmark_formats(remote_data_dir)
    jobs = []
    for rnd in range(max(XP_ROUNDS.values())):
        for dataset in DATASETS:
            for fmt in formats:
                for pattern in PATTERNS:
                    for open_mode in OPEN_MODES:
                        if rnd >= XP_ROUNDS[open_mode]:
                            continue
                        modes = list(XP_MODES)
                        random.shuffle(modes)
                        jobs += [(rnd, dataset, fmt, pattern, open_mode, m) for m in modes]
    total = len(jobs)
    i = 0
    for n, (rnd, dataset, fmt, pattern, open_mode, xp_mode) in enumerate(jobs, 1):
                    report = rnd == 0 and xp_mode == XP_REPORTED_MODE
                    os.environ["XP_MODE"] = xp_mode
                    drop_os_caches()

                    args = [
                        "bash",
                        str(SCRIPT_DIR / "bench-taskset.sh"),
                        BINARY,
                        "--datasets",
                        dataset,
                        "--formats",
                        fmt,
                        "--patterns",
                        pattern,
                        "--open-mode",
                        open_mode,
                        "--time-limit",
                        str(TIME_LIMIT_SECS),
                        "-d",
                        "gh-json",
                        "-o",
                        str(PARTS_DIR / (f"{i}.gh.json" if report else f"xp{n}.gh.json.skip")),
                    ]
                    if remote_data_dir:
                        args += ["--remote-data-dir", remote_data_dir]
                    if emit_ingest_records and report:
                        args += ["--ingest-jsonl", str(PARTS_DIR / f"{i}.ingest.jsonl")]
                    with log_group(f"Benchmark [{n}/{total}] round {rnd} {xp_mode} {dataset} / {fmt} / {pattern} / {open_mode}"):
                        print("+", f"XP_MODE={xp_mode}", " ".join(args), flush=True)
                        subprocess.run(args, check=True)
                    if report:
                        i += 1


"""
This function exists only because of taxi-legacy.

Every taxi invocation re-emits the pattern-less legacy taxi rows, so we need
the merge to drop the duplicates. Otherwise we could just merge JSONL lines.
"""


def merge(pattern: str, key: Callable[[dict], object], out_path: str) -> None:
    seen: set[object] = set()
    lines: list[str] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                identity = key(json.loads(line))
                if identity in seen:
                    continue
                seen.add(identity)
                lines.append(line)
    Path(out_path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def ingest_identity(record: dict) -> tuple[object, object, object, object]:
    return (
        record["kind"],
        record["dataset"],
        record["format"],
        record["open_mode"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-ingest-records",
        action="store_true",
        help="merge --ingest-jsonl records into results.ingest.jsonl",
    )
    parser.add_argument(
        "--remote-data-dir",
        help="read the benchmark data from this remote directory (e.g. s3://bucket/prefix/)",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="only generate the benchmark data files, without running the benchmark",
    )
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data(args.remote_data_dir)
        return

    run_combinations(args.emit_ingest_records, args.remote_data_dir)
    merge(f"{PARTS_DIR}/*.gh.json", lambda record: record["name"], "results.json")
    if args.emit_ingest_records:
        merge(
            f"{PARTS_DIR}/*.ingest.jsonl",
            ingest_identity,
            "results.ingest.jsonl",
        )


if __name__ == "__main__":
    main()
