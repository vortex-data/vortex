# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Run compress-bench once per dataset, drop OS cache between datasets,
merte outputs.
"""

import argparse
import glob
import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/compress-bench"
PARTS_DIR = Path("parts")

# TODO(myrrc): we should also drop CUDA allocator caches


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


def list_datasets(gpu_decompress: bool) -> list[str]:
    cmd = [BINARY, "--print-datasets"]
    if gpu_decompress:
        cmd.append("--gpu-decompress")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def prepare_data(gpu_decompress: bool) -> None:
    for dataset in list_datasets(gpu_decompress):
        args = [BINARY, "--prepare-data", "--datasets", f"^{re.escape(dataset)}$"]
        if gpu_decompress:
            args.append("--gpu-decompress")
        with log_group(f"Generate {dataset}"):
            print("+", " ".join(args), flush=True)
            subprocess.run(args, check=True)


def run_datasets(formats: str, emit_ingest_records: bool, gpu_decompress: bool, cpus: int | None) -> list[str]:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    datasets = list_datasets(gpu_decompress)
    for i, dataset in enumerate(datasets):
        drop_os_caches()

        args = [
            "bash",
            str(SCRIPT_DIR / "bench-taskset.sh"),
            BINARY,
            "--datasets",
            f"^{re.escape(dataset)}$",
        ]
        if gpu_decompress:
            # GPU mode fixes its own formats
            args.append("--gpu-decompress")
        else:
            args += ["--formats", formats]
        if cpus is not None:
            args += ["--threads", str(cpus)]
        args += ["-d", "gh-json", "-o", str(PARTS_DIR / f"{i}.gh.json")]
        if emit_ingest_records:
            args += ["--ingest-jsonl", str(PARTS_DIR / f"{i}.ingest.jsonl")]

        env = os.environ.copy()
        if cpus is not None:
            # Pin the process to `cpus` CPUs (bench-taskset.sh) and size the runtime to match,
            # so every format runs on the same fixed core budget.
            env["BENCH_MAX_CPUS"] = str(cpus)
        with log_group(f"Benchmark [{i + 1}/{len(datasets)}] {dataset}"):
            print("+", " ".join(args), flush=True)
            result = subprocess.run(args, check=not gpu_decompress, env=env)
        if result.returncode != 0:
            failures.append(dataset)
    return failures


def merge(pattern: str, out_path: str) -> None:
    lines: list[str] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    lines.append(line)
    Path(out_path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        default="arrow-ipc,parquet,vortex",
        help="comma-separated formats to forward to compress-bench",
    )
    parser.add_argument(
        "--emit-ingest-records",
        action="store_true",
        help="merge --ingest-jsonl records into results.ingest.jsonl",
    )
    parser.add_argument(
        "--gpu-decompress",
        action="store_true",
        help="run the GPU decompression suite (forwards --gpu-decompress, ignores --formats)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=16,
        help="CPU cores (and worker threads) the timed CPU benchmark runs on; 0 means all",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="only generate the benchmark data files, without running the benchmark",
    )
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data(args.gpu_decompress)
        return

    cpus = None if args.gpu_decompress or args.cpus <= 0 else args.cpus
    failures = run_datasets(args.formats, args.emit_ingest_records, args.gpu_decompress, cpus)
    merge(f"{PARTS_DIR}/*.gh.json", "results.json")
    if args.emit_ingest_records:
        merge(f"{PARTS_DIR}/*.ingest.jsonl", "results.ingest.jsonl")

    if failures:
        raise SystemExit("GPU decompression failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
