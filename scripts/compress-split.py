# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Run compress-bench once per dataset, drop OS cache between datasets,
merte outputs.
"""

import argparse
import glob
import re
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/compress-bench"
PARTS_DIR = Path("parts")


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


def list_datasets() -> list[str]:
    result = subprocess.run([BINARY, "--print-datasets"], check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run_datasets(formats: str, emit_ingest_records: bool) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, dataset in enumerate(list_datasets()):
        drop_os_caches()

        args = [
            "bash",
            str(SCRIPT_DIR / "bench-taskset.sh"),
            BINARY,
            "--datasets",
            f"^{re.escape(dataset)}$",
            "--formats",
            formats,
            "-d",
            "gh-json",
            "-o",
            str(PARTS_DIR / f"{i}.gh.json"),
        ]
        if emit_ingest_records:
            args += ["--ingest-jsonl", str(PARTS_DIR / f"{i}.ingest.jsonl")]
        print("+", " ".join(args), flush=True)
        subprocess.run(args, check=True)


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
        default="parquet,vortex",
        help="comma-separated formats to forward to compress-bench",
    )
    parser.add_argument(
        "--emit-ingest-records",
        action="store_true",
        help="merge --ingest-jsonl records into results.ingest.jsonl",
    )
    args = parser.parse_args()

    run_datasets(args.formats, args.emit_ingest_records)
    merge(f"{PARTS_DIR}/*.gh.json", "results.json")
    if args.emit_ingest_records:
        merge(f"{PARTS_DIR}/*.ingest.jsonl", "results.ingest.jsonl")


if __name__ == "__main__":
    main()
