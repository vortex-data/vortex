# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Run string-bench once per (column, encoder), drop OS cache between runs, merge
outputs
"""

import argparse
import glob
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/string-bench"
PARTS_DIR = Path("parts")

# Every string-bench input name (`Input::name` in benchmarks/string-bench/src/main.rs).
COLUMNS = [
    "URL",
    "l_comment",
    "amazon/book_reviews",
    "amazon/book_titles",
    "news/headlines",
    "twitter/sentiment140",
    "fineweb/text",
]
ENCODERS = ["onpair", "fsst"]


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


def prepare_data() -> None:
    for column in COLUMNS:
        args = [BINARY, "--prepare-data", "--columns", f"^{re.escape(column)}$"]
        with log_group(f"Generate {column}"):
            print("+", " ".join(args), flush=True)
            subprocess.run(args, check=True)


def run_combinations() -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(COLUMNS) * len(ENCODERS)
    i = 0
    for column in COLUMNS:
        for encoder in ENCODERS:
            drop_os_caches()

            args = [
                "bash",
                str(SCRIPT_DIR / "bench-taskset.sh"),
                BINARY,
                "--columns",
                f"^{re.escape(column)}$",
                "--encoders",
                encoder,
                "-d",
                "gh-json",
                "-o",
                str(PARTS_DIR / f"{i}.gh.json"),
            ]
            with log_group(f"Benchmark [{i + 1}/{total}] {column} / {encoder}"):
                print("+", " ".join(args), flush=True)
                subprocess.run(args, check=True)
            i += 1


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
        "--prepare-data",
        action="store_true",
        help="only generate the benchmark data files, without running the benchmark",
    )
    args = parser.parse_args()

    if args.prepare_data:
        prepare_data()
        return

    run_combinations()
    merge(f"{PARTS_DIR}/*.gh.json", "results.json")


if __name__ == "__main__":
    main()
