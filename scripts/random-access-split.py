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
from collections.abc import Callable
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/random-access-bench"
PARTS_DIR = Path("parts")

DATASETS = ["taxi", "feature-vectors", "nested-lists", "nested-structs"]
FORMATS = ["parquet", "lance", "vortex"]
PATTERNS = ["correlated", "uniform"]
OPEN_MODES = ["cached", "reopen"]


def run_combinations(emit_v3: bool) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    i = 0
    for dataset in DATASETS:
        for fmt in FORMATS:
            for pattern in PATTERNS:
                for open_mode in OPEN_MODES:
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
                        "-d",
                        "gh-json",
                        "-o",
                        str(PARTS_DIR / f"{i}.gh.json"),
                    ]
                    if emit_v3:
                        args += ["--gh-json-v3", str(PARTS_DIR / f"{i}.v3.jsonl")]
                    print("+", " ".join(args), flush=True)
                    subprocess.run(args, check=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3",
        action="store_true",
        help="merge --gh-json-v3 records into results.v3.jsonl",
    )
    args = parser.parse_args()

    run_combinations(args.v3)
    merge(f"{PARTS_DIR}/*.gh.json", lambda record: record["name"], "results.json")
    if args.v3:
        merge(
            f"{PARTS_DIR}/*.v3.jsonl",
            lambda record: (record["kind"], record["dataset"], record["format"]),
            "results.v3.jsonl",
        )


if __name__ == "__main__":
    main()
