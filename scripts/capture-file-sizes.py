#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Capture file sizes from benchmark data directories and output as JSONL."""

import argparse
import json
import sys
from pathlib import Path

DUCKDB_CATALOG_FILES = {"duckdb.db", "duckdb.db.wal"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture file sizes from benchmark data directories")
    parser.add_argument("data_dir", help="Data directory (e.g., vortex-bench/data)")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g., clickbench)")
    parser.add_argument(
        "--dataset-dir", required=True, help="The benchmark's directory under data_dir (e.g., clickbench_partitioned)"
    )
    parser.add_argument("--commit", required=True, help="Commit SHA")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--ingest-jsonl",
        help="Also write the sizes as `file_size` records for `post-ingest.py` to this JSONL file",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    benchmark_dir = data_dir / args.dataset_dir
    if not benchmark_dir.is_dir():
        print(f"Benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    # Every on-disk format, named as `Format::name()` in `vortex-bench/src/lib.rs` names its
    # directory. The "vortex" CLI argument maps to the "vortex-file-compressed" directory.
    formats_to_capture = {
        "arrow-ipc",
        "csv",
        "duckdb",
        "lance",
        "litdata",
        "parquet",
        "vortex-compact",
        "vortex-file-compressed",
        "vortex-spatial-native",
    }

    records = []

    # Walk subdirectories looking for format directories
    # Handle both direct format dirs (clickbench_partitioned/vortex-file-compressed/)
    # and scale factor subdirs (tpch/1.0/vortex-file-compressed/)
    for format_dir in benchmark_dir.rglob("*"):
        if not format_dir.is_dir():
            continue

        format_name = format_dir.name
        if format_name not in formats_to_capture:
            continue

        # Extract scale factor from path (e.g., "1.0" for tpch/1.0/vortex-file-compressed)
        # Default to "1.0" if no intermediate directory (e.g., clickbench)
        path_between = format_dir.relative_to(benchmark_dir).parent
        scale_factor = str(path_between) if str(path_between) != "." else "1.0"

        # Capture all files in this format directory
        for file_path in format_dir.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(format_dir)
            # DuckDB keeps a catalog of views over the files in each format directory it queries
            # (`duckdb.db`, and its WAL while open). Only the `duckdb` format's database is data.
            if format_name != "duckdb" and relative_path.name in DUCKDB_CATALOG_FILES:
                continue

            size_bytes = file_path.stat().st_size

            records.append(
                {
                    "metric": "file_size",
                    "unit": "bytes",
                    "value": size_bytes,
                    "commit_id": args.commit,
                    "file_size": {
                        "benchmark": args.benchmark,
                        "scale_factor": scale_factor,
                        "format": format_name,
                        "file": str(relative_path),
                    },
                }
            )

    # Sort for deterministic output
    records.sort(
        key=lambda r: (
            r["file_size"]["benchmark"],
            r["file_size"]["scale_factor"],
            r["file_size"]["format"],
            r["file_size"]["file"],
        )
    )

    # Write JSONL output
    with open(args.output, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    if args.ingest_jsonl:
        with open(args.ingest_jsonl, "w") as f:
            for record in records:
                f.write(
                    json.dumps(
                        {
                            "kind": "file_size",
                            "commit_sha": record["commit_id"],
                            **record["file_size"],
                            "value_bytes": record["value"],
                        }
                    )
                    + "\n"
                )

    print(f"Captured {len(records)} file sizes to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
