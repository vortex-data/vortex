#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Split a monolithic benchmark JSONL file into per-benchmark files.

Each record is routed to a file based on benchmark name patterns and metadata
(storage, scale_factor). The output file names match the CI matrix IDs used
in .github/workflows/*.yml.

Usage:
    python3 scripts/split-benchmark-data.py <input.json> <output_dir>
"""

import gzip
import json
import math
import os
import sys
from collections import defaultdict


def classify_record(record: dict) -> str | None:
    """Map a benchmark record to its CI matrix ID (S3 file key).

    Returns None for records that don't match any known benchmark.
    """
    name = record.get("name", "")
    lower = name.lower()

    # Random access benchmarks
    if lower.startswith("random-access/") or lower.startswith("random access/"):
        return "random-access-bench"

    # Compression benchmarks (timing, size, and ratio measurements)
    if any(
        lower.startswith(prefix)
        for prefix in [
            "compress time/",
            "decompress time/",
            "parquet_rs-zstd compress",
            "parquet_rs-zstd decompress",
            "lance compress",
            "lance decompress",
            "vortex:lance ratio",
            "vortex:parquet-zstd ratio",
            "vortex:raw ratio",
            "vortex size/",
            "vortex-file-compressed size/",
            "parquet size/",
            "lance size/",
        ]
    ) or any(
        pattern in lower
        for pattern in [
            ":raw size/",
            ":parquet-zstd size/",
            ":lance size/",
        ]
    ):
        return "compress-bench"

    # SQL query benchmarks: route by prefix + dataset metadata
    sql_suites = {
        "clickbench": {"fan_out": False, "dataset_key": None, "id": "clickbench-nvme"},
        "statpopgen": {"fan_out": False, "dataset_key": None, "id": "statpopgen"},
        "polarsignals": {"fan_out": False, "dataset_key": None, "id": "polarsignals"},
        "fineweb": {"fan_out": False, "dataset_key": None, "id": None},  # needs storage check
        "tpch": {"fan_out": True, "dataset_key": "tpch"},
        "tpcds": {"fan_out": True, "dataset_key": "tpcds"},
    }

    for prefix, suite in sql_suites.items():
        if not lower.startswith(prefix + "_q") and not lower.startswith(prefix + "/"):
            continue

        # Non-fan-out suites with fixed IDs
        if not suite["fan_out"] and suite.get("id"):
            return suite["id"]

        # FineWeb: check storage
        if prefix == "fineweb":
            storage = (record.get("storage") or "").upper()
            return "fineweb-s3" if storage == "S3" else "fineweb"

        # Fan-out suites: determine ID from storage + scale_factor
        storage = (record.get("storage") or "").upper()
        storage_suffix = "s3" if storage == "S3" else "nvme"

        dataset = record.get("dataset") or {}
        dataset_key = suite["dataset_key"]
        raw_sf = None
        if dataset_key and dataset_key in dataset:
            raw_sf = dataset[dataset_key].get("scale_factor")

        sf = round(float(raw_sf)) if raw_sf else 1

        # Map to CI matrix IDs:
        #   tpch-nvme (SF=1), tpch-s3 (SF=1)
        #   tpch-nvme-10, tpch-s3-10
        #   tpch-nvme-100, tpch-s3-100
        #   tpcds-nvme (SF=1)
        sf_suffix = "" if sf <= 1 else f"-{sf}"
        return f"{prefix}-{storage_suffix}{sf_suffix}"

    return None


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Accumulate records per benchmark ID
    buckets: dict[str, list[str]] = defaultdict(list)
    total = 0
    unclassified = 0
    unclassified_prefixes: set[str] = set()

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            benchmark_id = classify_record(record)
            if benchmark_id is None:
                unclassified += 1
                name = record.get("name", "")
                unclassified_prefixes.add(name.split("/")[0])
                continue

            buckets[benchmark_id].append(line)

    # Write gzipped JSONL files
    for benchmark_id, lines in sorted(buckets.items()):
        output_path = os.path.join(output_dir, f"{benchmark_id}.data.json.gz")
        with gzip.open(output_path, "wt") as f:
            for line in lines:
                f.write(line + "\n")
        print(f"  {benchmark_id}: {len(lines)} records")

    print(f"\nTotal: {total} records")
    print(f"Classified: {total - unclassified} records across {len(buckets)} files")
    if unclassified:
        print(f"Unclassified: {unclassified} records")
        print(f"  Prefixes: {', '.join(sorted(unclassified_prefixes)[:20])}")


if __name__ == "__main__":
    main()
