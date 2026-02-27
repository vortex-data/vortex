#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cudf-cu12",
# ]
#
# [tool.uv]
# extra-index-url = ["https://pypi.nvidia.com"]
# ///
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Benchmark reading a Parquet file into GPU memory using cuDF.
# This serves as the baseline for comparing against Vortex GPU scans.
#
# Usage:
#   uv run bench_parquet.py dataset.parquet --iterations 5

import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cuDF GPU parquet reads",
    )
    parser.add_argument("source", help="Path to parquet file")
    parser.add_argument("--iterations", type=int, default=1, help="Number of scan iterations")
    args = parser.parse_args()

    import cudf
    import fsspec

    source = args.source
    fs, fs_path = fsspec.core.url_to_fs(source)
    file_size = fs.size(fs_path)
    file_size_mb = file_size / (1024 * 1024)

    iteration_secs = []
    for i in range(args.iterations):
        start = time.perf_counter()
        df = cudf.read_parquet(source)
        elapsed = time.perf_counter() - start
        iteration_secs.append(elapsed)
        print(
            f"Iteration {i + 1}/{args.iterations}: {elapsed:.3f}s",
            file=sys.stderr,
        )
        del df

    avg_secs = sum(iteration_secs) / len(iteration_secs)
    throughput_mbs = file_size_mb / avg_secs

    print(file=sys.stderr)
    print("=== Benchmark Results ===", file=sys.stderr)
    print(f"Source:     {source}", file=sys.stderr)
    print(f"Iterations: {args.iterations}", file=sys.stderr)
    print(f"Avg time:   {avg_secs:.3f}s", file=sys.stderr)
    print(f"File size:  {file_size_mb:.2f} MB", file=sys.stderr)
    print(f"Throughput: {throughput_mbs:.2f} MB/s", file=sys.stderr)


if __name__ == "__main__":
    main()
