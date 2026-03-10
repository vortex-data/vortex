#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cudf-cu12",
#   "s3fs",
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
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cuDF GPU parquet reads",
    )
    parser.add_argument("source", help="Path to parquet file")
    parser.add_argument("--iterations", type=int, default=1, help="Number of scan iterations")
    parser.add_argument(
        "--row-group-batch-size",
        type=int,
        default=1,
        help="Number of parquet row groups to read per cuDF call when streaming",
    )
    parser.add_argument(
        "--full-file-read",
        action="store_true",
        help="Read the full parquet file in one call (old behavior, can OOM)",
    )
    args = parser.parse_args()

    import cudf
    import fsspec
    import pyarrow.parquet as pq

    source = args.source
    if args.row_group_batch_size < 1:
        raise ValueError("--row-group-batch-size must be >= 1")

    fs, fs_path = fsspec.core.url_to_fs(source)
    file_size = fs.size(fs_path)
    file_size_mb = file_size / (1024 * 1024)

    num_row_groups = None
    if not args.full_file_read:
        with fs.open(fs_path, "rb") as parquet_file:
            num_row_groups = pq.ParquetFile(parquet_file).metadata.num_row_groups
        print(
            f"Streaming parquet by row groups: {num_row_groups} total, "
            f"batch size={args.row_group_batch_size}",
            file=sys.stderr,
        )

    iteration_secs = []
    output_bytes = 0
    for i in range(args.iterations):
        start = time.perf_counter()
        iter_bytes = 0
        if args.full_file_read:
            df = cudf.read_parquet(source)
            iter_bytes = df.memory_usage(deep=True).sum()
            del df
        else:
            for rg_start in range(0, num_row_groups, args.row_group_batch_size):
                row_groups = list(
                    range(rg_start, min(rg_start + args.row_group_batch_size, num_row_groups))
                )
                df = cudf.read_parquet(source, row_groups=row_groups)
                iter_bytes += df.memory_usage(deep=True).sum()
                del df
        elapsed = time.perf_counter() - start
        iteration_secs.append(elapsed)
        if i == 0:
            output_bytes = iter_bytes
        print(
            f"Iteration {i + 1}/{args.iterations}: {elapsed:.3f}s",
            file=sys.stderr,
        )

    avg_secs = sum(iteration_secs) / len(iteration_secs)
    output_size_mb = output_bytes / (1024 * 1024)
    input_throughput_mbs = file_size_mb / avg_secs
    output_throughput_mbs = output_size_mb / avg_secs

    print(file=sys.stderr)
    print("=== Benchmark Results ===", file=sys.stderr)
    print(f"Source:      {source}", file=sys.stderr)
    print(f"Iterations:  {args.iterations}", file=sys.stderr)
    print(f"Avg time:    {avg_secs:.3f}s", file=sys.stderr)
    print(f"Input size:  {file_size_mb:.2f} MB", file=sys.stderr)
    print(f"Output size: {output_size_mb:.2f} MB", file=sys.stderr)
    print(f"Input throughput:  {input_throughput_mbs:.2f} MB/s", file=sys.stderr)
    print(f"Output throughput: {output_throughput_mbs:.2f} MB/s", file=sys.stderr)


if __name__ == "__main__":
    main()
