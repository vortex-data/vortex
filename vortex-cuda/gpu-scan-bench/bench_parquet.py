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
import os
import sys
import tempfile
import time

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cuDF GPU parquet reads",
    )
    parser.add_argument("source", help="Path to parquet file")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of scan iterations"
    )
    parser.add_argument(
        "--output-size-mb",
        type=float,
        default=None,
        help="Decompressed output size in MB (avoids reading the file to measure it)",
    )
    args = parser.parse_args()

    import cudf
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = args.source
    file_size = os.path.getsize(source)
    file_size_mb = file_size / (1024 * 1024)
    output_size_mb = args.output_size_mb

    # ---- Pre-compile cuDF/CUDA kernels (untimed) -------------------------
    # Read a tiny parquet file through cuDF so that all internal CUDA JIT /
    # kernel compilation happens before the timed region, matching how the
    # Vortex GPU bench pre-compiles its PTX modules.
    print("Pre-compiling cuDF kernels...", file=sys.stderr)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
        tbl = pa.table({"x": pa.array([1, 2, 3], type=pa.int64())})
        pq.write_table(tbl, tmp_path)
    _ = cudf.read_parquet(tmp_path)
    os.unlink(tmp_path)
    print("Pre-compilation done", file=sys.stderr)

    # ---- Timed iterations ------------------------------------------------
    iteration_secs = []
    for i in range(args.iterations):
        start = time.perf_counter()
        df = cudf.read_parquet(source)
        del df
        elapsed = time.perf_counter() - start
        iteration_secs.append(elapsed)
        print(
            f"Iteration {i + 1}/{args.iterations}: {elapsed:.3f}s",
            file=sys.stderr,
        )

    # ---- Results ---------------------------------------------------------
    first_secs = iteration_secs[0]
    best_secs = min(iteration_secs)

    print(file=sys.stderr)
    print("=== Benchmark Results ===", file=sys.stderr)
    print(f"Source:      {source}", file=sys.stderr)
    print(f"Iterations:  {args.iterations}", file=sys.stderr)
    print(f"File size:   {file_size_mb:,.2f} MB", file=sys.stderr)
    if output_size_mb is not None:
        print(f"Output size: {output_size_mb:,.2f} MB", file=sys.stderr)
        print(f"Compression: {output_size_mb / file_size_mb:.1f}x", file=sys.stderr)
    print(file=sys.stderr)

    cold_line = f"Cold (first iter):  {first_secs:.3f}s  input: {file_size_mb / first_secs:,.0f} MB/s"
    warm_line = f"Warm (best iter):   {best_secs:.3f}s  input: {file_size_mb / best_secs:,.0f} MB/s"
    if output_size_mb is not None:
        cold_line += f"  output: {output_size_mb / first_secs:,.0f} MB/s"
        warm_line += f"  output: {output_size_mb / best_secs:,.0f} MB/s"
    print(cold_line, file=sys.stderr)
    print(warm_line, file=sys.stderr)


if __name__ == "__main__":
    main()
