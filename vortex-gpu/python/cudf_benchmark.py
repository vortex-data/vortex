#!/usr/bin/env python3
"""
Benchmark script to read a Parquet file into cuDF and run a simple query (x+10).

This script demonstrates loading data into a GPU-backed DataFrame using cuDF
and measuring the performance of a simple arithmetic operation.

Usage:
    python cudf_benchmark.py <path_to_parquet_file> [--column <column_name>]

Example:
    python cudf_benchmark.py data.parquet --column x
"""

import argparse
import time
import cudf


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cuDF performance for reading Parquet and running x+10 query"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="Path to the Parquet file to read"
    )
    parser.add_argument(
        "--column",
        type=str,
        default="x",
        help="Name of the column to add 10 to (default: 'x')"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of times to run the query (default: 1)"
    )
    parser.add_argument(
        "--read-iterations",
        type=int,
        default=5,
        help="Number of times to read the Parquet file (default: 1)"
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="List of columns to read from Parquet (default: read all columns)"
    )

    args = parser.parse_args()

    print(f"Reading Parquet file: {args.parquet_file}")
    print(f"Read iterations: {args.read_iterations}")
    print(f"Query iterations: {args.iterations}")
    if args.columns:
        print(f"Reading columns: {args.columns}")

    # Time the Parquet file reading multiple times
    read_times = []
    df = None

    for i in range(args.read_iterations):
        start_read = time.perf_counter()
        df = cudf.read_parquet(args.parquet_file, columns=args.columns)
        end_read = time.perf_counter()

        read_time = end_read - start_read
        read_times.append(read_time)

        if args.read_iterations == 1 or i == 0:
            print(f"\nRead time (iteration {i+1}): {read_time:.6f} seconds")

    if args.read_iterations > 1:
        print("{}", read_times)
        avg_read_time = sum(read_times) / len(read_times)
        min_read_time = min(read_times)
        max_read_time = max(read_times)
        print(f"\nRead statistics over {args.read_iterations} iterations:")
        print(f"  Average: {avg_read_time:.6f} seconds")
        print(f"  Min: {min_read_time:.6f} seconds")
        print(f"  Max: {max_read_time:.6f} seconds")
    else:
        avg_read_time = read_times[0]

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if the specified column exists
    if args.column not in df.columns:
        print(f"\nError: Column '{args.column}' not found in the DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"\nColumn '{args.column}' dtype: {df[args.column].dtype}")
    print(f"Column '{args.column}' shape: {df[args.column].shape}")

    # Time the query execution (x+10)
    query_times = []

    for i in range(args.iterations):
        start_query = time.perf_counter()
        result = df[args.column] + 10
        # Force computation by accessing the result (synchronize GPU)
        _ = result.values
        end_query = time.perf_counter()

        query_time = end_query - start_query
        query_times.append(query_time)

        if args.iterations == 1 or i == 0:
            print(f"\nQuery execution time (iteration {i+1}): {query_time:.6f} seconds")

    if args.iterations > 1:
        print("{query_times}")
        avg_query_time = sum(query_times) / len(query_times)
        min_query_time = min(query_times)
        max_query_time = max(query_times)
        print(f"\nQuery statistics over {args.iterations} iterations:")
        print(f"  Average: {avg_query_time:.6f} seconds")
        print(f"  Min: {min_query_time:.6f} seconds")
        print(f"  Max: {max_query_time:.6f} seconds")

    # Show a sample of the result
    print(f"\nFirst 10 values of original column '{args.column}':")
    print(df[args.column].head(10))

    print(f"\nFirst 10 values after adding 10:")
    print(result.head(10))

    # Summary statistics
    avg_query_time = sum(query_times) / len(query_times) if query_times else 0
    total_read_time = sum(read_times)
    total_query_time = sum(query_times)

    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total read time ({args.read_iterations} iterations): {total_read_time:.6f} seconds")
    print(f"Average read time: {avg_read_time:.6f} seconds")
    print(f"\nTotal query time ({args.iterations} iterations): {total_query_time:.6f} seconds")
    if args.iterations > 1:
        print(f"Average query time: {avg_query_time:.6f} seconds")
    print(f"\nTotal time (all iterations): {total_read_time + total_query_time:.6f} seconds")
    print(f"Average time (read + query): {avg_read_time + avg_query_time:.6f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
