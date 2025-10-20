#!/usr/bin/env python3
"""
Generate a test Parquet file with two columns and configurable number of rows.

Columns:
  - u32_col: uint32 values (random between min_value and max_value)
  - u64_col: uint64 values (random between min_value and max_value)

Usage:
    python generate_test_data.py [output_file]

Example:
    python generate_test_data.py test_data.parquet
    python generate_test_data.py --rows 100000000 data.parquet
"""

import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(
        description="Generate a test Parquet file with u32 and u64 columns"
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default="test_data.parquet",
        help="Output Parquet file path (default: test_data.parquet)"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100_000_000,
        help="Number of rows to generate (default: 1,000,000)"
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=10_000,
        help="Minimum value for u64 column (default: 10,000)"
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=100_000,
        help="Maximum value for u64 column (default: 100,000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print(f"Generating Parquet file with {args.rows:,} rows...")
    print(f"  u32_col: random values between {args.min_value:,} and {args.max_value:,}")
    print(f"  u64_col: random values between {args.min_value:,} and {args.max_value:,}")
    print(f"  Random seed: {args.seed}")

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Generate data
    u32_col = np.random.randint(
        args.min_value,
        args.max_value + 1,  # +1 because randint is exclusive on upper bound
        size=args.rows,
        dtype=np.uint32
    )
    u64_col = np.random.randint(
        args.min_value,
        args.max_value + 1,  # +1 because randint is exclusive on upper bound
        size=args.rows,
        dtype=np.uint64
    )

    # Create PyArrow table
    table = pa.table({
        'u32_col': pa.array(u32_col, type=pa.uint32()),
        'u64_col': pa.array(u64_col, type=pa.uint64())
    })

    # Write to Parquet
    print(f"\nWriting to {args.output_file}...")
    pq.write_table(table, args.output_file)

    # Verify the file
    file_size_mb = pa.parquet.ParquetFile(args.output_file).metadata.serialized_size / (1024 * 1024)
    print(f"✓ File created successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Rows: {args.rows:,}")
    print(f"  Columns: u32_col (uint32), u64_col (uint64)")

    # Show sample data
    print("\nSample data (first 5 rows):")
    sample_table = pq.read_table(args.output_file, columns=['u32_col', 'u64_col'])
    print(sample_table.to_pandas().head())


if __name__ == "__main__":
    main()
