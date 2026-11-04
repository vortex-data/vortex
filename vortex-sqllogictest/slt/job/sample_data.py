# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Write a proportional sample of the JOB (IMDB) tables as Parquet.

Reads DuckDB's published JOB Parquet files over HTTP and keeps the first
SAMPLE_FRACTION of each table's rows, so only the leading row groups are
fetched instead of the full ~1.8 GB. Tables with at most SMALL_TABLE_ROWS rows
are lookup tables that the queries filter by value, so they are kept whole.
Keeping the relative table sizes keeps DataFusion's join planning close to the
full dataset. Output is zstd level 3, like the benchmark data.

Usage: sample_data.py OUTPUT_DIR
"""

import math
import sys
from pathlib import Path

import duckdb

UPSTREAM_BASE_URL = "https://github.com/duckdb/duckdb-data/releases/download/v1.0"
SAMPLE_FRACTION = 0.01
SMALL_TABLE_ROWS = 1000

# Keep in sync with `TABLES` in vortex-bench/src/job/mod.rs.
TABLES = [
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    # A single thread with insertion order preserved makes `LIMIT` pick the
    # same leading rows on every run.
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = true")
    for table in TABLES:
        url = f"{UPSTREAM_BASE_URL}/job_{table}.parquet"
        row = con.execute("SELECT num_rows FROM parquet_file_metadata(?)", [url]).fetchone()
        assert row is not None
        total = int(row[0])
        rows = total if total <= SMALL_TABLE_ROWS else math.ceil(total * SAMPLE_FRACTION)
        dst = output_dir / f"{table}.parquet"
        print(f"{table}: keeping {rows} of {total} rows", flush=True)
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{url}') LIMIT {rows}) TO '{dst}' "
            "(FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 3)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
