#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# One-time extraction for the `fsst_view_fineweb_queries` benchmark.
#
# Materializes, from the real HuggingFace FineWeb 10BT sample (the same file `vortex-bench` uses),
# the `url` and `text` string columns plus a per-row selection mask for each real benchmark
# predicate. Writes them as the simple length-prefixed / byte-per-row formats the bench reads.
#
#   pip install duckdb
#   python3 fineweb_queries_extract.py            # -> /tmp/fw_url.bin, fw_text.bin, fw_mask_*.bin
#   FINEWEB_DIR=/tmp cargo bench -p vortex-fsst --bench fsst_view_fineweb_queries
#
# The sample is ~2 GB; DuckDB streams it over HTTP range reads, so only the first N rows are read.

import os
import struct

import duckdb

SRC = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/v1.4.0/sample/10BT/001_00000.parquet"
N = 200_000
OUT = os.environ.get("FINEWEB_DIR", "/tmp")

# The row-selecting `WHERE` clauses of the `SELECT *` FineWeb queries in vortex-bench.
# (`file_path LIKE '%/CC-MAIN-2014-%'` matches zero rows in this sample, so it is omitted.)
QUERIES = {
    "dump_eq": "dump = 'CC-MAIN-2016-30'",
    "date_prefix": "date LIKE '2020-10-%'",
    "google_and": "url LIKE '%google%' AND text LIKE '%Google%'",
    "google_or": "url LIKE '%.google.%' OR text LIKE '% Google %'",
    "vortex": "text LIKE '% vortex %'",
    "espn_and": "url LIKE '%espn%' AND language = 'en' AND language_score > 0.92",
    "espn_or": "url LIKE '%espn%' OR url LIKE '%www.espn.go.com%' OR url LIKE '%espn.go.com%'",
}


def main() -> None:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(
        f"""CREATE TABLE fw AS
            SELECT row_number() OVER () AS rid, url, text, dump, date, file_path,
                   language, language_score
            FROM read_parquet('{SRC}') LIMIT {N}"""
    )

    def dump_col(col: str) -> None:
        rows = con.execute(f"SELECT {col} FROM fw ORDER BY rid").fetchall()
        path = os.path.join(OUT, f"fw_{col}.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(rows)))
            for (v,) in rows:
                b = (v or "").encode("utf-8")
                f.write(struct.pack("<I", len(b)))
                f.write(b)
        print(f"{col}: {len(rows)} rows -> {path}")

    dump_col("url")
    dump_col("text")

    for name, pred in QUERIES.items():
        rids = {r[0] for r in con.execute(f"SELECT rid FROM fw WHERE {pred}").fetchall()}
        path = os.path.join(OUT, f"fw_mask_{name}.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", N))
            f.write(bytes(1 if (i + 1) in rids else 0 for i in range(N)))
        print(f"mask {name}: kept {len(rids)} ({100 * len(rids) / N:.2f}%)")


if __name__ == "__main__":
    main()
