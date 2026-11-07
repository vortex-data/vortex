# TPC-DS benchmark

The industry-standard [TPC-DS](https://www.tpc.org/tpcds/) decision-support benchmark: 99
analytical queries (`01.sql` ... `99.sql`) over a retail snowflake schema. Compared to
TPC-H it has many more queries, a wider schema, and heavier use of joins, subqueries, and
window functions, making it a broad regression net for scan and pushdown behavior.

The benchmark harness lives in [`src/tpcds`](../../src/tpcds).

## CI variant

CI runs this suite at scale factor 1 from local NVMe as the `TPC-DS SF=1 on NVME` PR
comment, comparing DataFusion and DuckDB over Parquet, Vortex, and vortex-compact files
(plus a native DuckDB baseline).

## Running locally

```bash
vx-bench run tpcds --engine datafusion,duckdb --format parquet,vortex --opt scale-factor=1.0
```

## Clustered variant

`--opt tpcds-cluster=true` selects a copy of the data whose fact tables (`store_sales`,
`catalog_sales`, `web_sales`, the three returns tables, and `inventory`) are sorted by their
date key and then their item key, nulls last, so zone maps and dynamic join filters on those
keys can prune. It lives in `data/tpcds/<scale-factor>-clustered/`, next to the unclustered
data, and uses the same files and row group sizes. Pass the same option to `data-gen` and to
the benchmark:

```bash
cargo run --release -p vortex-bench --bin data-gen -- tpcds --formats parquet,vortex \
    --opt scale-factor=10,tpcds-cluster=true
cargo run --release -p datafusion-bench -- tpcds --formats parquet,vortex \
    --opt scale-factor=10,tpcds-cluster=true
```

The sort is external: runs of about 1 GiB are sorted in memory, spilled as LZ4 Arrow IPC to a
temporary directory in the output directory, and merged, so clustering needs no more memory at
larger scale factors, but needs about the table's uncompressed size in free disk.
