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
