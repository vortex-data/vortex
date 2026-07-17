# Appian benchmark

Mirrors the queries from DuckDB's in-tree
[`appian_benchmarks`](https://github.com/duckdb/duckdb/tree/main/benchmark/appian_benchmarks)
suite — a real-world business-application workload (customers, orders, addresses) with
selective filters and joins over camelCase-named columns.

The eight queries live in this directory (`q1.sql` ... `q8.sql`). Upstream ships the data
as a single `.duckdb` blob (~593 MB); the harness downloads it once and projects each
table into Parquet with lowercased column names so that DataFusion and DuckDB resolve the
verbatim queries identically — see [`src/appian`](../../src/appian) for the details.

## CI variant

CI runs this suite from local NVMe as the `Appian on NVME` PR comment, comparing
DataFusion and DuckDB over Parquet and Vortex files (plus a native DuckDB baseline).

## Running locally

```bash
vx-bench run appian --engine datafusion,duckdb --format parquet,vortex
```
