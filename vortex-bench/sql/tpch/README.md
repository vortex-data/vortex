# TPC-H benchmark

The industry-standard [TPC-H](https://www.tpc.org/tpch/) decision-support benchmark: 22
analytical queries over an 8-table warehouse schema (`lineitem`, `orders`, `customer`, ...).
It stresses scan throughput, filter pushdown, joins, and aggregations over mostly numeric
and date columns.

The queries in this directory (`q1.sql` ... `q22.sql`) are the standard TPC-H queries.
Data is generated deterministically by [`tpchgen`](../../src/tpch/tpchgen.rs); the benchmark
harness lives in [`src/tpch`](../../src/tpch).

## CI variants

CI runs this suite at scale factor 1 and 10, reading either from local NVMe or from S3
(the data is uploaded to a bucket first), as the `TPC-H SF={1,10} on {NVME,S3}` PR
comments. Each variant compares DataFusion and DuckDB over Parquet, Vortex, and
vortex-compact files (plus in-memory Arrow and native DuckDB baselines on NVMe).

## Running locally

```bash
vx-bench run tpch --engine datafusion,duckdb --format parquet,vortex --opt scale-factor=1.0
```
