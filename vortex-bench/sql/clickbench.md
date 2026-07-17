# ClickBench benchmark

[ClickBench](https://github.com/ClickHouse/ClickBench) is ClickHouse's web-analytics
benchmark: 43 queries over a single wide `hits` table (~100M rows of real-ish traffic
data). It is heavy on aggregations, `GROUP BY`s over high-cardinality columns, and
selective string filters, and is the main "wide table scan" workload in CI.

The queries live in [`clickbench_queries.sql`](./clickbench_queries.sql) (one query per
line, numbered from Q0 in file order). The harness lives in
[`src/clickbench`](../src/clickbench).

## CI variant

CI runs this suite from local NVMe as the `Clickbench on NVME` PR comment, comparing
DataFusion and DuckDB over Parquet, Vortex, and vortex-compact files (plus a native
DuckDB baseline).

## Sorted variant

`Clickbench Sorted on NVME` runs the same table sorted by event time, split into 100
shards whose filenames are shuffled so engines cannot rely on file order, exercising sort
pushdown and zone-map-style pruning on a subset of the queries. See
[`ClickBenchSortedBenchmark`](../src/clickbench/benchmark.rs).

## Running locally

```bash
vx-bench run clickbench --engine datafusion,duckdb --format parquet,vortex
vx-bench run clickbench-sorted --engine datafusion,duckdb --format parquet,vortex
```
