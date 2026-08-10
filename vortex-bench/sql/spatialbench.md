# SpatialBench benchmark

The [Apache Sedona SpatialBench](https://sedona.apache.org/spatialbench/) spatial
analytics benchmark: twelve queries (Q1 ... Q12 in [`spatialbench.sql`](./spatialbench.sql),
DuckDB dialect) over a trips/zones schema, exercising spatial predicates and functions such
as `ST_DWithin`, `ST_Intersects`, and `ST_Distance`. The query logic matches upstream
`sedona-spatialbench`; only formatting differs.

The harness lives in [`src/spatialbench`](../src/spatialbench).

## Running locally

```bash
vx-bench run spatialbench
```

The default command compares the Parquet and Vortex WKB representations with DuckDB. To run the
native Vortex spatial representation explicitly:

```bash
vx-bench run spatialbench --engine duckdb --format vortex-spatial-native
```

To compare all three representations in one run:

```bash
vx-bench run spatialbench --engine duckdb --format parquet,vortex,vortex-spatial-native
```
