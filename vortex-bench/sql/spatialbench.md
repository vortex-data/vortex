# SpatialBench benchmark

The [Apache Sedona SpatialBench](https://sedona.apache.org/spatialbench/) geospatial
analytics benchmark: twelve queries (Q1 ... Q12 in [`spatialbench.sql`](./spatialbench.sql),
DuckDB dialect) over a trips/zones schema, exercising spatial predicates and functions such
as `ST_DWithin`, `ST_Intersects`, and `ST_Distance`. The query logic matches upstream
`sedona-spatialbench`; only formatting differs.

The harness lives in [`src/spatialbench`](../src/spatialbench).

## Running locally

```bash
vx-bench run spatialbench
```
