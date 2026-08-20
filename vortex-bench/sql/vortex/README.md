# Vortex queries benchmark

A small suite of microbenchmark queries targeting Vortex-specific scan paths, run on
**every PR commit** (unlike the label-gated suites) with a high iteration count, so it is
the most sensitive — and most frequently seen — benchmark comment.

[`init.sql`](./init.sql) generates a 25M-row two-column table; the numbered queries each
pin down one scan behavior:

- [`0_sum-with-filter.sql`](./0_sum-with-filter.sql): a filtered aggregation that forces a
  linear scan today; once statistics are propagated to arrays it should use zone maps
  instead of decoding every row.
- [`1_sum.sql`](./1_sum.sql): an unfiltered aggregation that should eventually be answered
  from footer statistics without decoding data.

## CI variant

CI runs this as the `Vortex queries` PR comment (see
[`.github/workflows/sql-vortex-pr.yml`](../../../.github/workflows/sql-vortex-pr.yml)),
comparing DataFusion and DuckDB over Parquet and Vortex files with 100 iterations.

## Running locally

```bash
vx-bench run vortex --engine datafusion,duckdb --format parquet,vortex
```
