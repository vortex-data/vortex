# PolarSignals Profiling benchmark

A benchmark over continuous-profiling stacktrace data from
[Polar Signals](https://www.polarsignals.com/). The schema features a sparse struct (ten
nullable label fields covering five fill-rate tiers), deeply nested locations
(`List<Struct<..., List<Struct<...>>>>`), and several low-cardinality string columns —
making it the main deeply-nested / sparse-data workload in CI.

The queries in [`polarsignals.sql`](./polarsignals.sql) select stacktrace samples over
increasingly wide time ranges with equality filters on the low-cardinality metadata
columns. The harness lives in [`src/polarsignals`](../src/polarsignals).

## CI variant

CI runs this suite from local NVMe as the `PolarSignals Profiling` PR comment. It runs
DataFusion over Vortex only (there is no Parquet control for this suite, so its geomean is
reported without a drift-corrected verdict).

## Running locally

```bash
vx-bench run polarsignals --engine datafusion --format vortex
```
