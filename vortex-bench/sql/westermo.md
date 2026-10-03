# Westermo node_exporter metrics benchmark

A benchmark over real Prometheus metrics: the
[Westermo test system performance data set](https://github.com/westermo/test-system-performance-dataset)
by P E Strandberg and Y Marklund, licensed CC BY 4.0. Nineteen servers that drive nightly
testing at Westermo were scraped with node_exporter every 30 seconds for a month. The data set
has 19 CSV files with 22 or 23 metrics each: load averages, memory, CPU, disk, fork, interrupt
and context-switch rates, temperature, and a heartbeat. Three servers have no `sys-thermal`
metric.

The harness downloads the CSVs pinned to upstream commit `47e0ccdc`, then converts them to
Prometheus layout:

- One row per sample, with columns `labels`, `ts` and `value`.
- `labels` is a struct of dictionary-encoded strings: `__name__`, `instance` and `job`.
  Metric names have `-` replaced by `_`, `instance` is the file stem such as `system-7`, and
  `job` is `node`.
- `ts` is int64 milliseconds counted from the start of collection, the type Prometheus uses
  for sample timestamps. The source records offsets, not wall-clock times.
- Rows are sorted by label set and then by time, the order of a compacted Prometheus block.
  Missed scrapes stay missing, so the timestamp column has real gaps.

That gives 434 series and about 37.5 million samples.

The queries in [`westermo.sql`](./westermo.sql), numbered from Q0 in file order, are PromQL
expressions translated to SQL, with the PromQL in a comment above each. They cover single
series range reads, aggregations by label at a fixed step, regex and negated regex label
matchers, one-to-one vector matching, newest-point and newest-N queries, and the label values,
metric names and series metadata APIs. Every query runs unchanged on DataFusion and DuckDB:
time steps are integer arithmetic on `ts`, and regexes are anchored because DuckDB's `~`
matches the whole string. The harness lives in [`src/westermo`](../src/westermo).

The source values were exported from Grafana, so counters such as CPU seconds arrive already
converted to rates. The suite therefore has no `rate()` queries over raw counters.

## CI variant

CI runs this suite only under the `action/bench-sql-extended` label, on DataFusion and DuckDB
over Parquet and Vortex. It does not run on `develop` or under the other benchmark labels.

## Running locally

```bash
vx-bench run westermo --engine datafusion,duckdb --format parquet,vortex
```
