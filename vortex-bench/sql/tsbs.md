# TSBS

The [Time Series Benchmark Suite](https://github.com/timescale/tsbs) (TSBS) is the benchmark
most time-series databases publish results on. This suite runs its `cpu-only` use case.

## Data

TSBS simulates CPU metrics for a fleet of hosts. The data is a single table `cpu` with:

- a `time` column with one reading per host every 10 seconds,
- ten host tags such as `hostname`, `region` and `datacenter`, stored as strings,
- ten CPU usage metrics such as `usage_user` and `usage_idle`, stored as doubles.

Rows are ordered by time, with hosts interleaved. The suite uses the configuration from the
TSBS README: 4,000 hosts over three days from 2016-01-01, seed 123, which is 103,680,000 rows.

The harness runs `tsbs_generate_data` at a pinned commit through `go run` and converts its
output to Parquet. It types columns the way TSBS's TimescaleDB loader does: tags as text and
metrics as double precision. It lives in [`src/tsbs`](../src/tsbs).

## Queries

[`tsbs.sql`](./tsbs.sql) holds one query of each of the 15 DevOps query types, in this order:

| Query type | What it does |
|---|---|
| `single-groupby-1-1-1` | Max of 1 metric for 1 host, per minute, over 1 hour |
| `single-groupby-1-1-12` | Max of 1 metric for 1 host, per minute, over 12 hours |
| `single-groupby-1-8-1` | Max of 1 metric for 8 hosts, per minute, over 1 hour |
| `single-groupby-5-1-1` | Max of 5 metrics for 1 host, per minute, over 1 hour |
| `single-groupby-5-1-12` | Max of 5 metrics for 1 host, per minute, over 12 hours |
| `single-groupby-5-8-1` | Max of 5 metrics for 8 hosts, per minute, over 1 hour |
| `cpu-max-all-1` | Max of all metrics for 1 host, per hour, over 8 hours |
| `cpu-max-all-8` | Max of all metrics for 8 hosts, per hour, over 8 hours |
| `double-groupby-1` | Mean of 1 metric per host and hour, over 12 hours |
| `double-groupby-5` | Mean of 5 metrics per host and hour, over 12 hours |
| `double-groupby-all` | Mean of all metrics per host and hour, over 12 hours |
| `high-cpu-all` | All readings above a threshold, all hosts, over 12 hours |
| `high-cpu-1` | All readings above a threshold, 1 host, over 12 hours |
| `lastpoint` | The last reading for each host |
| `groupby-orderby-limit` | The last 5 per-minute maxima before a point in time |

The queries were produced by `tsbs_generate_queries` at the same pinned commit, with seed 123
and the data configuration above, in its TimescaleDB format without a tags table. They were
then changed in three ways so the same text runs on DataFusion and DuckDB:

- `time_bucket('60 seconds', time)` and `time_bucket('3600 seconds', time)` became
  `date_trunc('minute', time)` and `date_trunc('hour', time)`, which give the same buckets.
- Time literals with a `+0000` offset became `TIMESTAMP` literals, since the data is in UTC.
- The `time` column is quoted.

## Running locally

```bash
vx-bench run tsbs --engine datafusion,duckdb --format parquet,vortex
```

Preparing the data needs a Go toolchain on `PATH` and network access to the Go module proxy.
