# TSM-Bench

[TSM-Bench](https://github.com/eXascaleInfolab/TSM-Bench) is a benchmark for time-series
database systems, published at VLDB 2023. This suite runs its seven offline queries over its
`d1` dataset.

## Data

`d1` is augmented from a real seed dataset of water temperature and level readings provided
by the Swiss Federal Office for the Environment. It is a single table `d1` with:

- a `time` column sampled every 10 seconds from 2019-03-01 to 2019-04-30,
- an `id_station` column with 10 stations,
- 100 `DOUBLE` sensor columns, `s0` to `s99`.

Rows are ordered by station, then time. The harness downloads the archive from the TSM-Bench
repository at a pinned commit and converts it to Parquet with the DuckDB CLI. It lives in
[`src/tsmbench`](../src/tsmbench).

## Queries

The queries in [`tsm-bench.sql`](./tsm-bench.sql) are TSM-Bench's q1 to q7. The harness
reports them as Q0 to Q6.

| TSM-Bench query | What it does |
|---|---|
| q1 Selection | Raw readings of 10 sensors for one station over one day |
| q2 Filtering | q1 restricted to rows where one sensor exceeds a threshold |
| q3 Aggregation | Average of 10 sensors over one day |
| q4 Downsampling | Hourly averages of 10 sensors over one day |
| q5 Upsampling | Readings filled to a 5-second grid by linear interpolation |
| q6 Average | Row-wise average of two sensors |
| q7 Correlation | Correlation of two sensors |

TSM-Bench fills each query template with random stations, sensors and an anchor time. The
placeholders here come from TSM-Bench's own generator with its default seed and settings:
one station, 10 sensors and a one-day range, taken from the first iteration.

The SQL follows TSM-Bench's TimescaleDB and ClickHouse templates with these changes so the
same text runs on DataFusion and DuckDB:

- `time` is quoted and intervals are written as `INTERVAL '1 day'`.
- q5 uses `LEAD` to add the midpoint between neighbouring readings. TSM-Bench uses each
  engine's own gap filling, which has no portable SQL form.
- q6 follows the ClickHouse template. The TimescaleDB template joins against an unfilled
  placeholder.

## Running locally

```bash
vx-bench run tsm-bench --engine datafusion,duckdb --format parquet,vortex
```

Preparing the data needs `tar` and the DuckDB CLI on `PATH`, about 1.8 GB for the download
and about 5 GB of temporary space for the extracted CSV.
