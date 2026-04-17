# Benchmark Website Schema

This file is the reference for the **normalized DuckDB schema** used by the benchmark website.

For the upstream artifact generation, refresh pipeline, and table-to-table flow, see
[ETL.md](./ETL.md).

The implementation lives primarily in [`store/sql.js`](./store/sql.js),
[`store/metadata.js`](./store/metadata.js), and [`duckdb-store.js`](./duckdb-store.js).
For DuckDB's NDJSON reader semantics, see:
<https://duckdb.org/docs/stable/data/json/loading_json>.

## Core Fact Model

The important point is that the website does **not** store nested charts directly. It stores a
long-form fact table, `benchmark_points_active`, keyed in practice by:

```sql
(group_name, chart_name, series_name, commit_idx)
```

Everything else is metadata, lookup/config data, or a projection over that fact table.

## Configuration Tables

These are synthesized from [`src/config.js`](./src/config.js), not loaded from the JSON artifacts.

```sql
create table query_suites (
  prefix         varchar,
  display_name   varchar,
  query_prefix   varchar,
  dataset_key    varchar,
  fan_out        boolean,
  skip           boolean
);

create table valid_groups (
  group_name     varchar primary key
);

create table engine_renames (
  src            varchar primary key,
  dst            varchar
);
```

## Raw Views

These are views over the cached artifact files.

```sql
create view raw_commits as
select
  id,
  message,
  timestamp,
  author,
  url
from read_json(...);

create view raw_benchmarks as
select
  row_number() over () as benchmark_row,
  name,
  unit,
  value,
  storage,
  dataset,
  commit,
  commit_id
from read_json(...);
```

## Stage 1: Commit Ordering and Name Parsing

```sql
create table commit_dim (
  commit_idx      bigint primary key,
  id              varchar unique,
  message         varchar,
  timestamp_text  varchar,
  commit_ts       timestamptz,
  author          varchar,
  url             varchar
);

create table benchmarks_base (
  benchmark_row   bigint primary key,
  name            varchar,
  name_lower      varchar,
  part1           varchar,
  part1_lower     varchar,
  part2           varchar,
  part3           varchar,
  part4           varchar,
  part_count      bigint,
  unit            varchar,
  raw_value       double,
  storage         varchar,
  dataset_json    json,
  commit_json     json,
  commit_id       varchar
);

create table matched_suites (
  benchmark_row   bigint primary key,
  prefix          varchar,
  display_name    varchar,
  query_prefix    varchar,
  dataset_key     varchar,
  fan_out         boolean
);
```

Notes:

- `commit_dim` orders commits by parsed timestamp, then `id`.
- `benchmarks_base` is the normalized string-parsing layer over benchmark `name`.
- `matched_suites` assigns a benchmark row to the longest matching configured suite prefix.

## Stage 2: Classification

```sql
create table classified_benchmarks (
  benchmark_row       bigint primary key,
  name                varchar,
  resolved_commit_id  varchar,
  group_name          varchar,
  chart_name          varchar,
  series_name         varchar,
  sort_position       integer,
  unit                varchar,
  value               double
);
```

This is the semantic classification step:

- `resolved_commit_id` is `coalesce(commit.id, commit_id)`
- `group_name` is inferred from benchmark name prefixes, query suite config, storage, and scale
  factor metadata
- `chart_name` and `series_name` come from benchmark-name parsing
- `series_name` is passed through `engine_renames`
- `unit` and `value` are normalized for display:
  - `ns -> ms/iter`
  - `bytes -> MiB`

In other words, the benchmark-name grammar is part of the schema.

## Stage 3: Time-Series Fact Tables

```sql
create table benchmark_points (
  group_name      varchar,
  chart_name      varchar,
  series_name     varchar,
  sort_position   integer,
  unit            varchar,
  value           double,
  commit_idx      bigint
);

create table active_commits (
  original_commit_idx  bigint primary key,
  commit_idx           bigint unique,
  id                   varchar,
  message              varchar,
  timestamp            varchar,
  author               varchar,
  url                  varchar
);

create table benchmark_points_active (
  group_name      varchar,
  chart_name      varchar,
  series_name     varchar,
  sort_position   integer,
  unit            varchar,
  value           double,
  commit_idx      bigint
);
```

Notes:

- `benchmark_points` is the first fact table after commit resolution and group filtering.
- `active_commits` trims leading commits with no benchmark data and re-bases the visible commit
  axis to `0..N-1`.
- `benchmark_points_active` is the canonical fact table used for charting and summaries.

## Stage 4: Chart Metadata and Latest-Value Projections

```sql
create table chart_defs (
  group_name      varchar,
  chart_name      varchar,
  sort_position   integer,
  unit            varchar,
  primary key (group_name, chart_name)
);

create table chart_latest_idx (
  group_name        varchar,
  chart_name        varchar,
  latest_commit_idx bigint,
  primary key (group_name, chart_name)
);

create table chart_latest_values (
  group_name      varchar,
  chart_name      varchar,
  series_name     varchar,
  value           double
);

create table chart_series_latest_values (
  group_name      varchar,
  chart_name      varchar,
  series_name     varchar,
  latest_value    double,
  primary key (group_name, chart_name, series_name)
);
```

There are two different "latest" concepts:

- `chart_latest_values`
  values at the chart's latest commit index
- `chart_series_latest_values`
  latest non-null value seen for each series, via `arg_max(value, commit_idx)`

## API-Facing Reads

### `/api/metadata`

Built in [`store/metadata.js`](./store/metadata.js).

Primary sources:

- `active_commits`
- `chart_defs`
- `chart_series_latest_values`
- `chart_latest_values`
- `benchmark_points_active`

Logical output shape:

```sql
metadata (
  total_commits bigint,
  last_updated  timestamptz,
  commits       json,
  groups        json
)
```

Each group contains:

```sql
group_metadata (
  charts       json,
  total_charts bigint,
  has_data     boolean,
  summary      json
)
```

Summary sections are computed from SQL aggregates:

- random access ranking from one anchor chart
- compression geomeans from Vortex/Parquet ratio charts
- compression-size min/geomean/max
- query-suite geomeans with a missing-query penalty

### `/api/data/:group/:chart`

Built in [`duckdb-store.js`](./duckdb-store.js).

The query:

1. selects the requested commit range from `active_commits`
2. enumerates the series present for that chart
3. cross joins series against commits
4. left joins `benchmark_points_active`
5. aggregates with `list(value order by commit_idx)`

That produces a dense matrix with explicit `null` gaps:

```sql
series_points (
  series_name varchar,
  values      list<double>
)
```

The server then serializes the commit slice, applies downsampling if needed, and returns
`{ commits, series, requestedRange, downsampleLevel, ... }`.

## Practical Notes

- The stable fact table is `benchmark_points_active`.
- `commit_idx` is a presentation index, not the original Git ordering key.
- The website schema is intentionally lossy relative to the raw artifacts; it keeps only the
  columns needed for grouping, charting, ordering, and summary computation.
