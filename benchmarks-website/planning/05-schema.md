<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 05 - Proposed DuckDB schema

This is a **proposal**, not a commitment. The goal is to write down the shape
we think we want so we can argue with it. Final column names and types should
be settled when the ingester is being written.

## Design principles

1. **Long form, not pivoted.** One fact table where each row is one measurement
   value. Do not pre-aggregate (e.g. don't store `vortex_time/parquet_time`
   ratios as rows). Derived metrics are SQL views.

2. **Structured dimensions, no magic strings.** The `name` field in today's
   JSONL packs 4-6 dimensions together. v3 has an explicit column for each.

3. **Nullable where honest.** `scale_factor` is null for non-SQL benchmarks.
   `query_idx` is null for compression measurements. Don't fill with sentinel
   values like -1 or "".

4. **Commit as a foreign key.** `commits` is its own table. `measurements`
   refers to it by SHA, not by denormalizing timestamp/author into every row.

5. **Store the raw value, render in the view.** Persist `value_ns` /
   `value_bytes` as integers; convert to ms / MiB in queries or in the
   frontend. Don't bake display units into storage.

## Tables

### `commits`

One row per commit to `develop`.

```sql
CREATE TABLE commits (
    commit_sha       VARCHAR PRIMARY KEY,   -- 40-hex, lower case
    timestamp        TIMESTAMPTZ NOT NULL,  -- parsed from git's iso-strict
    message          VARCHAR NOT NULL,      -- first line of commit msg
    author_name      VARCHAR NOT NULL,
    author_email     VARCHAR NOT NULL,
    committer_name   VARCHAR NOT NULL,
    committer_email  VARCHAR NOT NULL,
    tree_sha         VARCHAR NOT NULL,
    url              VARCHAR NOT NULL       -- github.com commit URL
);
CREATE INDEX commits_ts_idx ON commits(timestamp);
```

### `measurements`

The fact table. One row per (commit x benchmark x target x metric).

```sql
CREATE TABLE measurements (
    measurement_id    BIGINT PRIMARY KEY,   -- auto / hash; see dedup notes

    commit_sha        VARCHAR NOT NULL REFERENCES commits(commit_sha),
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Dimensions ----------------------------------------------------------

    -- High level bucket. One of: 'random_access', 'compression_time',
    -- 'compression_size', 'query_time', 'query_memory'. Small closed set.
    metric_kind       VARCHAR NOT NULL,

    -- The data source for the benchmark (NULL for compression benchmarks).
    -- 'tpch' | 'tpcds' | 'clickbench' | 'public-bi' | 'statpopgen' |
    -- 'polarsignals' | 'fineweb' | 'gharchive'.
    dataset           VARCHAR,

    -- SF for tpch/tpcds, stringified (to preserve "0.1" exactly). NULL
    -- otherwise.
    scale_factor      VARCHAR,

    -- Clickbench flavor ('partitioned' | 'single'), public-bi name, or a
    -- free-form modifier for other datasets. NULL for datasets that have no
    -- modifier.
    dataset_variant   VARCHAR,

    -- Query index (1-22 for TPC-H, 1-99 for TPC-DS, etc). NULL for
    -- non-query benchmarks.
    query_idx         INTEGER,

    -- 'nvme' | 's3' | NULL. NULL means "not applicable" (e.g. in-memory
    -- compression benchmarks).
    storage           VARCHAR,

    -- The "engine" half of the old engine:format target.
    -- 'datafusion' | 'duckdb' | 'vortex' | 'arrow' | 'lance' | ...
    engine            VARCHAR NOT NULL,

    -- The "format" half. 'vortex-file-compressed' | 'parquet' | 'lance' |
    -- 'arrow' | etc. Use the raw name from Target::format.
    format            VARCHAR NOT NULL,

    -- Values --------------------------------------------------------------

    -- Nanoseconds for time measurements, bytes for sizes, NULL if the
    -- measurement is unitless (see value_unitless). Only one of value_ns /
    -- value_bytes / value_unitless is non-NULL per row.
    value_ns          BIGINT,
    value_bytes       BIGINT,
    value_unitless    DOUBLE,

    -- For memory measurements specifically.
    peak_physical     BIGINT,
    peak_virtual      BIGINT,
    physical_delta    BIGINT,
    virtual_delta     BIGINT,

    -- All individual run times for query measurements (NULL for others).
    all_runtimes_ns   BIGINT[],             -- DuckDB LIST<BIGINT>

    -- Run context --------------------------------------------------------

    -- CPU arch / OS / env triple, e.g. 'x86_64-linux-gnu'. Optional.
    env_triple        VARCHAR,

    -- Room to grow without another migration: any emitter-side structured
    -- fields we didn't anticipate. Kept small; don't abuse.
    extra_json        JSON
);

CREATE INDEX measurements_commit_idx ON measurements(commit_sha);
CREATE INDEX measurements_dims_idx   ON measurements(metric_kind, dataset, query_idx, engine, format);
```

**On `measurement_id` and dedup.** An ingester rerun should not double-insert.
Option (a): the id is `hash(commit_sha, metric_kind, dataset, scale_factor,
dataset_variant, query_idx, storage, engine, format)` - deterministic, allows
upsert. Option (b): a UNIQUE constraint on the dimensional key without a
synthetic PK. Pick one; either is fine. See [open questions](./09-open-questions.md).

### `known_engines`, `known_formats`, `known_datasets` (dimension lookup)

Tiny three-column tables: `name`, `display_name`, `color` (for engines/formats).
Populated by the ingester when a new value is seen. Lets the frontend render
colors/labels without the `ENGINE_RENAMES` / `SERIES_COLOR_MAP` tables being
baked into config files.

```sql
CREATE TABLE known_engines (
    name          VARCHAR PRIMARY KEY,
    display_name  VARCHAR NOT NULL,
    color_hex     VARCHAR        -- '#19a508' style, NULL for auto
);
-- same shape for known_formats, known_datasets.
```

These replace the ad-hoc `ENGINE_RENAMES`, `SERIES_COLOR_MAP`, etc. They can be
edited by hand (via SQL or a small admin route) when we want to curate labels.

### `benchmark_groups` (presentation metadata)

Optional. This is the one place where we let "the team curates what users see"
live in the DB rather than in code. A group is a user-visible section (e.g.
"TPC-H (NVMe) (SF=10)"); it is defined by a SQL predicate over `measurements`.

```sql
CREATE TABLE benchmark_groups (
    slug           VARCHAR PRIMARY KEY,    -- 'tpch-nvme-sf10'
    display_name   VARCHAR NOT NULL,       -- 'TPC-H (NVMe) (SF=10)'
    description    VARCHAR,
    category_tags  VARCHAR[],              -- ['Queries (NVMe)', 'TPC-H (SF=10)']
    sort_order     INTEGER NOT NULL,
    -- SQL predicate over measurements, evaluated at query time.
    -- Example: "metric_kind='query_time' AND dataset='tpch' AND
    --          scale_factor='10.0' AND storage='nvme'"
    filter_sql     VARCHAR NOT NULL,
    hidden         BOOLEAN NOT NULL DEFAULT FALSE
);
```

Storing a `filter_sql` string is a choice: it keeps the group definitions data,
not code. The alternative is to keep the `QUERY_SUITES` / `FAN_OUT_GROUPS`
table in Rust and generate groups from it. Either is defensible; the DB-driven
variant is more flexible but gives power to anyone with write access.

We do **not** need separate `chart` / `series` rows in the DB - a chart is
(within a group) `GROUP BY query_idx` (or similar) and a series is `GROUP BY
engine, format`. These aggregations are cheap in DuckDB.

## Views (derived metrics)

Keep these as views to avoid double-bookkeeping.

### `v_measurement_with_commit`

`measurements JOIN commits` so the frontend doesn't have to join every query.

### `v_compression_ratios`

```sql
CREATE VIEW v_compression_ratios AS
SELECT
    a.commit_sha,
    a.dataset,
    a.dataset_variant,
    'vortex_vs_parquet'   AS ratio_name,
    a.value_ns * 1.0 / b.value_ns AS ratio
FROM measurements a
JOIN measurements b USING (commit_sha, dataset, dataset_variant, metric_kind)
WHERE a.metric_kind = 'compression_time'
  AND a.format = 'vortex-file-compressed'
  AND b.format = 'parquet'
-- ...similarly for vortex vs lance, vortex vs raw, etc.
```

Replaces the `"vortex:parquet-zstd ratio compress time/..."` records we
currently *store*.

### `v_latest_per_group`

The latest value per series per group for the summary cards.

## Indexes / performance

DuckDB is columnar and does fine on this data size without indexes. The
indexes above are hints only; in practice we won't need them until we have
orders of magnitude more data than today.

## What's intentionally NOT in this schema

- No `groups/charts/series` normalization. We classify `name` → dimensions at
  ingest; groups/charts/series are derived at render time.
- No pre-downsampled aliases. Downsampling is a query-time concern; DuckDB can
  handle LTTB-style filtering cheaply if it ever becomes a bottleneck.
  (Alternatively, the Leptos server memoizes `(group, chart, downsample_level)`
  responses. See [`09-open-questions.md`](./09-open-questions.md).)
- No `schema_version` column - the ingester is the source of truth for shape,
  and if we evolve the schema we do a `ALTER TABLE` + re-migrate.
- No raw JSON blob of the source record. If we ever need it, we can re-ingest
  from the original `data.json.gz` (kept around as-is, see
  [`06-migration.md`](./06-migration.md)).
