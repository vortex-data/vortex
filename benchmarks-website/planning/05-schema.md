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

3. **Nullable where honest.** `scale_factor` is NULL for non-SQL benchmarks.
   `query_idx` is NULL for compression measurements. Don't fill with sentinel
   values like `-1` or `""` - NULL means "not applicable".

4. **Commit as a foreign key.** `commits` is its own table. `measurements`
   refers to it by SHA, not by denormalizing timestamp/author into every row.

5. **Store the raw value, render in the view.** Persist `value_ns` /
   `value_bytes` as integers; convert to ms / MiB in queries or in the
   frontend. Don't bake display units into storage.

6. **Escape hatch for per-benchmark parameters.** Any knob that isn't a
   cross-cutting dimension (query threshold, vector-search layout, criterion
   parameter sweep, etc.) goes in a `data_descriptor JSON` column. Keeps the
   fixed schema narrow; lets new benchmarks land without ALTER TABLE.

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
    -- Deterministic 64-bit hash of the dimensional tuple. Computed by the
    -- ingester; see "Primary key" below. Allows INSERT ... ON CONFLICT DO
    -- UPDATE and makes the row stably addressable across ingester re-runs.
    measurement_id    BIGINT PRIMARY KEY,

    commit_sha        VARCHAR NOT NULL REFERENCES commits(commit_sha),
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Dimensions ----------------------------------------------------------

    -- High level bucket. One of: 'random_access', 'compression_time',
    -- 'compression_size', 'query_time', 'query_memory', 'vector_search',
    -- 'microbench'. Small closed set; add values in code.
    metric_kind       VARCHAR NOT NULL,

    -- The data source for the benchmark. NULL when not applicable (e.g.
    -- compression benchmarks against synthetic data).
    -- 'tpch' | 'tpcds' | 'clickbench' | 'public-bi' | 'statpopgen' |
    -- 'polarsignals' | 'fineweb' | 'gharchive' | 'cohere-large-10m' | ...
    dataset           VARCHAR,

    -- SF for tpch/tpcds, stringified (to preserve "0.1" exactly). NULL for
    -- datasets with no scale factor.
    scale_factor      VARCHAR,

    -- Clickbench flavor, public-bi name, or a free-form modifier. NULL when
    -- the dataset has no sub-variant.
    dataset_variant   VARCHAR,

    -- Query index (1-22 for TPC-H, 1-99 for TPC-DS, etc). NULL for
    -- non-query benchmarks.
    query_idx         INTEGER,

    -- 'nvme' | 's3' | NULL. NULL means "not applicable" (e.g. in-memory
    -- compression benchmarks).
    storage           VARCHAR,

    -- The "engine" half of the old engine:format target.
    -- 'datafusion' | 'duckdb' | 'vortex' | 'arrow' | 'lance' | ...
    -- Non-null for almost all benchmarks; NULL is allowed for future cases
    -- where a benchmark has no meaningful engine.
    engine            VARCHAR,

    -- The "format" half. 'vortex-file-compressed' | 'parquet' | 'lance' |
    -- 'arrow' | 'vortex-turboquant' | etc. Same nullability note as engine.
    format            VARCHAR,

    -- Values --------------------------------------------------------------

    -- Nanoseconds for time measurements, bytes for sizes, value_unitless
    -- for ratios / counts / throughputs. Exactly one of these three SHOULD
    -- be non-NULL per row for non-memory metric kinds (not enforced via
    -- CHECK constraint because rules differ per metric_kind; it's the
    -- ingester's job to keep this consistent).
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

    -- CPU arch / OS / env triple, e.g. 'x86_64-linux-gnu'. NULL if not
    -- captured by the emitter.
    env_triple        VARCHAR,

    -- Benchmark-specific parameters that don't fit the fixed dimensions
    -- above. NULL or '{}' means "no extra params". Examples:
    --
    -- Vector search:
    --   {"layout": "partitioned", "threshold": 0.85, "n_dimensions": 1024,
    --    "n_rows": 10000000, "query_seed": 42, "iterations": 5}
    --
    -- Criterion microbench (future):
    --   {"criterion_group": "encode/u32", "parameter": "1024",
    --    "confidence_interval": [...]}
    --
    -- TPC-H (already has scale_factor column, descriptor can be NULL or {}):
    --   NULL
    data_descriptor   JSON
);

CREATE INDEX measurements_commit_idx ON measurements(commit_sha);
CREATE INDEX measurements_dims_idx   ON measurements(metric_kind, dataset, query_idx, engine, format);
```

#### Primary key

`measurement_id` is a deterministic hash of the dimensional tuple:

```text
measurement_id = xxhash64(
    commit_sha,
    metric_kind,
    dataset       // "NULL" sentinel for absent
    scale_factor,
    dataset_variant,
    query_idx,
    storage,
    engine,
    format,
    canonical_json(data_descriptor)  // sorted keys, stable format
)
```

Hashing tradeoffs and why we use this:

- **Idempotency.** Running the ingester twice against the same input produces
  the same `measurement_id`s, so `INSERT ... ON CONFLICT DO UPDATE` upserts
  cleanly. This is the single most important property - it's what makes the
  migration script (see [`06-migration.md`](./06-migration.md)) safe to re-run
  during development.
- **NULL handling.** Composite UNIQUE constraints have a well-known SQL
  gotcha: `NULL != NULL`, so a UNIQUE over nullable columns doesn't actually
  prevent duplicates where the variant is only-NULLs-match. A hashed PK
  sidesteps this: the ingester decides how to canonicalize NULL into the hash
  input (e.g. a reserved byte sequence), and two rows with the same NULL
  pattern unambiguously hash to the same id.
- **Preserving NULL semantics.** NULL continues to mean "not applicable" in
  the actual column. We don't substitute `""` sentinels; the hash just
  handles NULL-vs-value distinctly.
- **One-column FK surface.** If we ever add a sidecar table (e.g. a separate
  `all_runtimes` table so we don't LIST-pack them inline), the FK is a single
  column.

Cost: `measurement_id` is opaque to humans. You always have to join or look at
the dimension columns to understand what a row is. That's fine - nobody is
querying measurements by id directly.

The ingester's hash function is versioned in the **schema**, not in the data.
If we ever change how it canonicalizes NULL or JSON, we re-run the full
migrator (we keep the raw `data.json.gz` forever for this reason). Hash
algorithm pinned to something stable (xxhash64 or SHA1-truncated - either
works).

### `known_engines`, `known_formats`, `known_datasets` (dimension lookup)

Tiny three-column tables: `name`, `display_name`, `color_hex`. Populated by
the ingester when a new value is seen. Lets the frontend render colors /
labels without the `ENGINE_RENAMES` / `SERIES_COLOR_MAP` tables being baked
into config files.

```sql
CREATE TABLE known_engines (
    name          VARCHAR PRIMARY KEY,
    display_name  VARCHAR NOT NULL,
    color_hex     VARCHAR        -- '#19a508' style, NULL for auto
);
-- same shape for known_formats, known_datasets.
```

These are *pure data*, human-editable. They replace the ad-hoc `ENGINE_RENAMES`,
`SERIES_COLOR_MAP`, `SCALE_FACTOR_DESCRIPTIONS`, etc. tables from v2's
`config.js`. An admin route on the Leptos server (or a plain SQL UPDATE) can
tune labels and colors without a redeploy.

### Group definitions: in Rust, not SQL

Earlier drafts of this doc proposed a `benchmark_groups` table with a
`filter_sql VARCHAR` column that the server would stitch into `WHERE`
clauses at query time. **We are not doing that.** Concatenating stored SQL
into queries is a textbook injection risk, and it silently breaks when
columns are renamed (no compile-time check).

Instead, group definitions live in Rust code - roughly how v2's
`benchmarks-website/src/config.js::QUERY_SUITES` + `BESPOKE_CONFIGS` work
today, but typed and compiled:

```rust
enum BenchmarkGroupFilter {
    RandomAccess,
    Compression,
    CompressionSize,
    QuerySuite { dataset: &'static str },
    FanOut { dataset: &'static str, storage: Storage, scale_factor: &'static str },
    VectorSearch { /* later */ },
    Microbench { /* later */ },
}
```

Each variant compiles into a typed predicate fed to DuckDB via parameters
(never string-interpolated). Adding a new group type is a Rust PR, not a DB
write. Given how rarely group *kinds* change, this is the right trade.

Display strings (group `display_name`, description, category tags) can still
live in a small `benchmark_groups` data table keyed by an enum discriminator
if that's useful - but **no executable content**.

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

Replaces the `"vortex:parquet-zstd ratio compress time/..."` records that are
*stored* in v2.

### `v_latest_per_group`

The latest value per series per group for the summary cards.

## Indexes / performance

DuckDB is columnar and does fine on this data size without indexes. The
indexes above are hints only; in practice we won't need them until we have
orders of magnitude more data than today.

## What's intentionally NOT in this schema

- **No `groups/charts/series` normalization.** We classify raw input →
  dimensions at ingest; groups/charts/series are derived at render time.
- **No pre-downsampled aliases.** Downsampling is a query-time concern; DuckDB
  can handle LTTB-style filtering cheaply if it ever becomes a bottleneck.
  (Alternatively, the Leptos server memoizes `(group, chart, downsample_level)`
  responses.)
- **No `schema_version` column.** The ingester is the source of truth for
  shape. If we evolve the schema we do an `ALTER TABLE` + re-migrate. The raw
  JSONL on S3 stays the canonical backup.
- **No raw JSON blob of the source record.** `data_descriptor` is our
  escape hatch. For the first 6 months post-cutover we *may* want to keep the
  whole raw JSON line too (cheap storage-wise, helps triage surprises); that's
  listed in [`09-open-questions.md`](./09-open-questions.md).

## Extensibility notes

- **Vector-search benchmarks** (`benchmarks/vector-search-bench/`) land with
  `metric_kind IN ('vector_search_time', 'vector_search_count')`, `dataset =
  'cohere-large-10m'` (etc.), `engine = 'vortex'`, `format` = the flavor
  (`vortex-turboquant`), and `data_descriptor` carrying `{layout, threshold,
  n_dimensions, n_rows, iterations}`. No schema change needed.
- **Microbenchmarks** (things under `benches/`, criterion-style) land with
  `metric_kind = 'microbench'` and `data_descriptor` carrying
  `{criterion_group, parameter, ...}`. No schema change needed. We will want
  to think about how they aggregate on the website (one chart per criterion
  bench? summary tables?) when that work comes up; the storage is ready.
