// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB schema applied on server boot — one `commits` dim plus five fact
//! tables, one per measurement family.
//!
//! ## Design principles
//!
//! 1. **One fact table per (dim shape, value shape).** A row in any fact
//!    table has every value column populated; NULLs only appear in genuinely
//!    optional dimensions. The five families have different dim shapes, so
//!    forcing them into one wide table either bloats every row with NULL
//!    columns or splits a single scan's results across multiple rows that
//!    have to be re-joined to render one chart.
//! 2. **No discriminator columns spanning families.** No `metric_kind` enum
//!    forcing the five shapes into one row.
//! 3. **No JSON escape hatch.** New benchmark parameters become real columns.
//!    Adding a nullable column is cheap; the readability win is worth it.
//! 4. **Hashed primary key per fact table.** Every fact table's
//!    `measurement_id` is a deterministic 64-bit hash of `commit_sha` plus
//!    that table's dimensional tuple, computed in
//!    [`crate::db::measurement_id_query`] et al. Including `commit_sha`
//!    makes every (commit, dim) pair a distinct row — that is exactly what
//!    the chart pages render as a time series. Re-emission of the same
//!    (commit, dim) pair is the upsert case. The hash is **server-internal**
//!    and never crosses a process boundary; the wire never carries it.
//! 5. **`commits` is the only dim table.** Engine, format, dataset, etc.
//!    stay as inline strings; DuckDB's dictionary encoding makes a lookup
//!    table pointless.
//! 6. **Ratios are not stored.** Computed at query time from
//!    `compression_sizes`.
//!
//! ## Tables
//!
//! - **`commits`** — dim table. `commit_sha` is the PK. `timestamp`,
//!   `tree_sha`, and `url` are required (the server cannot render a chart
//!   without them); `message` and the author/committer name + email pair are
//!   nullable so v2-imported rows that lacked them survive. Populated on
//!   every `/api/ingest` from the envelope's `commit` block, and on every
//!   migrator run from `commits.json`.
//! - **`query_measurements`** — SQL query suite measurements (TPC-H, TPC-DS,
//!   ClickBench, StatPopGen, PolarSignals, Fineweb, GhArchive, Public-BI).
//!   Natural key: `(commit_sha, dataset, dataset_variant, scale_factor,
//!   query_idx, storage, engine, format)`. Memory columns
//!   (`peak_physical`, `peak_virtual`, `physical_delta`, `virtual_delta`)
//!   are populated together when the run was instrumented for memory and
//!   are NULL otherwise; the ingest path enforces "all four or none".
//!   `dataset_variant` carries a categorical sub-name (Public-BI dataset,
//!   ClickBench flavor); `scale_factor` is the TPC SF as a string.
//! - **`compression_times`** — encode/decode timings from `compress-bench`.
//!   Natural key: `(commit_sha, dataset, dataset_variant, format, op)`,
//!   where `op ∈ {encode, decode}`. Encode and decode share a table because
//!   they share dim and value shape; keeping them together makes the
//!   per-format chart a single SQL query.
//! - **`compression_sizes`** — on-disk sizes from `compress-bench`. One-shot
//!   (no per-iteration data, no `all_runtimes_ns`). Natural key:
//!   `(commit_sha, dataset, dataset_variant, format)`. Compression ratios
//!   (e.g. `vortex:parquet-zstd`) are NOT stored — they are a SELECT over
//!   this table joined to itself, computed in `api/summary.rs`.
//! - **`random_access_times`** — take-time timings from
//!   `random-access-bench`. Different dataset namespace from
//!   `compression_times` (chimp, taxi, etc.) — kept in its own table so
//!   dataset filters never have to disambiguate which suite a row belongs
//!   to. Natural key: `(commit_sha, dataset, format)`.
//! - **`vector_search_runs`** — cosine-similarity scans from
//!   `vector-search-bench`. The only family that emits a timing **plus**
//!   side counters (`matches`, `rows_scanned`, `bytes_scanned`) for the
//!   same scan; keeping them in one row avoids a 1:N split that has to be
//!   re-joined on read. Natural key: `(commit_sha, dataset, layout,
//!   flavor, threshold)`. `iterations` is not part of the dim hash — it is
//!   a side count, like `matches`.
//!
//! ## Column conventions
//!
//! - `commit_sha` is `TEXT NOT NULL` on every fact table and references the
//!   `commits.commit_sha` PK. There is no FK constraint declared at alpha;
//!   the ingest path upserts the commit before the records.
//! - `value_ns` is the median per-iteration nanosecond timing for timing
//!   tables. `value_bytes` is the on-disk byte count for `compression_sizes`.
//! - `all_runtimes_ns BIGINT[]` carries the per-iteration timings inline.
//!   DuckDB's list type avoids a child table; chart code only ever reads
//!   `value_ns`, so the list is effectively cold storage today, kept for
//!   future variance or distribution charts.
//! - `storage` (only on `query_measurements`) is `nvme` or `s3`. Legacy `gcs`
//!   was dropped during the v3 design pass.
//! - `env_triple` is the `arch-os-env` host triple captured at run time
//!   (e.g. `x86_64-linux-gnu`). Optional everywhere; useful for slicing
//!   results by host class once the data set has more than one host class.
//!
//! ## Schema changes
//!
//! There is no migration framework. If you change the schema:
//!
//! 1. Update [`SCHEMA_DDL`] and the matching [`crate::records`] struct.
//! 2. Update or delete any local `bench.duckdb` (the migrator's
//!    `open_target_db` already deletes-and-recreates).
//! 3. Bump [`SCHEMA_VERSION`] if the wire envelope's
//!    `run_meta.schema_version` semantics change.
//!
//! A real forward-only migration framework is post-cutover work.

/// DDL for the `commits` dim plus the five fact tables.
pub const SCHEMA_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS commits (
    commit_sha       TEXT        PRIMARY KEY NOT NULL,
    timestamp        TIMESTAMPTZ NOT NULL,
    message          TEXT,
    author_name      TEXT,
    author_email     TEXT,
    committer_name   TEXT,
    committer_email  TEXT,
    tree_sha         TEXT        NOT NULL,
    url              TEXT        NOT NULL
);

CREATE TABLE IF NOT EXISTS query_measurements (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,
    commit_sha       TEXT        NOT NULL,
    dataset          TEXT        NOT NULL,
    dataset_variant  TEXT,
    scale_factor     TEXT,
    query_idx        INTEGER     NOT NULL,
    storage          TEXT        NOT NULL,
    engine           TEXT        NOT NULL,
    format           TEXT        NOT NULL,
    value_ns         BIGINT      NOT NULL,
    all_runtimes_ns  BIGINT[]    NOT NULL,
    peak_physical    BIGINT,
    peak_virtual     BIGINT,
    physical_delta   BIGINT,
    virtual_delta    BIGINT,
    env_triple       TEXT
);

CREATE TABLE IF NOT EXISTS compression_times (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,
    commit_sha       TEXT        NOT NULL,
    dataset          TEXT        NOT NULL,
    dataset_variant  TEXT,
    format           TEXT        NOT NULL,
    op               TEXT        NOT NULL,
    value_ns         BIGINT      NOT NULL,
    all_runtimes_ns  BIGINT[]    NOT NULL,
    env_triple       TEXT
);

CREATE TABLE IF NOT EXISTS compression_sizes (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,
    commit_sha       TEXT        NOT NULL,
    dataset          TEXT        NOT NULL,
    dataset_variant  TEXT,
    format           TEXT        NOT NULL,
    value_bytes      BIGINT      NOT NULL
);

CREATE TABLE IF NOT EXISTS random_access_times (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,
    commit_sha       TEXT        NOT NULL,
    dataset          TEXT        NOT NULL,
    format           TEXT        NOT NULL,
    value_ns         BIGINT      NOT NULL,
    all_runtimes_ns  BIGINT[]    NOT NULL,
    env_triple       TEXT
);

CREATE TABLE IF NOT EXISTS vector_search_runs (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,
    commit_sha       TEXT        NOT NULL,
    dataset          TEXT        NOT NULL,
    layout           TEXT        NOT NULL,
    flavor           TEXT        NOT NULL,
    threshold        DOUBLE      NOT NULL,
    value_ns         BIGINT      NOT NULL,
    all_runtimes_ns  BIGINT[]    NOT NULL,
    matches          BIGINT      NOT NULL,
    rows_scanned     BIGINT      NOT NULL,
    bytes_scanned    BIGINT      NOT NULL,
    iterations       INTEGER     NOT NULL,
    env_triple       TEXT
);
"#;

/// Schema version expected by the server. The ingest envelope's
/// `run_meta.schema_version` must match this exactly at alpha.
pub const SCHEMA_VERSION: i32 = 1;
