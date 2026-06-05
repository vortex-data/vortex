-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- Initial benchmarks-website schema: the `commits` dim table plus the five
-- fact tables, one per measurement family. This is the Postgres translation of
-- the authoritative DuckDB DDL in `benchmarks-website/server/src/schema.rs`,
-- which the v3 server applies on boot. Column order, nullability, and dim-tuple
-- membership are preserved exactly from the DuckDB shape so that the v3 -> v4
-- row migration and the bit-exact `measurement_id` hash stay valid; the plan's
-- `Out of scope` list forbids changing the schema shape.
--
-- Type translations from DuckDB to Postgres (plan Table C):
--   `DOUBLE`      -> `DOUBLE PRECISION` (the only column-type name that differs).
--   `BIGINT[]`    -> `BIGINT[]` (native on both).
--   `TIMESTAMPTZ`, `TEXT`, `BIGINT`, `INTEGER` are spelled identically.
--
-- Composite indexes follow the read-path filter columns (the chart queries in
-- `benchmarks-website/server/src/api/charts.rs`), NOT the `measurement_id` hash
-- field order in plan Table B. The hash tuple leads with `commit_sha`, but every
-- chart query filters on the dimensional columns and joins to `commits` on
-- `commit_sha`, so an index leading with the dim filter columns is what serves
-- the read path. Uniqueness over the full hash tuple is already enforced by the
-- `measurement_id` primary key.

-- Dim table. `commit_sha` is the PK; `timestamp`, `tree_sha`, and `url` are
-- required (the server cannot render a chart without them); `message` and the
-- author/committer name + email pairs are nullable so v2-imported rows that
-- lacked them survive.
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

-- Every chart resolves its x-axis from `commits` ordered by `timestamp`: the
-- window subquery takes the most recent `n` via `ORDER BY timestamp DESC,
-- commit_sha DESC LIMIT ?`, and the eligible-commits CTE orders ascending. A
-- single DESC index serves the LIMIT directly and is scanned backwards for the
-- ascending case.
CREATE INDEX IF NOT EXISTS idx_commits_timestamp
    ON commits (timestamp DESC, commit_sha DESC);

-- SQL query suite measurements. Natural key: `(commit_sha, dataset,
-- dataset_variant, scale_factor, query_idx, storage, engine, format)`. The four
-- memory columns are populated together or all NULL ("all four or none",
-- enforced by the ingest path, not by a DB constraint).
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

-- The chart query filters on `(dataset, dataset_variant, scale_factor, storage,
-- query_idx)` and joins to `commits`; `engine` and `format` are projected into
-- series tags, not filtered, so they are not part of the index.
CREATE INDEX IF NOT EXISTS idx_query_measurements_chart
    ON query_measurements (dataset, dataset_variant, scale_factor, storage, query_idx);

-- Encode/decode timings from `compress-bench`. Natural key: `(commit_sha,
-- dataset, dataset_variant, format, op)`, where `op IN ('encode', 'decode')`.
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

CREATE INDEX IF NOT EXISTS idx_compression_times_chart
    ON compression_times (dataset, dataset_variant);

-- On-disk sizes from `compress-bench`. One-shot (no per-iteration data, no
-- `all_runtimes_ns`). Natural key: `(commit_sha, dataset, dataset_variant,
-- format)`. Compression ratios are computed at query time, not stored.
CREATE TABLE IF NOT EXISTS compression_sizes (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,
    commit_sha       TEXT        NOT NULL,
    dataset          TEXT        NOT NULL,
    dataset_variant  TEXT,
    format           TEXT        NOT NULL,
    value_bytes      BIGINT      NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_compression_sizes_chart
    ON compression_sizes (dataset, dataset_variant);

-- Take-time timings from `random-access-bench`. Natural key: `(commit_sha,
-- dataset, format)` -- no `dataset_variant`. Its own table so dataset filters
-- never have to disambiguate which suite a row belongs to.
CREATE TABLE IF NOT EXISTS random_access_times (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,
    commit_sha       TEXT        NOT NULL,
    dataset          TEXT        NOT NULL,
    format           TEXT        NOT NULL,
    value_ns         BIGINT      NOT NULL,
    all_runtimes_ns  BIGINT[]    NOT NULL,
    env_triple       TEXT
);

-- Single-column index: the random-access chart query filters on `dataset` only.
CREATE INDEX IF NOT EXISTS idx_random_access_times_chart
    ON random_access_times (dataset);

-- Cosine-similarity scans from `vector-search-bench`. Emits a timing plus side
-- counters (`matches`, `rows_scanned`, `bytes_scanned`) for the same scan.
-- Natural key: `(commit_sha, dataset, layout, flavor, threshold)`. `iterations`
-- is a side count, not part of the dim hash.
CREATE TABLE IF NOT EXISTS vector_search_runs (
    measurement_id   BIGINT           PRIMARY KEY NOT NULL,
    commit_sha       TEXT             NOT NULL,
    dataset          TEXT             NOT NULL,
    layout           TEXT             NOT NULL,
    flavor           TEXT             NOT NULL,
    threshold        DOUBLE PRECISION NOT NULL,
    value_ns         BIGINT           NOT NULL,
    all_runtimes_ns  BIGINT[]         NOT NULL,
    matches          BIGINT           NOT NULL,
    rows_scanned     BIGINT           NOT NULL,
    bytes_scanned    BIGINT           NOT NULL,
    iterations       INTEGER          NOT NULL,
    env_triple       TEXT
);

-- The chart query filters on `(dataset, layout, threshold)`; `flavor` is the
-- series key (projected, not filtered).
CREATE INDEX IF NOT EXISTS idx_vector_search_runs_chart
    ON vector_search_runs (dataset, layout, threshold);
