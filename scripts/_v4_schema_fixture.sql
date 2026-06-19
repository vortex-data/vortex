-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- TEST-ONLY fixture mirroring vortex-data/benchmarks-website migrations/001_initial_schema.sql
-- (plus the query_measurements.commit_timestamp column added by 006_read_path_perf.sql and the
-- covering index from 007_summary_covering_index.sql). Drift from the website repo is managed
-- by the SCHEMA_VERSION / column-list cross-repo contract; the role/grant migrations (002-005)
-- are omitted because the test suite connects as the container superuser (no roles needed).

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
-- enforced by the ingest path, not by a DB constraint). `commit_timestamp` is a
-- denormalized copy of `commits.timestamp` (added by migration 006) used by the
-- latest-per-series summary index scan; nullable so existing rows survive the
-- migration without a backfill precondition.
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
    env_triple       TEXT,
    commit_timestamp TIMESTAMPTZ
);

-- The chart query filters on `(dataset, dataset_variant, scale_factor, storage,
-- query_idx)` and joins to `commits`; `engine` and `format` are projected into
-- series tags, not filtered, so they are not part of the index.
CREATE INDEX IF NOT EXISTS idx_query_measurements_chart
    ON query_measurements (dataset, dataset_variant, scale_factor, storage, query_idx);

-- Covering index for the latest-per-series summary (migration 007): resolves
-- `DISTINCT ON (query_idx, engine, format) ORDER BY commit_timestamp DESC` to an
-- Index Only Scan.
CREATE INDEX IF NOT EXISTS idx_query_measurements_summary
    ON query_measurements (dataset, dataset_variant, scale_factor, storage,
                           query_idx, engine, format, commit_timestamp DESC)
    INCLUDE (value_ns);

-- Low-cardinality indexes backing the loose-index scan in `collectFilterUniverse`
-- (migration 006).
CREATE INDEX IF NOT EXISTS idx_query_measurements_engine
    ON query_measurements (engine);
CREATE INDEX IF NOT EXISTS idx_query_measurements_format
    ON query_measurements (format);

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

CREATE INDEX IF NOT EXISTS idx_compression_times_format
    ON compression_times (format);

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

CREATE INDEX IF NOT EXISTS idx_compression_sizes_format
    ON compression_sizes (format);

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

CREATE INDEX IF NOT EXISTS idx_random_access_times_format
    ON random_access_times (format);

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
