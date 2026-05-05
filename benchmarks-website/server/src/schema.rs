// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB schema DDL applied on server boot.
//!
//! See `benchmarks-website/planning/01-schema.md` for the column contracts.
//! There is no migration framework at alpha: if the schema changes, delete
//! the DuckDB file and restart.

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
