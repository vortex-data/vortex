// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire shapes for `POST /api/ingest`.
//!
//! Each [`Record`] variant deserializes one row destined for one of the five
//! fact tables in [`crate::schema`]. The producer side of the contract lives
//! in `vortex-bench/src/v3.rs` (the `--gh-json-v3` emitter); when changing a
//! shape here, change both sides in the same commit.
//!
//! ## Records are discriminated by `kind`
//!
//! Every record carries a `kind` field that picks one of the five fact
//! tables; serde drives this with `#[serde(tag = "kind", rename_all =
//! "snake_case")]`.
//!
//! | `kind`               | Destination table       |
//! |----------------------|-------------------------|
//! | `query_measurement`  | `query_measurements`    |
//! | `compression_time`   | `compression_times`     |
//! | `compression_size`   | `compression_sizes`     |
//! | `random_access_time` | `random_access_times`   |
//! | `vector_search_run`  | `vector_search_runs`    |
//!
//! Every record struct carries `#[serde(deny_unknown_fields)]`, so unknown
//! fields surface as a `400` with the offending record's index — version
//! skew is supposed to fail loudly. Unknown `kind` values produce the same
//! `400` from the outer enum's tag check.
//!
//! ## Ingest envelope
//!
//! `POST /api/ingest` accepts one [`Envelope`] per request. The envelope
//! wraps a heterogeneous batch of records (any mix of `kind`s):
//!
//! - `run_meta` — [`RunMeta`] with `benchmark_id`, `schema_version`
//!   (must equal [`crate::schema::SCHEMA_VERSION`]), and `started_at`.
//! - `commit` — [`CommitInfo`] with the columns of the `commits` dim table,
//!   keyed by their column names with `commit_sha` renamed to `sha`. The
//!   server upserts this row before applying any record.
//! - `records` — array of per-`kind` records.
//!
//! `vortex-bench --gh-json-v3 <path>` writes JSONL of bare records only —
//! the envelope (`run_meta` + `commit`) is added by the post-ingest script
//! before POSTing, which keeps the Rust emitter dependency-light and lets
//! CI fill the commit fields from `${{ github.sha }}` plus `git show`.

use serde::Deserialize;

/// One ingest payload.
///
/// `run_meta` and `commit` are added by the post-ingest script around the
/// JSONL of bare records the Rust emitter writes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    /// Per-run metadata, including the wire schema version.
    pub run_meta: RunMeta,
    /// Commit context — upserted into `commits` before any record is applied.
    pub commit: CommitInfo,
    /// Heterogeneous batch of fact-table records.
    pub records: Vec<Record>,
}

/// Run-level metadata. `schema_version` is checked against
/// [`crate::schema::SCHEMA_VERSION`] before any record is processed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunMeta {
    /// Free-form ID of the producing run (e.g. `bench.yml@<run_id>`).
    pub benchmark_id: String,
    /// Wire schema version. Must equal [`crate::schema::SCHEMA_VERSION`].
    pub schema_version: i32,
    /// RFC 3339 timestamp at which the run started.
    pub started_at: String,
}

/// Columns for the `commits` dim table. The wire field for `commit_sha` is
/// renamed to `sha` per the contract; every other field name matches the
/// column name in [`crate::schema`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitInfo {
    /// 40-hex lowercase commit SHA.
    pub sha: String,
    /// RFC 3339 / ISO 8601 timestamp of the commit.
    pub timestamp: String,
    /// Full commit message (the server renders only the first line).
    pub message: String,
    /// Author's display name.
    pub author_name: String,
    /// Author's email.
    pub author_email: String,
    /// Committer's display name.
    pub committer_name: String,
    /// Committer's email.
    pub committer_email: String,
    /// Git tree SHA the commit points at.
    pub tree_sha: String,
    /// GitHub URL for the commit (used as the click-through fallback when
    /// no `(#NNNN)` tag is present in the message).
    pub url: String,
}

/// A single ingest record, discriminated by `kind`.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    /// `query_measurement` → `query_measurements` table.
    QueryMeasurement(QueryMeasurement),
    /// `compression_time` → `compression_times` table.
    CompressionTime(CompressionTime),
    /// `compression_size` → `compression_sizes` table.
    CompressionSize(CompressionSize),
    /// `random_access_time` → `random_access_times` table.
    RandomAccessTime(RandomAccessTime),
    /// `vector_search_run` → `vector_search_runs` table.
    VectorSearchRun(VectorSearchRun),
}

/// SQL query suite measurement (TPC-H, ClickBench, ...). Lands in
/// `query_measurements`. Field names match the schema columns; per-suite dim
/// values are documented on
/// [`vortex_bench::v3::benchmark_dataset_dims`](../../../vortex-bench/src/v3.rs).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryMeasurement {
    /// 40-hex lowercase SHA of the producing commit.
    pub commit_sha: String,
    /// Top-level suite (e.g. `tpch`, `clickbench`, `public-bi`).
    pub dataset: String,
    /// Categorical sub-name (Public-BI dataset; ClickBench flavor).
    #[serde(default)]
    pub dataset_variant: Option<String>,
    /// TPC SF as a string. Populated for TPC-H/TPC-DS, NULL elsewhere.
    #[serde(default)]
    pub scale_factor: Option<String>,
    /// Query index within the suite. The convention (0-based or 1-based) is
    /// fixed per suite by the producing bench loop; the migrate classifier
    /// matches it by parsing literal digits out of `q07`-style v2 chart
    /// names.
    pub query_idx: i32,
    /// Storage backend the run targeted: `nvme` or `s3`. Validated on insert.
    pub storage: String,
    /// Engine (`datafusion`, `duckdb`, `vortex`, `arrow`).
    pub engine: String,
    /// On-disk format (`parquet`, `vortex-file-compressed`, `lance`, ...).
    pub format: String,
    /// Median per-iteration wall time in nanoseconds.
    pub value_ns: i64,
    /// Per-iteration wall times in nanoseconds (median of these is `value_ns`).
    pub all_runtimes_ns: Vec<i64>,
    /// Peak resident-set bytes during the query, when memory tracking was on.
    #[serde(default)]
    pub peak_physical: Option<i64>,
    /// Peak virtual-memory bytes during the query, when memory tracking was on.
    #[serde(default)]
    pub peak_virtual: Option<i64>,
    /// Resident-set delta across the query, when memory tracking was on.
    #[serde(default)]
    pub physical_delta: Option<i64>,
    /// Virtual-memory delta across the query, when memory tracking was on.
    #[serde(default)]
    pub virtual_delta: Option<i64>,
    /// Host environment triple (e.g. `x86_64-linux-gnu`).
    #[serde(default)]
    pub env_triple: Option<String>,
}

/// Encode-or-decode timing from `compress-bench`. Lands in
/// `compression_times`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionTime {
    /// 40-hex lowercase SHA of the producing commit.
    pub commit_sha: String,
    /// Compression dataset name.
    pub dataset: String,
    /// Optional dataset variant (reserved; unused at alpha).
    #[serde(default)]
    pub dataset_variant: Option<String>,
    /// On-disk format the timing applies to.
    pub format: String,
    /// `encode` or `decode`. The server treats it as opaque on the wire.
    pub op: String,
    /// Median per-iteration wall time in nanoseconds.
    pub value_ns: i64,
    /// Per-iteration wall times in nanoseconds.
    pub all_runtimes_ns: Vec<i64>,
    /// Host environment triple.
    #[serde(default)]
    pub env_triple: Option<String>,
}

/// On-disk size from `compress-bench`. One-shot, no per-iteration data.
/// Lands in `compression_sizes`. Compression ratios (e.g. `vortex/parquet`)
/// are NOT a separate record kind — they are computed at read time from
/// pairs of these rows.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionSize {
    /// 40-hex lowercase SHA of the producing commit.
    pub commit_sha: String,
    /// Compression dataset name.
    pub dataset: String,
    /// Optional dataset variant (reserved; unused at alpha).
    #[serde(default)]
    pub dataset_variant: Option<String>,
    /// On-disk format the size applies to.
    pub format: String,
    /// Compressed-file size in bytes.
    pub value_bytes: i64,
}

/// Take-time timing from `random-access-bench`. Lands in
/// `random_access_times`. Datasets here (chimp, taxi, ...) are a different
/// namespace from the SQL query suites' dataset names.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomAccessTime {
    /// 40-hex lowercase SHA of the producing commit.
    pub commit_sha: String,
    /// Random-access dataset name.
    pub dataset: String,
    /// On-disk format the timing applies to.
    pub format: String,
    /// Median per-iteration wall time in nanoseconds.
    pub value_ns: i64,
    /// Per-iteration wall times in nanoseconds.
    pub all_runtimes_ns: Vec<i64>,
    /// Host environment triple.
    #[serde(default)]
    pub env_triple: Option<String>,
}

/// Cosine-similarity scan from `vector-search-bench`. Lands in
/// `vector_search_runs`. The only family that emits timing **plus** side
/// counters in the same row.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSearchRun {
    /// 40-hex lowercase SHA of the producing commit.
    pub commit_sha: String,
    /// Vector dataset name (e.g. `cohere-large-10m`).
    pub dataset: String,
    /// Train-split layout label.
    pub layout: String,
    /// Compression flavor label.
    pub flavor: String,
    /// Cosine threshold passed to the scan filter.
    pub threshold: f64,
    /// Median per-scan wall time in nanoseconds.
    pub value_ns: i64,
    /// Per-iteration wall times in nanoseconds.
    pub all_runtimes_ns: Vec<i64>,
    /// Number of rows that survived the cosine filter.
    pub matches: i64,
    /// Total rows scanned.
    pub rows_scanned: i64,
    /// Total on-disk bytes scanned.
    pub bytes_scanned: i64,
    /// Number of timed iterations. Not part of the dim hash.
    pub iterations: i32,
    /// Host environment triple.
    #[serde(default)]
    pub env_triple: Option<String>,
}

impl Record {
    /// The `commit_sha` referenced by this record. Every record carries one;
    /// the server checks the envelope's `commit.sha` matches.
    pub fn commit_sha(&self) -> &str {
        match self {
            Self::QueryMeasurement(r) => &r.commit_sha,
            Self::CompressionTime(r) => &r.commit_sha,
            Self::CompressionSize(r) => &r.commit_sha,
            Self::RandomAccessTime(r) => &r.commit_sha,
            Self::VectorSearchRun(r) => &r.commit_sha,
        }
    }

    /// The wire `kind` string. Useful for logging and error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::QueryMeasurement(_) => "query_measurement",
            Self::CompressionTime(_) => "compression_time",
            Self::CompressionSize(_) => "compression_size",
            Self::RandomAccessTime(_) => "random_access_time",
            Self::VectorSearchRun(_) => "vector_search_run",
        }
    }
}
