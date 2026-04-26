// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire shapes for `POST /api/ingest`.
//!
//! These types deserialize the ingest envelope defined in
//! `benchmarks-website/planning/02-contracts.md`. Each variant of [`Record`]
//! is gated by `#[serde(deny_unknown_fields)]`, so unknown fields produce
//! a 400 with the offending record's index.

use serde::Deserialize;

/// One ingest payload.
///
/// `run_meta` and `commit` are added by the post-ingest script around the
/// JSONL of bare records the Rust emitter writes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub run_meta: RunMeta,
    pub commit: CommitInfo,
    pub records: Vec<Record>,
}

/// Run-level metadata. `schema_version` is checked against
/// [`crate::schema::SCHEMA_VERSION`] before any record is processed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunMeta {
    pub benchmark_id: String,
    pub schema_version: i32,
    pub started_at: String,
}

/// Columns for the `commits` dim table. The wire field for `commit_sha` is
/// renamed to `sha` per the contract.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitInfo {
    pub sha: String,
    pub timestamp: String,
    pub message: String,
    pub author_name: String,
    pub author_email: String,
    pub committer_name: String,
    pub committer_email: String,
    pub tree_sha: String,
    pub url: String,
}

/// A single ingest record, discriminated by `kind`.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    QueryMeasurement(QueryMeasurement),
    CompressionTime(CompressionTime),
    CompressionSize(CompressionSize),
    RandomAccessTime(RandomAccessTime),
    VectorSearchRun(VectorSearchRun),
}

/// SQL query suite measurement (TPC-H, ClickBench, ...).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryMeasurement {
    pub commit_sha: String,
    pub dataset: String,
    #[serde(default)]
    pub dataset_variant: Option<String>,
    #[serde(default)]
    pub scale_factor: Option<String>,
    pub query_idx: i32,
    pub storage: String,
    pub engine: String,
    pub format: String,
    pub value_ns: i64,
    pub all_runtimes_ns: Vec<i64>,
    #[serde(default)]
    pub peak_physical: Option<i64>,
    #[serde(default)]
    pub peak_virtual: Option<i64>,
    #[serde(default)]
    pub physical_delta: Option<i64>,
    #[serde(default)]
    pub virtual_delta: Option<i64>,
    #[serde(default)]
    pub env_triple: Option<String>,
}

/// Encode/decode timing from `compress-bench`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionTime {
    pub commit_sha: String,
    pub dataset: String,
    #[serde(default)]
    pub dataset_variant: Option<String>,
    pub format: String,
    pub op: String,
    pub value_ns: i64,
    pub all_runtimes_ns: Vec<i64>,
    #[serde(default)]
    pub env_triple: Option<String>,
}

/// On-disk size from `compress-bench`. One-shot, no per-iteration data.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionSize {
    pub commit_sha: String,
    pub dataset: String,
    #[serde(default)]
    pub dataset_variant: Option<String>,
    pub format: String,
    pub value_bytes: i64,
}

/// Take-time timing from `random-access-bench`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomAccessTime {
    pub commit_sha: String,
    pub dataset: String,
    pub format: String,
    pub value_ns: i64,
    pub all_runtimes_ns: Vec<i64>,
    #[serde(default)]
    pub env_triple: Option<String>,
}

/// Cosine-similarity scan from `vector-search-bench`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorSearchRun {
    pub commit_sha: String,
    pub dataset: String,
    pub layout: String,
    pub flavor: String,
    pub threshold: f64,
    pub value_ns: i64,
    pub all_runtimes_ns: Vec<i64>,
    pub matches: i64,
    pub rows_scanned: i64,
    pub bytes_scanned: i64,
    pub iterations: i32,
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
