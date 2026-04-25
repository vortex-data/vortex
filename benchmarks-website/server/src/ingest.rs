// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `POST /api/ingest` handler.
//!
//! All-or-nothing per envelope: every record is upserted in a single DuckDB
//! transaction or none of them are. The reported `inserted`/`updated` counts
//! aggregate across all five fact tables.

use anyhow::Context as _;
use anyhow::Result;
use axum::Json;
use axum::extract::State;
use axum::response::IntoResponse;
use duckdb::Connection;
use duckdb::params;
use serde::Serialize;
use serde_json::Value;

use crate::app::AppState;
use crate::db::measurement_id_compression_size;
use crate::db::measurement_id_compression_time;
use crate::db::measurement_id_query;
use crate::db::measurement_id_random_access;
use crate::db::measurement_id_vector_search;
use crate::db::{self};
use crate::error::IngestError;
use crate::records::CommitInfo;
use crate::records::Envelope;
use crate::records::QueryMeasurement;
use crate::records::Record;
use crate::records::VectorSearchRun;
use crate::schema::SCHEMA_VERSION;

/// Successful ingest response body.
#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub inserted: u64,
    pub updated: u64,
}

/// Handler for `POST /api/ingest`. Bearer auth is enforced by the
/// [`crate::auth::require_bearer`] middleware, not here.
pub async fn handle(
    State(state): State<AppState>,
    body: Json<Value>,
) -> Result<impl IntoResponse, IngestError> {
    let Json(value) = body;
    let envelope: Envelope =
        serde_json::from_value(value).map_err(|e| IngestError::Malformed(e.to_string()))?;
    validate_envelope(&envelope)?;

    let response = db::run_blocking(&state.db, move |conn| apply_envelope(conn, envelope))
        .await
        .map_err(|err| match err.downcast::<IngestError>() {
            Ok(ingest) => ingest,
            Err(other) => IngestError::Internal(other),
        })?;
    Ok(Json(response))
}

fn validate_envelope(env: &Envelope) -> Result<(), IngestError> {
    if env.run_meta.schema_version > SCHEMA_VERSION {
        return Err(IngestError::SchemaVersionTooNew {
            expected: SCHEMA_VERSION,
            got: env.run_meta.schema_version,
        });
    }
    if env.run_meta.schema_version < SCHEMA_VERSION {
        return Err(IngestError::Malformed(format!(
            "schema_version {} is older than server's {}",
            env.run_meta.schema_version, SCHEMA_VERSION
        )));
    }
    Ok(())
}

fn apply_envelope(conn: &mut Connection, env: Envelope) -> Result<IngestResponse> {
    let tx = conn.transaction().context("begin transaction")?;

    upsert_commit(&tx, &env.commit).context("upsert commit")?;

    let mut inserted = 0u64;
    let mut updated = 0u64;
    for (idx, record) in env.records.iter().enumerate() {
        if record.commit_sha() != env.commit.sha {
            return Err(IngestError::Record {
                index: idx,
                message: format!(
                    "record commit_sha {:?} does not match envelope commit.sha {:?}",
                    record.commit_sha(),
                    env.commit.sha,
                ),
            }
            .into());
        }
        match apply_record(&tx, record) {
            Ok(was_update) => {
                if was_update {
                    updated += 1;
                } else {
                    inserted += 1;
                }
            }
            Err(RecordError::Validation(msg)) => {
                return Err(IngestError::Record {
                    index: idx,
                    message: msg,
                }
                .into());
            }
            Err(RecordError::Internal(err)) => {
                return Err(err.context(format!("applying record at index {idx}")));
            }
        }
    }

    tx.commit().context("commit transaction")?;

    Ok(IngestResponse { inserted, updated })
}

fn upsert_commit(tx: &duckdb::Transaction<'_>, c: &CommitInfo) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO commits (
            commit_sha, timestamp, message, author_name, author_email,
            committer_name, committer_email, tree_sha, url
        ) VALUES (?, CAST(? AS TIMESTAMPTZ), ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (commit_sha) DO UPDATE SET
            timestamp       = excluded.timestamp,
            message         = excluded.message,
            author_name     = excluded.author_name,
            author_email    = excluded.author_email,
            committer_name  = excluded.committer_name,
            committer_email = excluded.committer_email,
            tree_sha        = excluded.tree_sha,
            url             = excluded.url
        "#,
        params![
            c.sha,
            c.timestamp,
            c.message,
            c.author_name,
            c.author_email,
            c.committer_name,
            c.committer_email,
            c.tree_sha,
            c.url,
        ],
    )?;
    Ok(())
}

/// Per-record error split: validation failures carry a message that the
/// caller turns into an [`IngestError::Record`] with the right index;
/// anything else bubbles up as a 500.
enum RecordError {
    Validation(String),
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for RecordError {
    fn from(err: anyhow::Error) -> Self {
        Self::Internal(err)
    }
}

impl From<duckdb::Error> for RecordError {
    fn from(err: duckdb::Error) -> Self {
        Self::Internal(err.into())
    }
}

fn apply_record(tx: &duckdb::Transaction<'_>, record: &Record) -> Result<bool, RecordError> {
    match record {
        Record::QueryMeasurement(r) => insert_query_measurement(tx, r),
        Record::CompressionTime(r) => {
            let mid = measurement_id_compression_time(r);
            let was_update = exists(tx, "compression_times", mid)?;
            tx.execute(
                r#"
                INSERT INTO compression_times (
                    measurement_id, commit_sha, dataset, dataset_variant,
                    format, op, value_ns, all_runtimes_ns, env_triple
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS BIGINT[]), ?)
                ON CONFLICT (measurement_id) DO UPDATE SET
                    commit_sha      = excluded.commit_sha,
                    value_ns        = excluded.value_ns,
                    all_runtimes_ns = excluded.all_runtimes_ns,
                    env_triple      = excluded.env_triple
                "#,
                params![
                    mid,
                    r.commit_sha,
                    r.dataset,
                    r.dataset_variant,
                    r.format,
                    r.op,
                    r.value_ns,
                    runtimes_literal(&r.all_runtimes_ns),
                    r.env_triple,
                ],
            )?;
            Ok(was_update)
        }
        Record::CompressionSize(r) => {
            let mid = measurement_id_compression_size(r);
            let was_update = exists(tx, "compression_sizes", mid)?;
            tx.execute(
                r#"
                INSERT INTO compression_sizes (
                    measurement_id, commit_sha, dataset, dataset_variant,
                    format, value_bytes
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (measurement_id) DO UPDATE SET
                    commit_sha   = excluded.commit_sha,
                    value_bytes  = excluded.value_bytes
                "#,
                params![
                    mid,
                    r.commit_sha,
                    r.dataset,
                    r.dataset_variant,
                    r.format,
                    r.value_bytes,
                ],
            )?;
            Ok(was_update)
        }
        Record::RandomAccessTime(r) => {
            let mid = measurement_id_random_access(r);
            let was_update = exists(tx, "random_access_times", mid)?;
            tx.execute(
                r#"
                INSERT INTO random_access_times (
                    measurement_id, commit_sha, dataset, format,
                    value_ns, all_runtimes_ns, env_triple
                ) VALUES (?, ?, ?, ?, ?, CAST(? AS BIGINT[]), ?)
                ON CONFLICT (measurement_id) DO UPDATE SET
                    commit_sha      = excluded.commit_sha,
                    value_ns        = excluded.value_ns,
                    all_runtimes_ns = excluded.all_runtimes_ns,
                    env_triple      = excluded.env_triple
                "#,
                params![
                    mid,
                    r.commit_sha,
                    r.dataset,
                    r.format,
                    r.value_ns,
                    runtimes_literal(&r.all_runtimes_ns),
                    r.env_triple,
                ],
            )?;
            Ok(was_update)
        }
        Record::VectorSearchRun(r) => insert_vector_search(tx, r),
    }
}

fn insert_query_measurement(
    tx: &duckdb::Transaction<'_>,
    r: &QueryMeasurement,
) -> Result<bool, RecordError> {
    if !matches!(r.storage.as_str(), "nvme" | "s3") {
        return Err(RecordError::Validation(format!(
            "storage must be 'nvme' or 's3', got {:?}",
            r.storage
        )));
    }
    if !memory_quartet_consistent(r) {
        return Err(RecordError::Validation(
            "memory fields must be populated together (all four or none)".into(),
        ));
    }
    let mid = measurement_id_query(r);
    let was_update = exists(tx, "query_measurements", mid)?;
    tx.execute(
        r#"
        INSERT INTO query_measurements (
            measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
            query_idx, storage, engine, format,
            value_ns, all_runtimes_ns,
            peak_physical, peak_virtual, physical_delta, virtual_delta,
            env_triple
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS BIGINT[]), ?, ?, ?, ?, ?)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            peak_physical   = excluded.peak_physical,
            peak_virtual    = excluded.peak_virtual,
            physical_delta  = excluded.physical_delta,
            virtual_delta   = excluded.virtual_delta,
            env_triple      = excluded.env_triple
        "#,
        params![
            mid,
            r.commit_sha,
            r.dataset,
            r.dataset_variant,
            r.scale_factor,
            r.query_idx,
            r.storage,
            r.engine,
            r.format,
            r.value_ns,
            runtimes_literal(&r.all_runtimes_ns),
            r.peak_physical,
            r.peak_virtual,
            r.physical_delta,
            r.virtual_delta,
            r.env_triple,
        ],
    )?;
    Ok(was_update)
}

fn insert_vector_search(
    tx: &duckdb::Transaction<'_>,
    r: &VectorSearchRun,
) -> Result<bool, RecordError> {
    let mid = measurement_id_vector_search(r);
    let was_update = exists(tx, "vector_search_runs", mid)?;
    tx.execute(
        r#"
        INSERT INTO vector_search_runs (
            measurement_id, commit_sha, dataset, layout, flavor, threshold,
            value_ns, all_runtimes_ns, matches, rows_scanned, bytes_scanned,
            iterations, env_triple
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS BIGINT[]), ?, ?, ?, ?, ?)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            matches         = excluded.matches,
            rows_scanned    = excluded.rows_scanned,
            bytes_scanned   = excluded.bytes_scanned,
            iterations      = excluded.iterations,
            env_triple      = excluded.env_triple
        "#,
        params![
            mid,
            r.commit_sha,
            r.dataset,
            r.layout,
            r.flavor,
            r.threshold,
            r.value_ns,
            runtimes_literal(&r.all_runtimes_ns),
            r.matches,
            r.rows_scanned,
            r.bytes_scanned,
            r.iterations,
            r.env_triple,
        ],
    )?;
    Ok(was_update)
}

fn exists(tx: &duckdb::Transaction<'_>, table: &str, mid: i64) -> Result<bool, RecordError> {
    // Table name is from a closed enum of literals above, never user input.
    let sql = format!("SELECT 1 FROM {table} WHERE measurement_id = ? LIMIT 1");
    let mut stmt = tx.prepare(&sql)?;
    let exists = stmt.exists(params![mid])?;
    Ok(exists)
}

fn runtimes_literal(values: &[i64]) -> String {
    let mut s = String::with_capacity(values.len() * 8 + 2);
    s.push('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
    s
}

fn memory_quartet_consistent(r: &QueryMeasurement) -> bool {
    let any = r.peak_physical.is_some()
        || r.peak_virtual.is_some()
        || r.physical_delta.is_some()
        || r.virtual_delta.is_some();
    let all = r.peak_physical.is_some()
        && r.peak_virtual.is_some()
        && r.physical_delta.is_some()
        && r.virtual_delta.is_some();
    !any || all
}
