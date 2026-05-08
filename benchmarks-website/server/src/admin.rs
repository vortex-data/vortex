// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Admin endpoints — bearer-gated DuckDB snapshot and read-only SQL.
//!
//! Mounted at `/api/admin/*` only when `ADMIN_BEARER_TOKEN` is set on the
//! server, surfaced through [`crate::app::AppState::with_admin`]. Both routes
//! require an `Authorization: Bearer <ADMIN_BEARER_TOKEN>` header — the
//! `INGEST_BEARER_TOKEN` will not work here, so the two can rotate
//! independently. The operator workflow is documented in
//! `benchmarks-website/ops/README.md`.
//!
//! ## Routes
//!
//! ### `POST /api/admin/snapshot?ts=<id>`
//!
//! Runs `EXPORT DATABASE '<snapshot_dir>/<ts>/' (FORMAT csv)` against the
//! live DuckDB connection. CSV is the only EXPORT format the
//! `bundled` libduckdb-sys feature ships with — switching to parquet or
//! a Vortex layout later means flipping the duckdb feature flag and
//! changing one literal below. CSV round-trips losslessly through
//! `IMPORT DATABASE` (a `schema.sql` is written alongside the data so
//! types and array columns rehydrate correctly).
//!
//! `ts` must match `[A-Za-z0-9_-]{1,64}`; the snapshot script
//! conventionally passes a UTC timestamp like `20260508T010000Z`. The
//! target subdirectory must not already exist (409 otherwise). The export
//! is transactionally consistent: writes during the export queue on the
//! connection mutex.
//!
//! ### `POST /api/admin/sql`
//!
//! Body: `{ "sql": "SELECT ..." }`. Query: `?format=json|table` (default
//! `json`). Only `SELECT`, `WITH`, `PRAGMA`, `SHOW`, `DESCRIBE`, and
//! `EXPLAIN` statements are allowed — anything else is rejected with 403.
//! The connection mutex is held for the duration of the call, so a slow
//! SELECT briefly delays ingest.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use axum::Json;
use axum::extract::Query;
use axum::extract::Request;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header::AUTHORIZATION;
use axum::http::header::CONTENT_TYPE;
use axum::middleware::Next;
use axum::response::IntoResponse;
use axum::response::Response;
use duckdb::Connection;
use duckdb::types::ValueRef;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use subtle::ConstantTimeEq;
use thiserror::Error;

use crate::app::AppState;
use crate::db;

/// Errors surfaced by `/api/admin/*` handlers. Auth (401) is handled by
/// [`require_admin_bearer`] and never reaches a handler.
#[derive(Debug, Error)]
pub enum AdminError {
    /// 400 — request shape is malformed (bad `ts`, bad SQL JSON body, …).
    #[error("bad request: {0}")]
    BadRequest(String),
    /// 403 — request is well-formed but the SQL statement is not on the
    /// read-only allow-list.
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// 409 — snapshot target directory already exists.
    #[error("conflict: {0}")]
    Conflict(String),
    /// 500 — anything else (DB error, IO error, …).
    #[error("internal server error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        let (status, body) = match &self {
            Self::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "bad_request", "message": msg }),
            ),
            Self::Forbidden(msg) => (
                StatusCode::FORBIDDEN,
                json!({ "error": "forbidden", "message": msg }),
            ),
            Self::Conflict(msg) => (
                StatusCode::CONFLICT,
                json!({ "error": "conflict", "message": msg }),
            ),
            Self::Internal(err) => {
                tracing::error!(error = ?err, "admin internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({ "error": "internal" }),
                )
            }
        };
        (status, Json(body)).into_response()
    }
}

/// Axum middleware enforcing the admin bearer token on `/api/admin/*`.
/// 401 if the header is missing, malformed, or wrong; 503 if the server
/// was started without `ADMIN_BEARER_TOKEN` (the admin router is unmounted
/// in that case, so this is just a defensive belt-and-braces check).
pub async fn require_admin_bearer(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, Response> {
    let Some(expected) = state.admin_bearer_token.as_ref() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "admin_not_configured" })),
        )
            .into_response());
    };
    let unauthorized = || {
        (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "unauthorized" })),
        )
            .into_response()
    };
    let header = req
        .headers()
        .get(AUTHORIZATION)
        .ok_or_else(unauthorized)?
        .to_str()
        .map_err(|_| unauthorized())?;
    let presented = header
        .strip_prefix("Bearer ")
        .ok_or_else(unauthorized)?
        .as_bytes();
    if presented.ct_eq(expected.as_bytes()).into() {
        Ok(next.run(req).await)
    } else {
        Err(unauthorized())
    }
}

#[derive(Debug, Deserialize)]
pub struct SnapshotQuery {
    /// Operator-supplied identifier for the snapshot, used as the leaf
    /// directory name. Must match `[A-Za-z0-9_-]{1,64}`.
    pub ts: String,
}

#[derive(Debug, Serialize)]
pub struct SnapshotResponse {
    /// Absolute path of the directory the export landed in.
    pub snapshot_dir: String,
}

/// Handler for `POST /api/admin/snapshot?ts=<id>`. Runs `EXPORT DATABASE`
/// to a fresh subdirectory under [`AppState::snapshot_dir`].
pub async fn snapshot(
    State(state): State<AppState>,
    Query(q): Query<SnapshotQuery>,
) -> Result<impl IntoResponse, AdminError> {
    validate_ts(&q.ts)?;
    let target: PathBuf = state.snapshot_dir.join(&q.ts);
    if target.exists() {
        return Err(AdminError::Conflict(format!(
            "snapshot directory already exists: {}",
            target.display()
        )));
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating snapshot parent {}", parent.display()))?;
    }
    let target_str = target
        .to_str()
        .ok_or_else(|| AdminError::Internal(anyhow::anyhow!("snapshot path is not UTF-8")))?
        .to_string();
    let target_for_response = target.clone();
    db::run_blocking(&state.db, move |conn| {
        // `target_str` is composed from the configured snapshot dir + a
        // validated [A-Za-z0-9_-] timestamp, so single-quote escaping is
        // a non-issue here.
        let sql = format!("EXPORT DATABASE '{target_str}' (FORMAT csv)");
        conn.execute_batch(&sql)
            .with_context(|| format!("EXPORT DATABASE to {target_str}"))
    })
    .await
    .map_err(AdminError::Internal)?;
    Ok(Json(SnapshotResponse {
        snapshot_dir: target_for_response.display().to_string(),
    }))
}

fn validate_ts(ts: &str) -> Result<(), AdminError> {
    if ts.is_empty() || ts.len() > 64 {
        return Err(AdminError::BadRequest("ts must be 1..=64 chars".into()));
    }
    if !ts
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(AdminError::BadRequest(
            "ts must match [A-Za-z0-9_-]+".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct SqlBody {
    pub sql: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct SqlQuery {
    #[serde(default)]
    pub format: SqlFormat,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum SqlFormat {
    /// Returns `{ columns, rows, row_count }` JSON.
    #[default]
    Json,
    /// Returns a `text/plain` ASCII table similar to `duckdb` CLI output.
    Table,
}

/// Handler for `POST /api/admin/sql`.
pub async fn sql(
    State(state): State<AppState>,
    Query(q): Query<SqlQuery>,
    Json(body): Json<SqlBody>,
) -> Result<Response, AdminError> {
    validate_read_only(&body.sql)?;
    let format = q.format;
    let sql_text = body.sql;
    let result = db::run_blocking(&state.db, move |conn| run_select(conn, &sql_text))
        .await
        .map_err(AdminError::Internal)?;
    Ok(match format {
        SqlFormat::Json => Json(json!({
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.rows.len(),
        }))
        .into_response(),
        SqlFormat::Table => (
            [(CONTENT_TYPE, "text/plain; charset=utf-8")],
            format_table(&result),
        )
            .into_response(),
    })
}

fn validate_read_only(sql: &str) -> Result<(), AdminError> {
    let trimmed = sql.trim_start_matches(|c: char| c.is_whitespace() || c == '(' || c == ';');
    let first_word: String = trimmed
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_uppercase();
    const ALLOWED: &[&str] = &["SELECT", "WITH", "PRAGMA", "SHOW", "DESCRIBE", "EXPLAIN"];
    if !ALLOWED.contains(&first_word.as_str()) {
        return Err(AdminError::Forbidden(format!(
            "only {ALLOWED:?} statements are allowed; got {first_word:?}"
        )));
    }
    Ok(())
}

struct QueryResult {
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
}

fn run_select(conn: &Connection, sql: &str) -> Result<QueryResult> {
    let mut stmt = conn.prepare(sql).context("prepare SQL")?;
    let mut rows_iter = stmt.query([]).context("execute SQL")?;
    // duckdb-rs panics on Statement::column_names() if the statement has not
    // executed yet — schema is only populated after `query()` runs. Pull it
    // off the live `Rows` iterator instead.
    let columns: Vec<String> = rows_iter
        .as_ref()
        .map(|s| s.column_names())
        .unwrap_or_default();
    let column_count = columns.len();
    let mut rows: Vec<Vec<Value>> = Vec::new();
    while let Some(row) = rows_iter.next().context("row iter")? {
        let mut out = Vec::with_capacity(column_count);
        for i in 0..column_count {
            let v = row.get_ref(i).context("get col")?;
            out.push(value_ref_to_json(v));
        }
        rows.push(out);
    }
    Ok(QueryResult { columns, rows })
}

fn value_ref_to_json(v: ValueRef<'_>) -> Value {
    match v {
        ValueRef::Null => Value::Null,
        ValueRef::Boolean(b) => Value::Bool(b),
        ValueRef::TinyInt(i) => Value::from(i),
        ValueRef::SmallInt(i) => Value::from(i),
        ValueRef::Int(i) => Value::from(i),
        ValueRef::BigInt(i) => Value::from(i),
        ValueRef::HugeInt(i) => Value::String(i.to_string()),
        ValueRef::UTinyInt(i) => Value::from(i),
        ValueRef::USmallInt(i) => Value::from(i),
        ValueRef::UInt(i) => Value::from(i),
        ValueRef::UBigInt(i) => Value::String(i.to_string()),
        ValueRef::Float(f) => f64::from(f).into(),
        ValueRef::Double(f) => f.into(),
        ValueRef::Text(bytes) => Value::String(String::from_utf8_lossy(bytes).into_owned()),
        ValueRef::Blob(_) => Value::String("<blob>".into()),
        other => Value::String(format!("{other:?}")),
    }
}

fn format_table(r: &QueryResult) -> String {
    if r.columns.is_empty() {
        return "(no columns)\n".into();
    }
    let row_strings: Vec<Vec<String>> = r
        .rows
        .iter()
        .map(|row| row.iter().map(value_display).collect())
        .collect();
    let mut widths: Vec<usize> = r.columns.iter().map(|c| c.chars().count()).collect();
    for row in &row_strings {
        for (i, cell) in row.iter().enumerate() {
            let w = cell.chars().count();
            if w > widths[i] {
                widths[i] = w;
            }
        }
    }
    let mut out = String::new();
    write_separator(&mut out, &widths, '┌', '┬', '┐');
    write_row(&mut out, &r.columns, &widths);
    write_separator(&mut out, &widths, '├', '┼', '┤');
    for row in &row_strings {
        write_row(&mut out, row, &widths);
    }
    write_separator(&mut out, &widths, '└', '┴', '┘');
    let _ = writeln!(
        out,
        "({} row{})",
        r.rows.len(),
        if r.rows.len() == 1 { "" } else { "s" }
    );
    out
}

fn value_display(v: &Value) -> String {
    match v {
        Value::Null => "NULL".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn write_row<S: AsRef<str>>(out: &mut String, cells: &[S], widths: &[usize]) {
    out.push('│');
    for (i, cell) in cells.iter().enumerate() {
        let s = cell.as_ref();
        let pad = widths[i].saturating_sub(s.chars().count());
        out.push(' ');
        out.push_str(s);
        for _ in 0..pad {
            out.push(' ');
        }
        out.push(' ');
        out.push('│');
    }
    out.push('\n');
}

fn write_separator(out: &mut String, widths: &[usize], left: char, mid: char, right: char) {
    out.push(left);
    for (i, w) in widths.iter().enumerate() {
        if i > 0 {
            out.push(mid);
        }
        for _ in 0..(*w + 2) {
            out.push('─');
        }
    }
    out.push(right);
    out.push('\n');
}
