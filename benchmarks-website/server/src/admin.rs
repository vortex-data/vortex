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
//! Writes a snapshot directory `<snapshot_dir>/<ts>/` containing:
//! - `schema.sql` — verbatim copy of [`crate::schema::SCHEMA_DDL`], so a
//!   restore knows how to recreate the tables before bulk-loading.
//! - `<table>.vortex` for every table in [`crate::schema::TABLES`] —
//!   each produced by a `COPY (SELECT * FROM <table>) TO …
//!   (FORMAT vortex)`. The vortex DuckDB extension is auto-installed
//!   from the community repo on first call, then `LOAD`ed.
//!
//! Vortex compresses the BIGINT[] runtime arrays and string columns
//! roughly an order of magnitude better than gzipped CSV on this shape;
//! it is also the project's own format, which is the obvious dogfood.
//!
//! `ts` must match `[A-Za-z0-9_-]{1,64}`; the snapshot script
//! conventionally passes a UTC timestamp like `20260508T010000Z`. The
//! target subdirectory must not already exist (409 otherwise). All
//! per-table COPY statements run on a connection cloned from the
//! shared handle, so concurrent ingest writes are not blocked.
//!
//! ### `POST /api/admin/sql`
//!
//! Body: `{ "sql": "SELECT ..." }`. Query: `?format=json|table` (default
//! `json`). Only `SELECT`, `WITH`, `PRAGMA`, `SHOW`, `DESCRIBE`, and
//! `EXPLAIN` statements are allowed — anything else is rejected with 403.
//! The connection mutex is held for the duration of the call, so a slow
//! SELECT briefly delays ingest.

use std::fmt::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

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
use crate::schema;

const ADMIN_SQL_ROW_LIMIT: usize = 10_000;

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

/// Handler for `POST /api/admin/snapshot?ts=<id>`. Writes
/// `schema.sql` plus one `<table>.vortex` file per fact/dim table into
/// a fresh subdirectory under [`AppState::snapshot_dir`].
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

    let tmp = tmp_snapshot_dir(&target, &q.ts);
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)
            .with_context(|| format!("removing stale temp snapshot dir {}", tmp.display()))?;
    }
    std::fs::create_dir_all(&tmp)
        .with_context(|| format!("creating temp snapshot dir {}", tmp.display()))?;

    let result = write_snapshot(&state, &tmp).await;
    if let Err(err) = result {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(AdminError::Internal(err));
    }
    // Re-check `target` immediately before the rename. `std::fs::rename` on
    // Linux overwrites an existing destination atomically, so without this
    // guard two concurrent calls with the same `ts` would both finish and
    // the second `rename` would silently clobber the first snapshot. A small
    // theoretical window remains between this check and the rename itself,
    // but it closes the practical race and needs no platform-specific code.
    if target.exists() {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(AdminError::Conflict(format!(
            "snapshot directory already exists: {}",
            target.display()
        )));
    }
    if let Err(err) = std::fs::rename(&tmp, &target).with_context(|| {
        format!(
            "moving snapshot dir {} to {}",
            tmp.display(),
            target.display()
        )
    }) {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(AdminError::Internal(err));
    }
    Ok(Json(SnapshotResponse {
        snapshot_dir: target.display().to_string(),
    }))
}

/// Per-call unique temp directory used to stage a snapshot before the atomic
/// rename onto `target`. Includes a process-local counter so two concurrent
/// calls with the same `ts` in the same server process never share a staging
/// directory and clobber each other's in-progress writes.
fn tmp_snapshot_dir(target: &Path, ts: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    target.with_file_name(format!("{ts}.tmp-{}-{}", std::process::id(), id))
}

async fn write_snapshot(state: &AppState, target: &Path) -> Result<()> {
    // Schema is just our DDL string verbatim; restore reads this with
    // `duckdb -init schema.sql` (or `.read schema.sql`) before
    // bulk-loading the per-table vortex files.
    std::fs::write(target.join("schema.sql"), schema::SCHEMA_DDL)
        .with_context(|| format!("writing schema.sql under {}", target.display()))?;

    let target_for_db = target.to_path_buf();
    db::run_blocking(&state.db, move |conn| {
        export_snapshot_tables(conn, &target_for_db)
    })
    .await
}

fn export_snapshot_tables(conn: &mut Connection, target: &Path) -> Result<()> {
    // Idempotent — `INSTALL` is a no-op if the extension is already
    // present, `LOAD` is cheap once the binary is on disk. The
    // bundled libduckdb-sys has autoload enabled, so the very first
    // call also auto-fetches the extension from the DuckDB
    // community repo. Subsequent calls are entirely local.
    conn.execute_batch("INSTALL vortex FROM community; LOAD vortex;")
        .context("INSTALL/LOAD vortex extension")?;
    for table in schema::TABLES {
        let path = target.join(format!("{table}.vortex"));
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("snapshot path is not UTF-8: {}", path.display()))?;
        let sql = format!(
            "COPY (SELECT * FROM {table}) TO {} (FORMAT vortex)",
            sql_string_literal(path_str)
        );
        conn.execute_batch(&sql)
            .with_context(|| format!("COPY {table} TO {path_str}"))?;
    }
    Ok(())
}

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
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
            "truncated": result.truncated,
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
    ensure_single_statement(sql)?;
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

fn ensure_single_statement(sql: &str) -> Result<(), AdminError> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Normal,
        SingleQuote,
        DoubleQuote,
        LineComment,
        BlockComment,
    }

    let mut state = State::Normal;
    let mut chars = sql.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        match state {
            State::Normal => match ch {
                '\'' => state = State::SingleQuote,
                '"' => state = State::DoubleQuote,
                '-' if chars.peek().is_some_and(|(_, next)| *next == '-') => {
                    chars.next();
                    state = State::LineComment;
                }
                '/' if chars.peek().is_some_and(|(_, next)| *next == '*') => {
                    chars.next();
                    state = State::BlockComment;
                }
                ';' if !sql[idx + ch.len_utf8()..].trim().is_empty() => {
                    return Err(AdminError::Forbidden(
                        "admin SQL accepts a single statement only".into(),
                    ));
                }
                _ => {}
            },
            State::SingleQuote => {
                if ch == '\'' {
                    if chars.peek().is_some_and(|(_, next)| *next == '\'') {
                        chars.next();
                    } else {
                        state = State::Normal;
                    }
                }
            }
            State::DoubleQuote => {
                if ch == '"' {
                    if chars.peek().is_some_and(|(_, next)| *next == '"') {
                        chars.next();
                    } else {
                        state = State::Normal;
                    }
                }
            }
            State::LineComment => {
                if ch == '\n' {
                    state = State::Normal;
                }
            }
            State::BlockComment => {
                if ch == '*' && chars.peek().is_some_and(|(_, next)| *next == '/') {
                    chars.next();
                    state = State::Normal;
                }
            }
        }
    }
    Ok(())
}

struct QueryResult {
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
    truncated: bool,
}

fn run_select(conn: &mut Connection, sql: &str) -> Result<QueryResult> {
    conn.execute_batch("BEGIN TRANSACTION READ ONLY")
        .context("begin read-only admin SQL transaction")?;
    let result = run_select_in_transaction(conn, sql);
    match result {
        Ok(value) => {
            conn.execute_batch("COMMIT")
                .context("commit read-only admin SQL transaction")?;
            Ok(value)
        }
        Err(err) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(err)
        }
    }
}

fn run_select_in_transaction(conn: &Connection, sql: &str) -> Result<QueryResult> {
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
    let mut truncated = false;
    while let Some(row) = rows_iter.next().context("row iter")? {
        if rows.len() == ADMIN_SQL_ROW_LIMIT {
            truncated = true;
            break;
        }
        let mut out = Vec::with_capacity(column_count);
        for i in 0..column_count {
            let v = row.get_ref(i).context("get col")?;
            out.push(value_ref_to_json(v));
        }
        rows.push(out);
    }
    Ok(QueryResult {
        columns,
        rows,
        truncated,
    })
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
        "({} row{}{})",
        r.rows.len(),
        if r.rows.len() == 1 { "" } else { "s" },
        if r.truncated { "; truncated" } else { "" },
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
