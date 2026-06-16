// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Admin endpoints - bearer-gated DuckDB snapshot and read-only SQL.
//!
//! Mounted at `/api/admin/*` only when `ADMIN_BEARER_TOKEN` is set on the
//! server, surfaced through [`crate::app::AppState::with_admin`]. Both routes
//! require an `Authorization: Bearer <ADMIN_BEARER_TOKEN>` header - the
//! `INGEST_BEARER_TOKEN` will not work here, so the two can rotate
//! independently. The operator workflow is documented in
//! `benchmarks-website/ops/README.md`.
//!
//! ## Routes
//!
//! ### `POST /api/admin/snapshot?ts=<id>`
//!
//! Writes a snapshot directory `<snapshot_dir>/<ts>/` containing:
//! - `schema.sql` - concatenated DDL ([`crate::schema::COMMITS_DDL`] plus
//!   every [`crate::family::FAMILIES`] entry's `schema_ddl`), so a
//!   restore knows how to recreate the tables before bulk-loading.
//! - `<table>.vortex` for every table in [`crate::schema::TABLES`] -
//!   each produced by a `COPY (SELECT * FROM <table>) TO …
//!   (FORMAT vortex)`. The vortex DuckDB extension is auto-installed
//!   from the DuckDB core extension repo on first call, then `LOAD`ed.
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
//! `EXPLAIN` statements are allowed - anything else is rejected with 403.
//! Results are capped at `ADMIN_SQL_ROW_LIMIT` rows; responses past
//! that cap include `"truncated": true`. The handler runs each query on
//! its own cloned connection inside a `BEGIN TRANSACTION READ ONLY`
//! wrapper, so concurrent ingest writes proceed without contention.

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
use crate::family;
use crate::schema;

const ADMIN_SQL_ROW_LIMIT: usize = 10_000;

/// Errors surfaced by `/api/admin/*` handlers. Auth (401) is handled by
/// [`require_admin_bearer`] and never reaches a handler.
#[derive(Debug, Error)]
pub enum AdminError {
    /// 400 - request shape is malformed (bad `ts`, bad SQL JSON body, …).
    #[error("bad request: {0}")]
    BadRequest(String),
    /// 403 - request is well-formed but the SQL statement is not on the
    /// read-only allow-list.
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// 409 - snapshot target directory already exists.
    #[error("conflict: {0}")]
    Conflict(String),
    /// 500 - anything else (DB error, IO error, …).
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

    // Process-local `ts` reservation. Two concurrent calls with the
    // same `ts` would otherwise both write tmp directories and then
    // race at the `rename(2)` step - Linux silently overwrites an
    // existing destination, so the loser's snapshot disappears with no
    // signal. The reservation closes that race within a single
    // `vortex-bench-server` process (the supported deployment).
    let _ticket = SnapshotTicket::acquire(&state, &q.ts, &target)?;

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
        cleanup_partial(&tmp);
        return Err(AdminError::Internal(err));
    }
    // The ticket guarantees no other in-process call has the same `ts`
    // reserved, so the final `rename(2)` will land cleanly. We still
    // recheck `target.exists()` because a different process or an
    // operator hand-creating the dir would also lose data on a silent
    // overwrite.
    if target.exists() {
        cleanup_partial(&tmp);
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
        cleanup_partial(&tmp);
        return Err(AdminError::Internal(err));
    }
    Ok(Json(SnapshotResponse {
        snapshot_dir: target.display().to_string(),
    }))
}

/// RAII guard that holds a `ts` in [`AppState::pending_snapshots`] for the
/// duration of one `/api/admin/snapshot` call. Dropping the guard always
/// releases the reservation, even on panic or early-return error paths.
struct SnapshotTicket {
    state: AppState,
    ts: String,
}

impl SnapshotTicket {
    fn acquire(state: &AppState, ts: &str, target: &Path) -> Result<Self, AdminError> {
        let inserted = state.pending_snapshots.lock().insert(ts.to_string());
        if !inserted {
            return Err(AdminError::Conflict(format!(
                "snapshot for ts={ts} is already in flight (target {})",
                target.display()
            )));
        }
        Ok(Self {
            state: state.clone(),
            ts: ts.to_string(),
        })
    }
}

impl Drop for SnapshotTicket {
    fn drop(&mut self) {
        self.state.pending_snapshots.lock().remove(&self.ts);
    }
}

/// Best-effort cleanup of a partially-written snapshot tmp dir. Logs the
/// failure rather than silently discarding it, so a wedge (disk full,
/// permission flip) is visible in the journal even when no automated
/// sweeper is wired up.
fn cleanup_partial(path: &Path) {
    if let Err(err) = std::fs::remove_dir_all(path) {
        // ENOENT just means the dir never got created or was already
        // cleaned up by a sibling caller; ignore it. Anything else
        // deserves a warn.
        if err.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(
                path = %path.display(),
                error = ?err,
                "failed to clean up partial snapshot tmp dir; manual sweep may be needed"
            );
        }
    }
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
    // bulk-loading the per-table vortex files. The DDL is assembled
    // from the commits dim + every fact-table family in the same order
    // `db::open()` applies them.
    let mut schema_sql = String::with_capacity(8 * 1024);
    schema_sql.push_str(schema::COMMITS_DDL);
    for fam in family::FAMILIES {
        schema_sql.push_str(fam.schema_ddl);
    }
    std::fs::write(target.join("schema.sql"), schema_sql)
        .with_context(|| format!("writing schema.sql under {}", target.display()))?;

    let target_for_db = target.to_path_buf();
    db::run_blocking(&state.db, move |conn| {
        export_snapshot_tables(conn, &target_for_db)
    })
    .await
}

fn export_snapshot_tables(conn: &mut Connection, target: &Path) -> Result<()> {
    // Idempotent - `INSTALL` is a no-op if the extension is already
    // present, `LOAD` is cheap once the binary is on disk. Vortex is a
    // DuckDB core extension (not community), so the unqualified `INSTALL`
    // hits the right repo on first call; subsequent calls are local.
    // Runs outside the snapshot transaction because extension installation
    // is not transactional.
    conn.execute_batch("INSTALL vortex; LOAD vortex;")
        .context("INSTALL/LOAD vortex extension")?;

    // All per-table COPYs share one `READ ONLY` transaction. Otherwise an
    // ingest commit between the `commits` export and the
    // `query_measurements` export yields an inconsistent backup - facts
    // referencing a commit row that is not in the snapshot, or vice
    // versa. The transaction's READ ONLY guard also belts-and-braces
    // against the snapshot path accidentally writing.
    conn.execute_batch("BEGIN TRANSACTION READ ONLY")
        .context("begin read-only snapshot transaction")?;
    if let Err(err) = copy_each_table(conn, target) {
        if let Err(rb_err) = conn.execute_batch("ROLLBACK") {
            tracing::warn!(
                error = ?rb_err,
                "rolling back snapshot read-only transaction failed; the original \
                 export error (returned to the caller) is the actionable one"
            );
        }
        return Err(err);
    }
    conn.execute_batch("COMMIT")
        .context("commit read-only snapshot transaction")?;
    Ok(())
}

fn copy_each_table(conn: &Connection, target: &Path) -> Result<()> {
    for table in schema::TABLES.iter() {
        let path = target.join(format!("{table}.vortex"));
        let path_str = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("snapshot path is not UTF-8: {}", path.display()))?;
        let sql = format!(
            "COPY (SELECT * FROM {table}) TO {} (FORMAT vortex)",
            db::sql_string_literal(path_str)
        );
        conn.execute_batch(&sql)
            .with_context(|| format!("COPY {table} TO {path_str}"))?;
    }
    Ok(())
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

/// Strips leading whitespace, parens, semicolons, and SQL comments (both `--`
/// line comments and `/* ... */` block comments) from `sql`. Returns the byte
/// offset of the first non-comment, non-whitespace token. Used by
/// [`validate_read_only`] so a query like `-- justify the call\nSELECT 1` is
/// not rejected with `only [...] are allowed; got ""`.
fn skip_leading_noise(sql: &str) -> usize {
    let bytes = sql.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == b'(' || b == b';' {
            i += 1;
            continue;
        }
        if b == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'-' {
            // Line comment runs to end of line.
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            // Block comment, search for the matching `*/`.
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2;
            } else {
                // Unterminated block comment; let the SQL parser surface
                // the error rather than guessing.
                return i;
            }
            continue;
        }
        break;
    }
    i
}

fn validate_read_only(sql: &str) -> Result<(), AdminError> {
    ensure_single_statement(sql)?;
    let start = skip_leading_noise(sql);
    let first_word: String = sql[start..]
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
                ';' => {
                    // Allow trailing whitespace and SQL comments after the
                    // terminator (`SELECT 1; -- note` and `SELECT 1; /* a */`
                    // are valid single statements). Only error if a
                    // non-comment, non-whitespace token follows.
                    let suffix_start = idx + ch.len_utf8();
                    let after = skip_leading_noise(&sql[suffix_start..]);
                    if !sql[suffix_start + after..].is_empty() {
                        return Err(AdminError::Forbidden(
                            "admin SQL accepts a single statement only".into(),
                        ));
                    }
                    return Ok(());
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
            if let Err(rb_err) = conn.execute_batch("ROLLBACK") {
                tracing::warn!(
                    error = ?rb_err,
                    "rolling back admin SQL read-only transaction failed; the \
                     original query error (returned to the caller) is the \
                     actionable one"
                );
            }
            Err(err)
        }
    }
}

fn run_select_in_transaction(conn: &Connection, sql: &str) -> Result<QueryResult> {
    let mut stmt = conn.prepare(sql).context("prepare SQL")?;
    let mut rows_iter = stmt.query([]).context("execute SQL")?;
    // duckdb-rs panics on Statement::column_names() if the statement has not
    // executed yet - schema is only populated after `query()` runs. Pull it
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

/// Coerce a DuckDB [`ValueRef`] into a JSON [`Value`] for the admin SQL API.
///
/// `String::from_utf8_lossy` is used for `Text`: non-UTF-8 bytes in a TEXT
/// column are a misuse but not a reason to fail the request; the lossy
/// replacement (U+FFFD) surfaces so the caller can see something is wrong.
///
/// `Decimal` is rendered via its Display impl. `Timestamp` is rendered as
/// `<unit>:<raw>` (one of `s|ms|us|ns:<count-since-epoch>`) so it
/// round-trips through JSON unambiguously without pulling chrono / time
/// in as a dependency; consumers that want a human-readable ISO-8601
/// can post-process the string. Other compound types (`List`, `Struct`,
/// `Array`, `Map`, `Union`, `Enum`) are rare in this database's schema;
/// they fall back to a best-effort Debug rendering tagged with the type
/// name so the caller can see something printable and we can extend
/// this match when we hit one.
fn value_ref_to_json(v: ValueRef<'_>) -> Value {
    use duckdb::types::TimeUnit;
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
        ValueRef::Decimal(d) => Value::String(d.to_string()),
        ValueRef::Timestamp(unit, raw) => {
            // DuckDB stores timestamps as an integer count since
            // 1970-01-01 UTC at the named precision. Surface them as a
            // stable structured string keyed by the unit ("s:1700000000",
            // "ms:1700000000000", etc.) so a future consumer can parse
            // unambiguously without us reaching for chrono / time as a
            // dependency in this slice.
            let unit_str = match unit {
                TimeUnit::Second => "s",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Microsecond => "us",
                TimeUnit::Nanosecond => "ns",
            };
            Value::String(format!("{unit_str}:{raw}"))
        }
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
    // writeln! into a String only errors if the underlying Write impl
    // returns one - fmt::Write for String is infallible - so the
    // Result is discarded by design.
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
