// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for `/api/admin/*` — round-trips the bearer check,
//! the read-only SQL allow-list, the snapshot endpoint's path validation,
//! and verifies that admin routes 404 when `ADMIN_BEARER_TOKEN` is unset.

use std::net::SocketAddr;

use anyhow::Context as _;
use anyhow::Result;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use vortex_bench_server::app::AppState;
use vortex_bench_server::app::router;

const INGEST_TOKEN: &str = "ingest-test-token";
const ADMIN_TOKEN: &str = "admin-test-token";

struct Server {
    addr: SocketAddr,
    snapshot_dir: std::path::PathBuf,
    _tmp: TempDir,
    handle: JoinHandle<()>,
}

impl Server {
    /// Start a server with the admin router enabled.
    async fn start_with_admin() -> Result<Self> {
        Self::start_inner(true).await
    }

    /// Start a server without the admin router (verifies admin routes 404).
    async fn start_no_admin() -> Result<Self> {
        Self::start_inner(false).await
    }

    async fn start_inner(enable_admin: bool) -> Result<Self> {
        let tmp = TempDir::new()?;
        let db_path = tmp.path().join("bench.duckdb");
        let snapshot_dir = tmp.path().join("snapshots");
        let mut state = AppState::open(&db_path, INGEST_TOKEN.to_string())?
            .with_snapshot_dir(snapshot_dir.clone());
        if enable_admin {
            state = state.with_admin(ADMIN_TOKEN.to_string());
        }
        let app = router(state);
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Ok(Self {
            addr,
            snapshot_dir,
            _tmp: tmp,
            handle,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[tokio::test]
async fn admin_sql_select_round_trips() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    // The schema is applied on AppState::open, so commits exists with 0 rows.
    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT COUNT(*) AS n FROM commits" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["columns"], json!(["n"]));
    assert_eq!(body["rows"], json!([[0]]));
    assert_eq!(body["row_count"], json!(1));
    Ok(())
}

#[tokio::test]
async fn admin_sql_table_format_renders_ascii() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.url("/api/admin/sql?format=table"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1 AS x, 'hello' AS y" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("text/plain"), "got content-type {ct:?}");
    let body = resp.text().await?;
    assert!(
        body.contains("│ x │ y"),
        "missing column header row in:\n{body}"
    );
    assert!(
        body.contains("│ 1 │ hello │"),
        "missing data row in:\n{body}"
    );
    assert!(body.contains("(1 row)"), "missing row count in:\n{body}");
    Ok(())
}

#[tokio::test]
async fn admin_sql_allows_single_trailing_semicolon() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1 AS x;" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([[1]]));
    Ok(())
}

#[tokio::test]
async fn admin_sql_allows_semicolon_inside_string_literal() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 'not; another statement' AS text" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([["not; another statement"]]));
    Ok(())
}

#[tokio::test]
async fn admin_sql_rejects_multi_statement_reads() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1; DROP TABLE commits; SELECT 2" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 403);
    let body: Value = resp.json().await?;
    assert_eq!(body["error"], json!("forbidden"));

    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT COUNT(*) AS n FROM commits" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    Ok(())
}

#[tokio::test]
async fn admin_sql_caps_large_results() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT * FROM range(10005)" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["row_count"], json!(10000));
    assert_eq!(body["truncated"], json!(true));
    Ok(())
}

#[tokio::test]
async fn admin_sql_rejects_writes() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    for sql in [
        "DELETE FROM commits",
        "UPDATE commits SET sha = 'x'",
        "DROP TABLE commits",
        "INSERT INTO commits VALUES ('x')",
        "CREATE TABLE foo (a INT)",
        "ATTACH ':memory:' AS bar",
    ] {
        let resp = client
            .post(server.url("/api/admin/sql"))
            .bearer_auth(ADMIN_TOKEN)
            .json(&json!({ "sql": sql }))
            .send()
            .await?;
        assert_eq!(resp.status(), 403, "expected 403 for {sql:?}");
        let body: Value = resp.json().await?;
        assert_eq!(body["error"], json!("forbidden"));
    }
    Ok(())
}

#[tokio::test]
async fn admin_sql_allows_pragma_show_describe_explain_with() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    for sql in [
        "PRAGMA database_size",
        "SHOW TABLES",
        "DESCRIBE commits",
        "EXPLAIN SELECT 1",
        "WITH x AS (SELECT 1 AS a) SELECT * FROM x",
    ] {
        let resp = client
            .post(server.url("/api/admin/sql"))
            .bearer_auth(ADMIN_TOKEN)
            .json(&json!({ "sql": sql }))
            .send()
            .await?;
        assert_eq!(resp.status(), 200, "{sql:?} should be allowed");
    }
    Ok(())
}

#[tokio::test]
async fn admin_requires_admin_bearer_not_ingest_bearer() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let body = json!({ "sql": "SELECT 1" });

    // No header.
    let resp = client
        .post(server.url("/api/admin/sql"))
        .json(&body)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);

    // Wrong token.
    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth("wrong")
        .json(&body)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);

    // Ingest token explicitly does NOT work on admin routes.
    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(INGEST_TOKEN)
        .json(&body)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);

    // Right token.
    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&body)
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    Ok(())
}

#[tokio::test]
async fn admin_unmounted_when_admin_token_absent() -> Result<()> {
    let server = Server::start_no_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1" }))
        .send()
        .await?;
    // Without with_admin, the route is not registered at all → 404.
    assert_eq!(resp.status(), 404);

    let resp = client
        .post(server.url("/api/admin/snapshot?ts=20260101T000000Z"))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    assert_eq!(resp.status(), 404);
    Ok(())
}

// The snapshot endpoint INSTALLs and LOADs the vortex DuckDB community
// extension on first call; that needs outbound network to
// `community-extensions.duckdb.org` which sandboxed CI environments
// generally don't allow. Run manually before merge:
//   cargo test -p vortex-bench-server --test admin -- --ignored
#[tokio::test]
#[ignore = "needs network to install the vortex DuckDB community extension"]
async fn admin_snapshot_creates_export_directory() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let ts = "20260101T000000Z";
    let resp = client
        .post(server.url(&format!("/api/admin/snapshot?ts={ts}")))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    assert_eq!(status, 200, "snapshot failed: {text}");
    let body: Value = serde_json::from_str(&text)?;
    let dir = body["snapshot_dir"]
        .as_str()
        .context("snapshot_dir field")?;
    let dir_path = std::path::PathBuf::from(dir);
    assert!(dir_path.exists(), "{dir} should exist");
    // schema.sql is written verbatim from SCHEMA_DDL.
    assert!(
        dir_path.join("schema.sql").exists(),
        "{dir}/schema.sql should exist"
    );
    // One .vortex file per table — `commits` is the dim table and is
    // present even when the DB is otherwise empty (the schema was
    // applied at AppState::open).
    assert!(
        dir_path.join("commits.vortex").exists(),
        "{dir}/commits.vortex should exist"
    );
    assert!(
        dir_path.join("query_measurements.vortex").exists(),
        "{dir}/query_measurements.vortex should exist"
    );
    // And the directory should be under the configured snapshot dir.
    assert!(
        dir_path.starts_with(&server.snapshot_dir),
        "{dir} not under {:?}",
        server.snapshot_dir
    );
    Ok(())
}

#[tokio::test]
#[ignore = "needs network to install the vortex DuckDB community extension"]
async fn admin_snapshot_rejects_existing_directory() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let ts = "20260102T000000Z";
    let resp = client
        .post(server.url(&format!("/api/admin/snapshot?ts={ts}")))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    assert_eq!(resp.status(), 200);

    // Second call with same ts → 409.
    let resp = client
        .post(server.url(&format!("/api/admin/snapshot?ts={ts}")))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    assert_eq!(resp.status(), 409);
    Ok(())
}

#[tokio::test]
async fn admin_snapshot_validates_ts() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let too_long = "x".repeat(65);
    for bad_ts in ["", "../oops", "with space", too_long.as_str()] {
        let url = server.url(&format!("/api/admin/snapshot?ts={}", urlencoding(bad_ts)));
        let resp = client.post(&url).bearer_auth(ADMIN_TOKEN).send().await?;
        assert_eq!(resp.status(), 400, "expected 400 for ts={bad_ts:?}");
    }
    Ok(())
}

/// Tiny URL-encoder so the test doesn't grow another dep. Only handles the
/// characters our bad-ts cases produce.
fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
