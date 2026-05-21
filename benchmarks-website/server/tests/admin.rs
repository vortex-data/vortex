// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for `/api/admin/*` - round-trips the bearer check,
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
use vortex_bench_server::app::admin_router;
use vortex_bench_server::app::public_router;

const INGEST_TOKEN: &str = "ingest-test-token";
const ADMIN_TOKEN: &str = "admin-test-token";

/// Two-listener test harness that mirrors prod: public traffic on one
/// loopback port, admin traffic on a second loopback port. The
/// `admin_addr` is `None` when the server was started without
/// `with_admin`. Both listener tasks are aborted on `Drop`.
struct Server {
    addr: SocketAddr,
    admin_addr: Option<SocketAddr>,
    snapshot_dir: std::path::PathBuf,
    state: AppState,
    _tmp: TempDir,
    handle: JoinHandle<()>,
    admin_handle: Option<JoinHandle<()>>,
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

        let public_app = public_router(state.clone());
        let admin_app = admin_router(state.clone());

        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let handle = tokio::spawn(async move {
            axum::serve(listener, public_app).await.unwrap();
        });

        let (admin_addr, admin_handle) = match admin_app {
            Some(app) => {
                let listener = TcpListener::bind("127.0.0.1:0").await?;
                let admin_addr = listener.local_addr()?;
                let admin_handle = tokio::spawn(async move {
                    axum::serve(listener, app).await.unwrap();
                });
                (Some(admin_addr), Some(admin_handle))
            }
            None => (None, None),
        };

        Ok(Self {
            addr,
            admin_addr,
            snapshot_dir,
            state,
            _tmp: tmp,
            handle,
            admin_handle,
        })
    }

    /// URL for a path on the public listener.
    fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }

    /// URL for a path on the admin listener. Panics if admin was not
    /// enabled - tests calling this should have used `start_with_admin`.
    fn admin_url(&self, path: &str) -> String {
        let addr = self
            .admin_addr
            .expect("admin_url called on a server started without admin");
        format!("http://{addr}{path}")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.handle.abort();
        if let Some(handle) = self.admin_handle.take() {
            handle.abort();
        }
    }
}

#[tokio::test]
async fn admin_sql_select_round_trips() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    // The schema is applied on AppState::open, so commits exists with 0 rows.
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
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
        .post(server.admin_url("/api/admin/sql?format=table"))
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
    // Header check explicitly anchors on the trailing `│` to symmetrically
    // match the data-row check, so a regression that mangled the right
    // boundary in the header row would not silently pass. The header text
    // `y` is padded to the widest cell value (`hello`, 5 chars), hence
    // the spaces. `│ x │ y     │` is the literal rendered shape.
    assert!(
        body.contains("│ x │ y     │"),
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
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1 AS x;" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([[1]]));
    Ok(())
}

/// `validate_read_only` / `ensure_single_statement` skip leading and
/// trailing SQL comments (both `-- line` and `/* block */`) without
/// confusing them for additional statements. Regression test for the
/// cycle-1 fix that added the `skip_leading_noise` helper.
#[tokio::test]
async fn admin_sql_accepts_leading_line_comment() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "-- justify the call\nSELECT 1 AS x" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([[1]]));
    Ok(())
}

#[tokio::test]
async fn admin_sql_accepts_leading_block_comment() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "/* note */ SELECT 1 AS x" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([[1]]));
    Ok(())
}

#[tokio::test]
async fn admin_sql_accepts_trailing_line_comment_after_semicolon() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1 AS x; -- trailing note\n" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([[1]]));
    Ok(())
}

#[tokio::test]
async fn admin_sql_accepts_trailing_block_comment_after_semicolon() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1 AS x; /* trailing note */" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([[1]]));
    Ok(())
}

#[tokio::test]
async fn admin_sql_rejects_second_statement_after_comment() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1; /* a */ SELECT 2" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 403);
    Ok(())
}

#[tokio::test]
async fn admin_sql_allows_semicolon_inside_string_literal() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.admin_url("/api/admin/sql"))
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
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1; DROP TABLE commits; SELECT 2" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 403);
    let body: Value = resp.json().await?;
    assert_eq!(body["error"], json!("forbidden"));

    let resp = client
        .post(server.admin_url("/api/admin/sql"))
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
        .post(server.admin_url("/api/admin/sql"))
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
            .post(server.admin_url("/api/admin/sql"))
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
async fn admin_sql_read_only_blocks_explain_analyze_writes() -> Result<()> {
    // `validate_read_only` allow-lists `EXPLAIN`, which means
    // `EXPLAIN ANALYZE` slips through the SQL allow-list - DuckDB actually
    // executes the wrapped statement. The defense in depth is the
    // `BEGIN TRANSACTION READ ONLY` wrapper in `run_select`. Pin that
    // behavior: if this test ever flips green by silently allowing the
    // write, the read-only safety net is gone.
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({
            "sql": "EXPLAIN ANALYZE INSERT INTO commits \
                    (commit_sha, timestamp, tree_sha, url) \
                    VALUES ('deadbeef', NOW(), 'tree', 'http://x')"
        }))
        .send()
        .await?;
    assert!(
        resp.status().is_server_error(),
        "expected DuckDB to refuse the write, got {}",
        resp.status()
    );

    // And verify nothing landed in `commits`.
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT COUNT(*) AS n FROM commits" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["rows"], json!([[0]]));
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
            .post(server.admin_url("/api/admin/sql"))
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
        .post(server.admin_url("/api/admin/sql"))
        .json(&body)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);

    // Wrong token.
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth("wrong")
        .json(&body)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);

    // Ingest token explicitly does NOT work on admin routes.
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(INGEST_TOKEN)
        .json(&body)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);

    // Right token.
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
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

    // Without `with_admin` there is no admin listener at all, so the
    // probe has to hit the public listener. The contract is "404 for any
    // /api/admin/* path on the public listener" regardless of whether
    // admin is configured.
    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 404);

    let resp = client
        .post(server.url("/api/admin/snapshot?ts=20260101T000000Z"))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    assert_eq!(resp.status(), 404);
    Ok(())
}

#[tokio::test]
async fn admin_routes_not_served_on_public_listener_when_admin_enabled() -> Result<()> {
    // The whole point of `VORTEX_BENCH_ADMIN_BIND` is that
    // `/api/admin/*` is unreachable on the public bind even when admin
    // is otherwise configured. Hitting the *public* listener with the
    // correct admin bearer must still 404.
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let resp = client
        .post(server.url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 404);

    let resp = client
        .post(server.url("/api/admin/snapshot?ts=20260101T000000Z"))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    assert_eq!(resp.status(), 404);

    // Sanity: the same admin SQL call on the *admin* listener works.
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({ "sql": "SELECT 1 AS x" }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    Ok(())
}

// The snapshot endpoint INSTALLs and LOADs the vortex DuckDB core
// extension on first call; that needs outbound network to
// `extensions.duckdb.org`, which sandboxed CI environments generally
// don't allow. Run manually before merge with the same invocation form
// the CI workflow uses, so behaviour matches what gates the merge:
//   cargo nextest run -p vortex-bench-server --test admin --run-ignored only
#[tokio::test]
#[ignore = "needs network to install the vortex DuckDB core extension"]
async fn admin_snapshot_creates_export_directory() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let ts = "20260101T000000Z";
    let resp = client
        .post(server.admin_url(&format!("/api/admin/snapshot?ts={ts}")))
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
    // schema.sql is the assembled COMMITS_DDL + per-family DDL concatenation.
    assert!(
        dir_path.join("schema.sql").exists(),
        "{dir}/schema.sql should exist"
    );
    // One .vortex file per table - `commits` is the dim table and is
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
#[ignore = "needs network to install the vortex DuckDB core extension"]
async fn admin_snapshot_rejects_existing_directory() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let ts = "20260102T000000Z";
    let resp = client
        .post(server.admin_url(&format!("/api/admin/snapshot?ts={ts}")))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    assert_eq!(resp.status(), 200);

    // Second call with same ts → 409.
    let resp = client
        .post(server.admin_url(&format!("/api/admin/snapshot?ts={ts}")))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    assert_eq!(resp.status(), 409);
    Ok(())
}

#[tokio::test]
#[ignore = "needs network to install the vortex DuckDB core extension"]
async fn admin_snapshot_concurrent_same_ts_yields_one_200_and_one_409() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let ts = "20260103T000000Z";

    let client_a = reqwest::Client::new();
    let client_b = reqwest::Client::new();
    let url = server.admin_url(&format!("/api/admin/snapshot?ts={ts}"));
    let url_a = url.clone();
    let url_b = url;

    let a =
        tokio::spawn(async move { client_a.post(&url_a).bearer_auth(ADMIN_TOKEN).send().await });
    let b =
        tokio::spawn(async move { client_b.post(&url_b).bearer_auth(ADMIN_TOKEN).send().await });

    let resp_a = a.await??;
    let resp_b = b.await??;
    let mut codes = [resp_a.status().as_u16(), resp_b.status().as_u16()];
    codes.sort();
    assert_eq!(
        codes,
        [200, 409],
        "expected exactly one 200 and one 409 for concurrent snapshots with same ts"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "needs network to install the vortex DuckDB core extension"]
async fn admin_snapshot_captures_committed_row_under_read_only_transaction() -> Result<()> {
    // The per-table COPYs run inside a single `BEGIN TRANSACTION READ ONLY`.
    // The contract is that a row already committed to `commits` lands in
    // the snapshot's `commits.vortex`. A full "ingest mid-snapshot" race
    // is hard to make deterministic, so this test pins the easier half:
    // commit-then-snapshot must see the commit.
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();
    let sha = "deadbeefcafebabedeadbeefcafebabedeadbeef";
    {
        let sha = sha.to_string();
        vortex_bench_server::db::run_blocking(&server.state.db, move |conn| {
            conn.execute_batch(&format!(
                "INSERT INTO commits (commit_sha, timestamp, tree_sha, url) \
                 VALUES ('{sha}', NOW(), 'tree-sha', 'http://example/{sha}')"
            ))?;
            Ok(())
        })
        .await?;
    }

    let ts = "20260104T000000Z";
    let resp = client
        .post(server.admin_url(&format!("/api/admin/snapshot?ts={ts}")))
        .bearer_auth(ADMIN_TOKEN)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    assert_eq!(status, 200, "snapshot failed: {text}");
    let body: Value = serde_json::from_str(&text)?;
    let snapshot_dir = body["snapshot_dir"]
        .as_str()
        .context("snapshot_dir field")?
        .to_string();
    let commits_path = format!("{snapshot_dir}/commits.vortex");

    // Round-trip through the read-only `/api/admin/sql` endpoint so the
    // verification uses the same DuckDB + Vortex extension the server
    // produced the snapshot with - no separate process or path to keep
    // in sync.
    let resp = client
        .post(server.admin_url("/api/admin/sql"))
        .bearer_auth(ADMIN_TOKEN)
        .json(&json!({
            "sql": format!(
                "SELECT commit_sha FROM read_vortex('{commits_path}')"
            )
        }))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(
        body["rows"],
        json!([[sha]]),
        "snapshot's commits.vortex did not contain the committed row"
    );
    Ok(())
}

#[tokio::test]
async fn admin_snapshot_validates_ts() -> Result<()> {
    let server = Server::start_with_admin().await?;
    let client = reqwest::Client::new();

    let too_long = "x".repeat(65);
    for bad_ts in [
        "",
        "../oops",
        "with space",
        too_long.as_str(),
        // ISO-8601-looking inputs that operators might paste but the
        // regex `[A-Za-z0-9_-]+` does not accept (`.` for fractional
        // seconds, `:` for time, `+` for offset). Pin the rejection so a
        // future regex relaxation does not silently let through path
        // characters that break the snapshot dir layout.
        "2026-01-01T00:00:00Z",
        "2026-01-01+00:00",
        "20260101.000000Z",
    ] {
        let url = server.admin_url(&format!("/api/admin/snapshot?ts={}", urlencoding(bad_ts)));
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
