// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests covering the acceptance criteria from
//! `benchmarks-website/planning/components/server.md`.

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

const TOKEN: &str = "test-bearer-token";

struct Server {
    addr: SocketAddr,
    _tmp: TempDir,
    handle: JoinHandle<()>,
}

impl Server {
    async fn start() -> Result<Self> {
        let tmp = TempDir::new()?;
        let db_path = tmp.path().join("bench.duckdb");
        let state = AppState::open(&db_path, TOKEN.to_string())?;
        let app = router(state);

        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Ok(Self {
            addr,
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

fn fixture_envelope() -> Value {
    let raw = include_str!("../fixtures/envelope.json");
    serde_json::from_str(raw).expect("fixture envelope is valid JSON")
}

#[tokio::test]
async fn happy_path_then_idempotent_reingest() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();
    let envelope = fixture_envelope();

    let resp = client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 200, "first ingest should be 200");
    let body: Value = resp.json().await?;
    assert_eq!(body["inserted"].as_u64(), Some(5));
    assert_eq!(body["updated"].as_u64(), Some(0));

    let resp = client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 200, "second ingest should be 200");
    let body: Value = resp.json().await?;
    assert_eq!(body["inserted"].as_u64(), Some(0), "no new rows on re-emit");
    assert!(
        body["updated"].as_u64().context("updated is u64")? > 0,
        "re-emit must report at least one updated row"
    );
    Ok(())
}

#[tokio::test]
async fn missing_bearer_is_unauthorized() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();
    let envelope = fixture_envelope();

    let resp = client
        .post(server.url("/api/ingest"))
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);
    Ok(())
}

#[tokio::test]
async fn wrong_bearer_is_unauthorized() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();
    let envelope = fixture_envelope();

    let resp = client
        .post(server.url("/api/ingest"))
        .bearer_auth("not-the-real-token")
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 401);
    Ok(())
}

#[tokio::test]
async fn unknown_kind_is_400() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let envelope = json!({
        "run_meta": {
            "benchmark_id": "fixture",
            "schema_version": 1,
            "started_at": "2026-04-25T00:00:00Z"
        },
        "commit": {
            "sha": "0123456789abcdef0123456789abcdef01234567",
            "timestamp": "2026-04-25T00:00:00Z",
            "message": "x", "author_name": "x", "author_email": "x@x",
            "committer_name": "x", "committer_email": "x@x",
            "tree_sha": "fedcba9876543210fedcba9876543210fedcba98",
            "url": "https://example.com"
        },
        "records": [
            { "kind": "made_up_kind", "commit_sha": "0123456789abcdef0123456789abcdef01234567" }
        ]
    });
    let resp = client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 400);
    Ok(())
}

#[tokio::test]
async fn unknown_field_is_400() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let mut envelope = fixture_envelope();
    envelope["records"][0]["surprise_field"] = json!("oops");
    let resp = client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 400);
    Ok(())
}

#[tokio::test]
async fn schema_version_too_new_is_409() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let mut envelope = fixture_envelope();
    envelope["run_meta"]["schema_version"] = json!(99);
    let resp = client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 409);
    Ok(())
}

#[tokio::test]
async fn invalid_storage_is_400_record_error() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let mut envelope = fixture_envelope();
    envelope["records"][0]["storage"] = json!("gcs");
    let resp = client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&envelope)
        .send()
        .await?;
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await?;
    assert_eq!(body["record_index"], json!(0));
    Ok(())
}

#[tokio::test]
async fn health_reports_after_ingest() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    // Pre-ingest: counts are zero.
    let resp = client.get(server.url("/health")).send().await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["schema_version"], 1);
    assert_eq!(body["row_counts"]["commits"], 0);
    assert!(body["latest_commit_timestamp"].is_null());

    // Ingest, then re-check.
    client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&fixture_envelope())
        .send()
        .await?;

    let resp = client.get(server.url("/health")).send().await?;
    let body: Value = resp.json().await?;
    assert_eq!(body["row_counts"]["commits"], 1);
    assert_eq!(body["row_counts"]["query_measurements"], 1);
    assert_eq!(body["row_counts"]["compression_times"], 1);
    assert_eq!(body["row_counts"]["compression_sizes"], 1);
    assert_eq!(body["row_counts"]["random_access_times"], 1);
    assert_eq!(body["row_counts"]["vector_search_runs"], 1);
    assert!(!body["latest_commit_timestamp"].is_null());
    Ok(())
}

#[tokio::test]
async fn read_routes_serve_after_ingest() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    client
        .post(server.url("/api/ingest"))
        .bearer_auth(TOKEN)
        .json(&fixture_envelope())
        .send()
        .await?;

    let resp = client.get(server.url("/api/groups")).send().await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    let groups = body["groups"].as_array().context("groups is array")?;
    assert!(
        !groups.is_empty(),
        "groups should not be empty after ingest"
    );

    // Pick the first chart slug and round-trip it.
    let first_chart = groups
        .iter()
        .find_map(|g| g["charts"].as_array().and_then(|c| c.first()))
        .context("at least one chart")?;
    let slug = first_chart["slug"]
        .as_str()
        .context("slug is a string")?
        .to_string();

    let resp = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?;
    assert_eq!(resp.status(), 200, "chart {slug} should resolve");
    let body: Value = resp.json().await?;
    assert!(body["display_name"].is_string());
    assert!(body["unit_kind"].is_string());
    assert!(body["commits"].is_array());
    assert_eq!(
        body["commits"]
            .as_array()
            .context("commits is array")?
            .len(),
        1
    );
    assert!(body["series"].is_object());
    Ok(())
}

#[tokio::test]
async fn unknown_slug_is_404() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let resp = client
        .get(server.url("/api/chart/qm.aGVsbG8"))
        .send()
        .await?;
    // Either 400 (couldn't decode JSON) or 404 (decoded but no rows). Both are
    // acceptable per the contract; we just need it to not be a 500.
    assert!(
        resp.status() == 400 || resp.status() == 404,
        "got {}",
        resp.status()
    );
    Ok(())
}
