// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for the web-ui HTML routes.
//!
//! Builds a temp DuckDB via the same `/api/ingest` path real callers use,
//! seeds it with a multi-commit fixture so chart series have more than one
//! point, then snapshots the rendered HTML for both routes plus a chart slug
//! round-trip.

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

/// Three synthetic commits, oldest first. Picked so the rendered output has
/// short SHAs that are visually distinct in snapshots.
fn commits() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        (
            "1111111111111111111111111111111111111111",
            "2026-04-23T12:00:00Z",
            "first commit",
        ),
        (
            "2222222222222222222222222222222222222222",
            "2026-04-24T12:00:00Z",
            "second commit",
        ),
        (
            "3333333333333333333333333333333333333333",
            "2026-04-25T12:00:00Z",
            "third commit",
        ),
    ]
}

/// Build a fixture envelope for one commit; `value_bias` is added to each
/// numeric measurement so successive commits produce a non-flat time series.
fn envelope_for(sha: &str, ts: &str, msg: &str, value_bias: i64) -> Value {
    json!({
        "run_meta": {
            "benchmark_id": "web-ui-fixture",
            "schema_version": 1,
            "started_at": ts
        },
        "commit": {
            "sha": sha,
            "timestamp": ts,
            "message": msg,
            "author_name": "Test Author",
            "author_email": "author@example.com",
            "committer_name": "Test Committer",
            "committer_email": "committer@example.com",
            "tree_sha": "fedcba9876543210fedcba9876543210fedcba98",
            "url": format!("https://github.com/vortex-data/vortex/commit/{sha}")
        },
        "records": [
            {
                "kind": "query_measurement",
                "commit_sha": sha,
                "dataset": "tpch",
                "scale_factor": "1",
                "query_idx": 1,
                "storage": "nvme",
                "engine": "datafusion",
                "format": "vortex-file-compressed",
                "value_ns": 1_000_000 + value_bias,
                "all_runtimes_ns": [1_000_000 + value_bias]
            },
            {
                "kind": "query_measurement",
                "commit_sha": sha,
                "dataset": "tpch",
                "scale_factor": "1",
                "query_idx": 1,
                "storage": "nvme",
                "engine": "duckdb",
                "format": "parquet",
                "value_ns": 800_000 + value_bias,
                "all_runtimes_ns": [800_000 + value_bias]
            },
            {
                "kind": "compression_time",
                "commit_sha": sha,
                "dataset": "tpch-lineitem",
                "format": "vortex-file-compressed",
                "op": "encode",
                "value_ns": 9_000 + value_bias,
                "all_runtimes_ns": [9_000 + value_bias]
            },
            {
                "kind": "compression_size",
                "commit_sha": sha,
                "dataset": "tpch-lineitem",
                "format": "vortex-file-compressed",
                "value_bytes": 4_000 + value_bias
            },
            {
                "kind": "random_access_time",
                "commit_sha": sha,
                "dataset": "taxi",
                "format": "vortex-file-compressed",
                "value_ns": 500 + value_bias,
                "all_runtimes_ns": [500 + value_bias]
            },
            {
                "kind": "vector_search_run",
                "commit_sha": sha,
                "dataset": "cohere-large-10m",
                "layout": "partitioned",
                "flavor": "vortex-turboquant",
                "threshold": 0.75,
                "value_ns": 7_000 + value_bias,
                "all_runtimes_ns": [7_000 + value_bias],
                "matches": 42,
                "rows_scanned": 1_000_000,
                "bytes_scanned": 5_000_000,
                "iterations": 1
            }
        ]
    })
}

async fn seed(server: &Server) -> Result<()> {
    let client = reqwest::Client::new();
    for (i, (sha, ts, msg)) in commits().iter().enumerate() {
        let bias = (i as i64) * 50_000;
        let resp = client
            .post(server.url("/api/ingest"))
            .bearer_auth(TOKEN)
            .json(&envelope_for(sha, ts, msg, bias))
            .send()
            .await?;
        anyhow::ensure!(
            resp.status().is_success(),
            "seed ingest #{i} failed: {}",
            resp.status()
        );
    }
    Ok(())
}

fn insta_settings() -> insta::Settings {
    let mut s = insta::Settings::clone_current();
    s.set_snapshot_path("snapshots");
    s.set_prepend_module_to_snapshot(false);
    s
}

#[tokio::test]
async fn landing_page_snapshot() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let resp = client.get(server.url("/")).send().await?;
    assert_eq!(resp.status(), 200);
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("text/html"),
        "expected text/html, got {content_type:?}"
    );
    let body = resp.text().await?;

    insta_settings().bind(|| {
        insta::assert_snapshot!("landing_page", body);
    });
    Ok(())
}

#[tokio::test]
async fn chart_page_snapshot() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    // Pick the query_measurements chart: it has two series (engine:format
    // combinations) so the snapshot exercises multi-series rendering.
    let groups: Value = client
        .get(server.url("/api/groups"))
        .send()
        .await?
        .json()
        .await?;
    let slug = groups["groups"]
        .as_array()
        .context("groups is array")?
        .iter()
        .find(|g| {
            g["name"]
                .as_str()
                .map(|s| s.starts_with("tpch"))
                .unwrap_or(false)
        })
        .and_then(|g| g["charts"].as_array())
        .and_then(|c| c.first())
        .and_then(|c| c["slug"].as_str())
        .context("tpch chart slug")?
        .to_string();

    let resp = client
        .get(server.url(&format!("/chart/{slug}")))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body = resp.text().await?;
    assert!(
        body.contains(r#"<script id="chart-data" type="application/json">"#),
        "chart payload must be embedded inline"
    );
    assert!(
        body.contains(r#"<script src="/static/chart.umd.js""#),
        "Chart.js must be referenced from the static asset route"
    );

    insta_settings().bind(|| {
        insta::assert_snapshot!("chart_page_query", body);
    });
    Ok(())
}

#[tokio::test]
async fn chart_page_round_trips_every_slug() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let groups: Value = client
        .get(server.url("/api/groups"))
        .send()
        .await?
        .json()
        .await?;
    let slugs: Vec<String> = groups["groups"]
        .as_array()
        .context("groups is array")?
        .iter()
        .flat_map(|g| g["charts"].as_array().cloned().unwrap_or_default())
        .filter_map(|c| c["slug"].as_str().map(str::to_string))
        .collect();
    anyhow::ensure!(!slugs.is_empty(), "expected at least one chart slug");

    for slug in &slugs {
        let resp = client
            .get(server.url(&format!("/chart/{slug}")))
            .send()
            .await?;
        assert_eq!(
            resp.status(),
            200,
            "chart page for slug {slug} should be 200"
        );
        let body = resp.text().await?;
        assert!(
            body.contains(r#"id="chart-data""#),
            "missing inline chart data for slug {slug}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn unknown_slug_renders_404() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let resp = client.get(server.url("/chart/qm.aGVsbG8")).send().await?;
    assert_eq!(resp.status(), 404);
    let body = resp.text().await?;
    assert!(body.contains("chart not found"));
    Ok(())
}

#[tokio::test]
async fn empty_landing_page_renders() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let resp = client.get(server.url("/")).send().await?;
    assert_eq!(resp.status(), 200);
    let body = resp.text().await?;
    assert!(body.contains("No data ingested yet"));
    Ok(())
}

#[tokio::test]
async fn static_assets_are_served() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    for (path, ct_prefix) in [
        ("/static/chart.umd.js", "application/javascript"),
        ("/static/chart-init.js", "application/javascript"),
        ("/static/style.css", "text/css"),
    ] {
        let resp = client.get(server.url(path)).send().await?;
        assert_eq!(resp.status(), 200, "GET {path} should be 200");
        let ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(
            ct.starts_with(ct_prefix),
            "GET {path}: content-type {ct:?} should start with {ct_prefix:?}"
        );
        let bytes = resp.bytes().await?;
        assert!(!bytes.is_empty(), "GET {path}: body must not be empty");
    }
    Ok(())
}
