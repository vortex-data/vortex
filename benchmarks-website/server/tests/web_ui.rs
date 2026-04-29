// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for the web-ui HTML routes.
//!
//! Builds a temp DuckDB via the same `/api/ingest` path real callers use,
//! seeds it with a multi-commit fixture so chart series have more than one
//! point, then snapshots the rendered HTML for each route plus a chart slug
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
                "kind": "query_measurement",
                "commit_sha": sha,
                "dataset": "tpch",
                "scale_factor": "1",
                "query_idx": 2,
                "storage": "nvme",
                "engine": "datafusion",
                "format": "vortex-file-compressed",
                "value_ns": 600_000 + value_bias,
                "all_runtimes_ns": [600_000 + value_bias]
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
                "kind": "compression_time",
                "commit_sha": sha,
                "dataset": "tpch-lineitem",
                "format": "vortex-file-compressed",
                "op": "decode",
                "value_ns": 5_000 + value_bias,
                "all_runtimes_ns": [5_000 + value_bias]
            },
            {
                "kind": "compression_time",
                "commit_sha": sha,
                "dataset": "tpch-lineitem",
                "format": "parquet",
                "op": "encode",
                "value_ns": 18_000 + (2 * value_bias),
                "all_runtimes_ns": [18_000 + (2 * value_bias)]
            },
            {
                "kind": "compression_time",
                "commit_sha": sha,
                "dataset": "tpch-lineitem",
                "format": "parquet",
                "op": "decode",
                "value_ns": 10_000 + (2 * value_bias),
                "all_runtimes_ns": [10_000 + (2 * value_bias)]
            },
            {
                "kind": "compression_size",
                "commit_sha": sha,
                "dataset": "tpch-lineitem",
                "format": "vortex-file-compressed",
                "value_bytes": 4_000 + value_bias
            },
            {
                "kind": "compression_size",
                "commit_sha": sha,
                "dataset": "tpch-lineitem",
                "format": "parquet",
                "value_bytes": 8_000 + (2 * value_bias)
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
                "kind": "random_access_time",
                "commit_sha": sha,
                "dataset": "taxi",
                "format": "parquet",
                "value_ns": 1_000 + (2 * value_bias),
                "all_runtimes_ns": [1_000 + (2 * value_bias)]
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

/// Lift a single chart slug from `/api/groups`, picking from a group whose
/// name matches `predicate`. Used by tests that need a real slug to drive
/// `/chart/{slug}` and `/api/chart/{slug}` round-trips.
async fn pick_chart_slug(server: &Server, predicate: impl Fn(&str) -> bool) -> Result<String> {
    let client = reqwest::Client::new();
    let groups: Value = client
        .get(server.url("/api/groups"))
        .send()
        .await?
        .json()
        .await?;
    groups["groups"]
        .as_array()
        .context("groups is array")?
        .iter()
        .find(|g| g["name"].as_str().is_some_and(&predicate))
        .and_then(|g| g["charts"].as_array())
        .and_then(|c| c.first())
        .and_then(|c| c["slug"].as_str())
        .map(str::to_string)
        .context("matching chart slug")
}

async fn pick_group_slug(server: &Server, predicate: impl Fn(&str) -> bool) -> Result<String> {
    let client = reqwest::Client::new();
    let groups: Value = client
        .get(server.url("/api/groups"))
        .send()
        .await?
        .json()
        .await?;
    groups["groups"]
        .as_array()
        .context("groups is array")?
        .iter()
        .find(|g| g["name"].as_str().is_some_and(&predicate))
        .and_then(|g| g["slug"].as_str())
        .map(str::to_string)
        .context("matching group slug")
}

fn group_by_name<'a>(groups: &'a Value, name: &str) -> Result<&'a Value> {
    groups["groups"]
        .as_array()
        .context("groups is array")?
        .iter()
        .find(|g| g["name"].as_str() == Some(name))
        .with_context(|| format!("group {name:?} exists"))
}

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta < 0.000_001,
        "expected {actual} to be close to {expected}"
    );
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

    // Inline canvas + chart-data-0 from the open-by-default first group.
    assert!(
        body.contains("<canvas"),
        "landing page must render at least one <canvas>"
    );
    assert!(
        body.contains(r#"id="chart-data-0""#),
        "the open-by-default first group must inline its chart payload"
    );
    assert!(
        body.contains(r#"data-chart-slug="#),
        "every chart card carries data-chart-slug for the lazy-fetch path"
    );
    assert!(
        !body.contains(r#"id="group-search""#),
        "landing page should not render the old group search bar"
    );
    assert!(
        body.contains(r#"class="sticky-header""#),
        "landing page should render the v2-style top navbar"
    );
    assert!(
        body.contains(r#"data-action="expand-all""#)
            && body.contains(r#"data-action="collapse-all""#),
        "navbar should expose expand/collapse controls"
    );
    assert!(
        body.contains(r#"data-role="theme-toggle""#),
        "navbar should expose a theme toggle"
    );
    assert!(
        body.contains(r#"class="btn-icon""#)
            || body.contains(r#"class="btn-icon theme-icon theme-icon-light""#),
        "navbar controls should render icons"
    );
    assert!(
        body.contains(r#"vortex_black_nobg.svg"#) && body.contains(r#"vortex_white_nobg.svg"#),
        "navbar should render the Vortex logo assets"
    );
    assert!(
        body.contains("⚡") && body.contains("📤") && body.contains("⬇️") && body.contains("📊"),
        "summaries should render the v2 summary icons"
    );

    insta_settings().bind(|| {
        insta::assert_snapshot!("landing_page", body);
    });
    Ok(())
}

/// The first group disclosure is rendered with the `open` attribute; every
/// other group lacks it, so the user sees only the first group's charts on
/// first paint.
#[tokio::test]
async fn details_first_group_open_others_closed() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;

    let opens: Vec<_> = body
        .match_indices(r#"<details class="group-disclosure""#)
        .map(|(i, _)| {
            let tag_end = body[i..].find('>').map(|p| i + p).unwrap_or(i);
            body[i..=tag_end].contains(" open")
        })
        .collect();
    assert!(!opens.is_empty(), "landing page must render <details>");
    assert!(opens[0], "first group must be open");
    for (i, is_open) in opens.iter().enumerate().skip(1) {
        assert!(!is_open, "group #{i} must be closed by default");
    }
    Ok(())
}

#[tokio::test]
async fn collapsed_groups_still_show_summaries() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;

    let mut found_visible_summary = false;
    for (group_start, _) in body.match_indices(r#"<section class="group-details""#) {
        let details_start = body[group_start..]
            .find(r#"<details class="group-disclosure""#)
            .map(|p| group_start + p)
            .context("group contains disclosure")?;
        let details_tag_end = body[details_start..]
            .find('>')
            .map(|p| details_start + p)
            .context("details tag closes")?;
        let is_open = body[details_start..=details_tag_end].contains(" open");
        if is_open {
            continue;
        }

        let summary_end = body[details_start..]
            .find("</details>")
            .map(|p| details_start + p)
            .context("disclosure closes")?;
        let chart_grid_start = body[summary_end..]
            .find(r#"<div class="chart-grid">"#)
            .map(|p| summary_end + p)
            .context("details contains chart grid")?;
        let visible_region = &body[summary_end..chart_grid_start];
        if visible_region.contains(r#"class="benchmark-scores-summary""#) {
            found_visible_summary = true;
            break;
        }
    }

    assert!(
        found_visible_summary,
        "at least one closed group should render its score summary before the hidden chart grid"
    );
    Ok(())
}

/// Every `.chart-card` carries a compact `.toolbar.toolbar--card` so the user
/// has per-chart controls. There is no page-level toolbar, no preset scope
/// button row, and no abs/rel mode toggle.
#[tokio::test]
async fn chart_card_carries_per_chart_toolbar() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;

    let card_count = body.matches(r#"<section class="chart-card""#).count();
    let toolbar_count = body.matches(r#"class="toolbar toolbar--card""#).count();
    let strip_count = body.matches(r#"data-role="range-strip""#).count();
    assert!(card_count > 0, "landing page must render chart cards");
    assert_eq!(
        toolbar_count, card_count,
        "every chart-card must contain a toolbar--card ({card_count} cards / {toolbar_count} toolbars)"
    );
    assert_eq!(
        strip_count, card_count,
        "every chart-card must carry a range-strip below the canvas \
         ({card_count} cards / {strip_count} strips)"
    );
    assert!(
        body.contains(r#"data-role="range-window""#)
            && body.contains(r#"data-role="range-handle-left""#)
            && body.contains(r#"data-role="range-handle-right""#),
        "range-strip must include a draggable window and two resize handles"
    );
    assert!(
        !body.contains(r#"data-mode="#),
        "abs/rel mode buttons should not render"
    );
    assert!(
        !body.contains(r#"data-scope="#),
        "preset scope buttons should not render; use the slider instead"
    );
    assert!(
        body.contains(r#"data-role="scope-slider""#),
        "scope slider should remain available"
    );
    assert!(
        !body.contains(r#"scope-slider-label"#),
        "scope value labels should not add repeated numbers to every card"
    );

    // Same invariant on /chart/{slug}.
    let slug = pick_chart_slug(&server, |s| s.starts_with("TPC-H")).await?;
    let body = client
        .get(server.url(&format!("/chart/{slug}")))
        .send()
        .await?
        .text()
        .await?;
    assert!(
        body.contains(r#"class="toolbar toolbar--card""#),
        "chart page must carry a per-chart toolbar"
    );
    assert!(!body.contains(r#"data-mode="#));
    assert!(!body.contains(r#"data-scope="#));
    assert!(body.contains(r#"data-role="scope-slider""#));
    assert!(!body.contains(r#"scope-slider-label"#));

    // Same invariant on /group/{slug}.
    let group_slug = pick_group_slug(&server, |s| s.starts_with("TPC-H")).await?;
    let body = client
        .get(server.url(&format!("/group/{group_slug}")))
        .send()
        .await?
        .text()
        .await?;
    assert!(
        body.contains(r#"class="toolbar toolbar--card""#),
        "group page must carry per-chart toolbars"
    );
    assert!(!body.contains(r#"data-mode="#));
    assert!(!body.contains(r#"data-scope="#));
    assert!(body.contains(r#"data-role="scope-slider""#));
    assert!(!body.contains(r#"scope-slider-label"#));
    Ok(())
}

/// Landing-page `<details>` summaries appear in the canonical v2 order: the
/// fixture seeds Random Access, Compression, Compression Size, TPC-H, and a
/// vector-search group. The first three are in [`api::GROUP_ORDER`] in the
/// expected positions; TPC-H follows; the unknown vector-search group sorts
/// last (alphabetical fallback after the listed names).
#[tokio::test]
async fn landing_groups_render_in_v2_order() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;

    // Extract group names in render order from the `data-group-name=` attrs.
    let mut names = Vec::new();
    for window in body.split("data-group-name=\"").skip(1) {
        if let Some(end) = window.find('"') {
            names.push(window[..end].to_string());
        }
    }
    let expected = [
        "Random Access",
        "Compression",
        "Compression Size",
        "TPC-H (NVMe) (SF=1)",
        "cohere-large-10m / partitioned",
    ];
    assert_eq!(names, expected, "v2 ordering");
    Ok(())
}

#[tokio::test]
async fn chart_page_snapshot() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    // The query_measurements chart has two series so the snapshot
    // exercises multi-series rendering.
    let slug = pick_chart_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let resp = client
        .get(server.url(&format!("/chart/{slug}")))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body = resp.text().await?;
    assert!(
        body.contains(r#"<script id="chart-data-0" type="application/json">"#),
        "chart payload must be embedded inline"
    );
    assert!(
        body.contains(r#"<script src="/static/chart.umd.js"#),
        "Chart.js must be referenced from the static asset route"
    );
    assert!(
        body.contains(r#"<script src="/static/chartjs-plugin-zoom.umd.min.js"#),
        "zoom plugin must be loaded for /chart pages"
    );
    assert!(
        body.contains(r#"class="toolbar toolbar--card""#),
        "per-chart toolbar must be rendered on chart page"
    );

    insta_settings().bind(|| {
        insta::assert_snapshot!("chart_page_query", body);
    });
    Ok(())
}

#[tokio::test]
async fn group_page_snapshot() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let slug = pick_group_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let resp = client
        .get(server.url(&format!("/group/{slug}")))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body = resp.text().await?;
    assert!(
        body.contains(r#"id="chart-data-0""#),
        "group page must embed at least one chart payload inline"
    );
    assert!(
        body.contains(r#"class="toolbar toolbar--card""#),
        "per-chart toolbar must be rendered on group page"
    );
    insta_settings().bind(|| {
        insta::assert_snapshot!("group_page_query", body);
    });
    Ok(())
}

#[tokio::test]
async fn group_api_returns_charts() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let slug = pick_group_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let client = reqwest::Client::new();
    let resp = client
        .get(server.url(&format!("/api/group/{slug}")))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    let charts = body["charts"].as_array().context("charts is array")?;
    assert!(!charts.is_empty(), "group must have at least one chart");
    let first = &charts[0];
    assert!(first["slug"].as_str().is_some(), "chart slug present");
    assert!(first["name"].as_str().is_some(), "chart name present");
    assert!(
        first["commits"].as_array().is_some(),
        "embedded chart commits"
    );
    assert!(
        first["series"].as_object().is_some(),
        "embedded chart series"
    );
    assert_eq!(
        body["summary"]["type"].as_str(),
        Some("queryBenchmark"),
        "group API should include the server-computed summary"
    );
    Ok(())
}

#[tokio::test]
async fn group_summaries_match_v2_contract() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let groups: Value = client
        .get(server.url("/api/groups"))
        .send()
        .await?
        .json()
        .await?;

    let random_access = &group_by_name(&groups, "Random Access")?["summary"];
    assert_eq!(random_access["type"].as_str(), Some("randomAccess"));
    let rankings = random_access["rankings"]
        .as_array()
        .context("random access rankings")?;
    assert_eq!(rankings[0]["name"].as_str(), Some("vortex-file-compressed"));
    assert_eq!(rankings[1]["name"].as_str(), Some("parquet"));
    assert_close(rankings[1]["ratio"].as_f64().context("random ratio")?, 2.0);

    let compression = &group_by_name(&groups, "Compression")?["summary"];
    assert_eq!(compression["type"].as_str(), Some("compression"));
    assert_close(
        compression["compressRatio"]
            .as_f64()
            .context("compressRatio")?,
        2.0,
    );
    assert_close(
        compression["decompressRatio"]
            .as_f64()
            .context("decompressRatio")?,
        2.0,
    );
    assert_eq!(compression["datasetCount"].as_u64(), Some(1));

    let compression_size = &group_by_name(&groups, "Compression Size")?["summary"];
    assert_eq!(compression_size["type"].as_str(), Some("compressionSize"));
    assert_close(
        compression_size["meanRatio"]
            .as_f64()
            .context("meanRatio")?,
        0.5,
    );
    assert_eq!(compression_size["datasetCount"].as_u64(), Some(1));

    let query = &group_by_name(&groups, "TPC-H (NVMe) (SF=1)")?["summary"];
    assert_eq!(query["type"].as_str(), Some("queryBenchmark"));
    let rankings = query["rankings"].as_array().context("query rankings")?;
    assert_eq!(
        rankings[0]["name"].as_str(),
        Some("datafusion:vortex-file-compressed"),
        "query summary should include v2's missing-series penalty"
    );
    assert_eq!(rankings[1]["name"].as_str(), Some("duckdb:parquet"));
    let first_score = rankings[0]["score"].as_f64().context("first score")?;
    let second_score = rankings[1]["score"].as_f64().context("second score")?;
    assert!(
        first_score < second_score,
        "lower query score should rank first"
    );

    Ok(())
}

/// `GET /api/chart/{slug}` returns the JSON shape the chart-init.js fetches
/// on lazy-load (closed-by-default `<details>` groups).
#[tokio::test]
async fn chart_api_returns_payload_shape() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let slug = pick_chart_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let client = reqwest::Client::new();
    let resp = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        ct.starts_with("application/json"),
        "expected JSON content-type, got {ct:?}"
    );
    let body: Value = resp.json().await?;
    assert!(body["display_name"].as_str().is_some(), "display_name");
    assert!(body["unit"].as_str().is_some(), "unit");
    assert!(body["commits"].as_array().is_some(), "commits");
    assert!(body["series"].as_object().is_some(), "series");

    // ?n= narrows the commit count.
    let one: Value = client
        .get(server.url(&format!("/api/chart/{slug}?n=1")))
        .send()
        .await?
        .json()
        .await?;
    let one_count = one["commits"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(one_count, 1, "?n=1 should keep exactly one commit");

    // A valid-shape slug for a chart that doesn't exist in the DB returns
    // 404. (Malformed slugs are 400, covered by the HTML route tests.)
    let unknown_slug = vortex_bench_server::slug::ChartKey::QueryMeasurement {
        dataset: "missing-dataset".into(),
        dataset_variant: None,
        scale_factor: None,
        storage: "nvme".into(),
        query_idx: 99,
    }
    .to_slug();
    let missing = client
        .get(server.url(&format!("/api/chart/{unknown_slug}")))
        .send()
        .await?;
    assert_eq!(missing.status(), 404);
    Ok(())
}

#[tokio::test]
async fn chart_page_window_caps_commits() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let slug = pick_chart_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let client = reqwest::Client::new();
    // Without ?n, default is the 1000-commit per-chart cap — fixture has 3.
    let full: Value = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?
        .json()
        .await?;
    let full_count = full["commits"].as_array().map(|a| a.len()).unwrap_or(0);
    assert!(full_count >= 3, "expected ≥ 3 commits, got {full_count}");

    // ?n=1 should cap to one commit.
    let one: Value = client
        .get(server.url(&format!("/api/chart/{slug}?n=1")))
        .send()
        .await?
        .json()
        .await?;
    let one_count = one["commits"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(one_count, 1, "?n=1 should keep exactly one commit");

    // ?n=all bypasses the cap.
    let all: Value = client
        .get(server.url(&format!("/api/chart/{slug}?n=all")))
        .send()
        .await?
        .json()
        .await?;
    let all_count = all["commits"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(all_count, full_count, "?n=all should match unbounded view");

    // Malformed ?n gracefully falls back to default.
    let bad = client
        .get(server.url(&format!("/api/chart/{slug}?n=banana")))
        .send()
        .await?;
    assert_eq!(bad.status(), 200);
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
            body.contains(r#"id="chart-data-0""#),
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
        (
            "/static/chartjs-plugin-zoom.umd.min.js",
            "application/javascript",
        ),
        ("/static/chart-init.js", "application/javascript"),
        ("/static/style.css", "text/css"),
        ("/vortex_black_nobg.svg", "image/svg+xml"),
        ("/vortex_white_nobg.svg", "image/svg+xml"),
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
        let cache_control = resp
            .headers()
            .get(reqwest::header::CACHE_CONTROL)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(
            cache_control.contains("no-cache"),
            "GET {path}: static assets should revalidate so UI CSS/JS changes are not stale"
        );
        let bytes = resp.bytes().await?;
        assert!(!bytes.is_empty(), "GET {path}: body must not be empty");
    }
    Ok(())
}
