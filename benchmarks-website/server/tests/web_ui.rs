// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for the web-ui HTML routes.
//!
//! Builds a temp DuckDB via the same `/api/ingest` path real callers use,
//! seeds it with a multi-commit fixture so chart series have more than one
//! point, then snapshots the rendered HTML for each route plus a chart slug
//! round-trip.

use std::io::Read as _;
use std::net::SocketAddr;

use anyhow::Context as _;
use anyhow::Result;
use flate2::read::GzDecoder;
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

/// Slim ingest envelope carrying just a `random_access_time` pair so we can
/// drive a long-history fixture cheaply (the full envelope is ~12 records;
/// this is two). Used by the downsample tests.
fn ra_envelope_for(sha: &str, ts: &str, msg: &str, bias: i64) -> Value {
    json!({
        "run_meta": {
            "benchmark_id": "downsample-fixture",
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
                "kind": "random_access_time",
                "commit_sha": sha,
                "dataset": "taxi",
                "format": "vortex-file-compressed",
                "value_ns": 500 + bias,
                "all_runtimes_ns": [500 + bias]
            },
            {
                "kind": "random_access_time",
                "commit_sha": sha,
                "dataset": "taxi",
                "format": "parquet",
                "value_ns": 1_000 + (2 * bias),
                "all_runtimes_ns": [1_000 + (2 * bias)]
            }
        ]
    })
}

/// Seed a `Random Access` chart with `n` synthetic commits so the
/// downsampler has something to chew on. SHAs are deterministic
/// `{i:040x}`; timestamps are 1 minute apart starting 2025-01-01 so the
/// commits sort stably.
async fn seed_long_history(server: &Server, n: usize) -> Result<()> {
    let client = reqwest::Client::new();
    for i in 0..n {
        let sha = format!("{i:040x}");
        let minutes = i;
        let ts = format!(
            "2025-01-01T{:02}:{:02}:00Z",
            (minutes / 60) % 24,
            minutes % 60
        );
        // Sinusoidal bias so the series has interior peaks LTTB will retain.
        let bias = ((i as f64).sin() * 1_000.0) as i64 + i as i64 * 10;
        let resp = client
            .post(server.url("/api/ingest"))
            .bearer_auth(TOKEN)
            .json(&ra_envelope_for(&sha, &ts, "synthetic", bias))
            .send()
            .await?;
        anyhow::ensure!(
            resp.status().is_success(),
            "long-history ingest #{i} failed: {}",
            resp.status()
        );
    }
    Ok(())
}

/// Pull the inline `<script id="chart-data-N">…</script>` JSON out of an
/// HTML body. Returns `None` if the script tag isn't present.
fn extract_chart_data(body: &str, idx: usize) -> Option<Value> {
    let needle = format!(r#"<script id="chart-data-{idx}" type="application/json">"#);
    let start = body.find(&needle)? + needle.len();
    let end = body[start..].find("</script>")? + start;
    // Reverse the `</` neutralisation done by `escape_json_for_script`.
    let json = body[start..end].replace(r"<\/", "</");
    serde_json::from_str(&json).ok()
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

/// All group disclosures render closed by default — the user picks which
/// to expand. The first group's chart payloads are still inlined in the
/// HTML (so opening it skips the JS fetch), but the disclosure itself
/// stays collapsed until clicked.
#[tokio::test]
async fn details_all_groups_closed_by_default() -> Result<()> {
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
    for (i, is_open) in opens.iter().enumerate() {
        assert!(!is_open, "group #{i} must be closed by default");
    }
    // The first group's chart payload should still be inlined — fast
    // hydration on toggle without a network round-trip.
    assert!(
        body.contains(r#"id="chart-data-0""#),
        "first group's chart payload should be inlined for fast on-toggle hydration",
    );
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
    // Without `?n`, the API default is `Last(DEFAULT_COMMIT_WINDOW)`. The
    // fixture has 3 commits which fits comfortably.
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

    // ?n=all returns the unbounded view (the per-chart hard cap is gone).
    let all: Value = client
        .get(server.url(&format!("/api/chart/{slug}?n=all")))
        .send()
        .await?
        .json()
        .await?;
    let all_count = all["commits"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(all_count, full_count, "?n=all should match unbounded view");

    // Even very large `?n` survives without being clamped.
    let huge: Value = client
        .get(server.url(&format!("/api/chart/{slug}?n=99999")))
        .send()
        .await?
        .json()
        .await?;
    let huge_count = huge["commits"].as_array().map(|a| a.len()).unwrap_or(0);
    assert_eq!(
        huge_count, full_count,
        "?n=99999 should no longer be clamped to 1000"
    );

    // Malformed ?n gracefully falls back to default.
    let bad = client
        .get(server.url(&format!("/api/chart/{slug}?n=banana")))
        .send()
        .await?;
    assert_eq!(bad.status(), 200);
    Ok(())
}

/// `/chart/{slug}` and `/group/{slug}` permalinks default to the unbounded
/// commit window, and the inlined JSON payload contains the full raw
/// history (no server-side downsampling). Visual downsampling now lives in
/// `chart-init.js` and runs on the *visible* commit range only.
#[tokio::test]
async fn permalink_pages_inline_full_raw_history() -> Result<()> {
    let server = Server::start().await?;
    seed_long_history(&server, 200).await?;

    let chart_slug = pick_chart_slug(&server, |s| s == "Random Access").await?;
    let group_slug = pick_group_slug(&server, |s| s == "Random Access").await?;
    let client = reqwest::Client::new();

    let chart_body = client
        .get(server.url(&format!("/chart/{chart_slug}")))
        .send()
        .await?
        .text()
        .await?;
    let chart_payload =
        extract_chart_data(&chart_body, 0).context("chart inline payload present")?;
    assert_eq!(
        chart_payload["commits"]
            .as_array()
            .context("commits is array")?
            .len(),
        200,
        "/chart permalink should inline the full raw history",
    );

    let group_body = client
        .get(server.url(&format!("/group/{group_slug}")))
        .send()
        .await?
        .text()
        .await?;
    let group_payload =
        extract_chart_data(&group_body, 0).context("group inline payload present")?;
    assert_eq!(
        group_payload["commits"]
            .as_array()
            .context("commits is array")?
            .len(),
        200,
        "/group permalink should inline the full raw history",
    );

    Ok(())
}

/// The wire payload no longer carries a `raw_commit_count` field — visual
/// downsampling moved to the client, so the server has no opinion on
/// rendered point count.
#[tokio::test]
async fn chart_payload_does_not_carry_raw_commit_count() -> Result<()> {
    let server = Server::start().await?;
    seed_long_history(&server, 50).await?;

    let slug = pick_chart_slug(&server, |s| s == "Random Access").await?;
    let client = reqwest::Client::new();
    let body: Value = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?
        .json()
        .await?;
    assert!(
        body.get("raw_commit_count").is_none(),
        "raw_commit_count should not appear on the wire; got {body:?}"
    );
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

/// Landing page renders the global filter dropdown inside the sticky
/// header, with chip rows for engine and format sourced from the seeded
/// data — no hard-coding.
#[tokio::test]
async fn landing_page_renders_global_filter_bar() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;

    // The dropdown lives inside the sticky header so it stays on-screen
    // while the user scrolls.
    let header_chunk = body
        .split(r#"class="sticky-header""#)
        .nth(1)
        .and_then(|s| s.split("</header>").next())
        .context("sticky header chunk")?;
    assert!(
        header_chunk.contains(r#"data-role="global-filter-bar""#),
        "filter dropdown must live inside the sticky header"
    );
    assert!(header_chunk.contains(r#"data-role="filter-trigger""#));
    assert!(header_chunk.contains(r#"data-role="filter-panel""#));
    assert!(header_chunk.contains(r#"data-filter="engine""#));
    assert!(header_chunk.contains(r#"data-filter="format""#));
    // Engines + formats from the seed fixture must appear as chips.
    assert!(body.contains(r#"data-value="datafusion""#));
    assert!(body.contains(r#"data-value="duckdb""#));
    assert!(body.contains(r#"data-value="vortex-file-compressed""#));
    assert!(body.contains(r#"data-value="parquet""#));
    // Both rows have an "all" reset chip.
    assert!(body.matches(r#"data-value="*""#).count() >= 2);
    // The "all" chip is now a one-shot reset and is never rendered active —
    // active chips reflect the visible engine/format set.
    assert!(
        !body.contains(r#"class="filter-chip filter-chip--all filter-chip--active""#),
        "the 'all' chip should never start active"
    );
    // No filter applied by default → every specific chip is active.
    let engine_section = filter_section(&body, "engine");
    for engine in ["datafusion", "duckdb"] {
        assert!(
            extract_chip(&engine_section, engine).contains("filter-chip--active"),
            "engine chip {engine} should be active when no filter is applied"
        );
    }
    // No badge on the trigger when nothing is hidden.
    assert!(
        !body.contains(r#"data-role="filter-badge""#),
        "filter badge should be absent when no chips are off"
    );
    // Embedded filter state JSON for the client to pick up.
    assert!(body.contains(r#"id="bench-filter-state""#));

    insta_settings().bind(|| {
        insta::assert_snapshot!("landing_page_filter_bar", filter_bar_section(&body));
    });
    Ok(())
}

/// Landing page honours `?engine=`/`?format=` and reflects them as the
/// active chip set + initial filter-state JSON, so a refresh preserves view.
#[tokio::test]
async fn landing_page_honours_filter_query_params() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client
        .get(server.url("/?engine=duckdb&format=vortex-file-compressed"))
        .send()
        .await?
        .text()
        .await?;

    assert!(
        body.contains(r#"{"engines":["duckdb"],"formats":["vortex-file-compressed"]}"#),
        "filter state JSON should reflect query params"
    );
    let engine_section = filter_section(&body, "engine");
    assert!(
        engine_section.contains(r#"data-value="duckdb""#)
            && extract_chip(&engine_section, "duckdb").contains("filter-chip--active"),
        "duckdb chip should be active"
    );
    assert!(
        !extract_chip(&engine_section, "datafusion").contains("filter-chip--active"),
        "datafusion chip should NOT be active when engine=duckdb"
    );
    assert!(
        !extract_chip(&engine_section, "*").contains("filter-chip--active"),
        "the 'all' chip is a reset, never active"
    );
    // Trigger should show a badge counting the off chips (1 engine + 1 format).
    assert!(
        body.contains(r#"data-role="filter-badge""#),
        "trigger should render a badge when chips are filtered off"
    );
    Ok(())
}

/// Permalink pages render the same filter dropdown in the navbar (so the
/// user can adjust visibility from any page) and embed the filter-state
/// JSON so chart-init.js applies the filter on hydration.
#[tokio::test]
async fn permalink_pages_embed_filter_state() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let chart_slug = pick_chart_slug(&server, |s| s.starts_with("TPC-H")).await?;
    let group_slug = pick_group_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let chart_body = client
        .get(server.url(&format!("/chart/{chart_slug}?engine=duckdb&format=parquet")))
        .send()
        .await?
        .text()
        .await?;
    assert!(
        chart_body.contains(r#"id="bench-filter-state""#),
        "chart permalink must embed filter state"
    );
    assert!(
        chart_body.contains(r#"{"engines":["duckdb"],"formats":["parquet"]}"#),
        "chart permalink must echo the query-param filter state"
    );

    let group_body = client
        .get(server.url(&format!("/group/{group_slug}?engine=duckdb")))
        .send()
        .await?
        .text()
        .await?;
    assert!(
        group_body.contains(r#"{"engines":["duckdb"],"formats":[]}"#),
        "group permalink must echo the query-param filter state"
    );
    Ok(())
}

/// Chart payload exposes per-series engine/format tags so the global filter
/// has the metadata it needs to drive bulk hide/show.
#[tokio::test]
async fn chart_payload_includes_series_meta() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let slug = pick_chart_slug(&server, |s| s.starts_with("TPC-H")).await?;
    let client = reqwest::Client::new();
    let body: Value = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?
        .json()
        .await?;

    let meta = body["series_meta"]
        .as_object()
        .context("series_meta must be present for query measurements")?;
    let row = meta
        .get("datafusion:vortex-file-compressed")
        .context("expected series tag")?;
    assert_eq!(row["engine"].as_str(), Some("datafusion"));
    assert_eq!(row["format"].as_str(), Some("vortex-file-compressed"));

    // Compression-time series carry a format tag but no engine.
    let comp_slug = pick_chart_slug(&server, |s| s == "Compression").await?;
    let comp_body: Value = client
        .get(server.url(&format!("/api/chart/{comp_slug}")))
        .send()
        .await?
        .json()
        .await?;
    let comp_meta = comp_body["series_meta"]
        .as_object()
        .context("series_meta must be present for compression times")?;
    let row = comp_meta
        .get("vortex-file-compressed:encode")
        .context("expected encode series tag")?;
    assert!(row["engine"].is_null() || row.get("engine").is_none());
    assert_eq!(row["format"].as_str(), Some("vortex-file-compressed"));
    Ok(())
}

/// Pull just the `<div class="filter-dropdown" …>…</div>` substring of the
/// filter dropdown — its trigger button and the chip panel. Keeps the
/// snapshot focused on the chip markup and stable against changes elsewhere
/// on the page.
fn filter_bar_section(body: &str) -> String {
    let needle = r#"<div class="filter-dropdown" data-role="global-filter-bar""#;
    let Some(start) = body.find(needle) else {
        return "<missing filter bar>".to_string();
    };
    let tail = &body[start..];
    // The dropdown is `<div ...><button>...</button><div class="filter-panel">...</div></div>`.
    // We need to find the matching `</div>` for the outer wrapper. The
    // simplest robust approach is to scan and balance.
    let bytes = tail.as_bytes();
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if tail[i..].starts_with("<div") {
                depth += 1;
                i += 4;
                continue;
            }
            if tail[i..].starts_with("</div>") {
                depth -= 1;
                if depth == 0 {
                    return tail[..i + "</div>".len()].to_string();
                }
                i += "</div>".len();
                continue;
            }
        }
        i += 1;
    }
    tail.to_string()
}

/// Pull the `<div class="global-filter-row">` containing chips for one
/// dimension (`"engine"` or `"format"`).
fn filter_section(body: &str, dim: &str) -> String {
    let bar = filter_bar_section(body);
    let needle = format!(r#"data-filter="{dim}""#);
    let Some(_) = bar.find(&needle) else {
        return String::new();
    };
    // Walk back to the enclosing `<div class="global-filter-row">`.
    let row_open = r#"<div class="global-filter-row">"#;
    let row_close = "</div>";
    bar.split(row_open)
        .find(|chunk| chunk.contains(&needle))
        .and_then(|chunk| chunk.split(row_close).next())
        .map(str::to_string)
        .unwrap_or_default()
}

/// Pull a single chip's opening tag for assertions.
fn extract_chip(section: &str, value: &str) -> String {
    let needle = format!(r#"data-value="{value}""#);
    let Some(idx) = section.find(&needle) else {
        return String::new();
    };
    let head = &section[..idx];
    let chip_start = head.rfind("<button").unwrap_or(0);
    let tail = &section[chip_start..];
    let chip_end = tail.find('>').map(|p| p + 1).unwrap_or(tail.len());
    tail[..chip_end].to_string()
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

/// Every response — landing HTML, chart JSON, bundled JS — flows through
/// `tower-http`'s `CompressionLayer` so a client advertising
/// `Accept-Encoding: gzip` gets a gzipped (or brotli) body. The
/// reqwest dev-dependency is built without `gzip`/`brotli` features, so the
/// transport hands us the compressed bytes verbatim and we can both inspect
/// the `content-encoding` response header and decompress the body manually
/// to confirm it matches the uncompressed snapshot.
#[tokio::test]
async fn responses_are_compressed_when_client_accepts_gzip() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();

    // 1. Landing HTML.
    let plain_body = client.get(server.url("/")).send().await?.text().await?;
    let resp = client
        .get(server.url("/"))
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let encoding = resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        encoding, "gzip",
        "GET / with Accept-Encoding: gzip should respond with gzip"
    );
    let compressed = resp.bytes().await?;
    assert!(
        compressed.len() < plain_body.len(),
        "compressed body ({} B) should be smaller than plain body ({} B)",
        compressed.len(),
        plain_body.len(),
    );
    let mut decoded = String::new();
    GzDecoder::new(&compressed[..]).read_to_string(&mut decoded)?;
    assert_eq!(
        decoded, plain_body,
        "gzipped landing body should decompress to the uncompressed body"
    );

    // 2. Bundled JS — the heaviest static asset; gzip is the whole point.
    let plain_js = client
        .get(server.url("/static/chart.umd.js"))
        .send()
        .await?
        .bytes()
        .await?;
    let js_resp = client
        .get(server.url("/static/chart.umd.js"))
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    assert_eq!(js_resp.status(), 200);
    let js_encoding = js_resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        js_encoding, "gzip",
        "/static/chart.umd.js must compress so the cold load isn't dominated by ~200KB of JS"
    );
    let compressed_js = js_resp.bytes().await?;
    let mut decoded_js = Vec::new();
    GzDecoder::new(&compressed_js[..]).read_to_end(&mut decoded_js)?;
    assert_eq!(
        decoded_js,
        plain_js.as_ref(),
        "decompressed chart.umd.js should match the unencoded body byte-for-byte"
    );

    // 3. Brotli is also offered when the client prefers it.
    let br_resp = client
        .get(server.url("/"))
        .header(reqwest::header::ACCEPT_ENCODING, "br")
        .send()
        .await?;
    assert_eq!(br_resp.status(), 200);
    let br_encoding = br_resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert_eq!(
        br_encoding, "br",
        "GET / with Accept-Encoding: br should respond with brotli"
    );

    Ok(())
}
