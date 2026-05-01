// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for `/chart/{slug}` and `/api/chart/{slug}`.

mod common;

use anyhow::Context as _;
use anyhow::Result;
use serde_json::Value;

use self::common::Server;
use self::common::insta_settings;
use self::common::pick_chart_slug;
use self::common::seed;
use self::common::seed_long_history;

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
