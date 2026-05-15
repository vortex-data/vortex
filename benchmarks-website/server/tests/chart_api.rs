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
    assert_eq!(
        body["unit_kind"].as_str(),
        Some("time_ns"),
        "TPC-H query timing chart must declare its base unit as time_ns"
    );
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
async fn default_chart_api_serves_materialized_encoded_artifact() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let slug = pick_chart_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let client = reqwest::Client::new();
    let resp = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get(reqwest::header::CONTENT_ENCODING)
            .and_then(|v| v.to_str().ok()),
        Some("gzip"),
        "default latest-100 chart endpoint should serve precompressed gzip"
    );
    assert!(
        resp.headers()
            .get(reqwest::header::VARY)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.eq_ignore_ascii_case("Accept-Encoding")),
        "materialized artifact should vary on Accept-Encoding"
    );
    let etag = resp
        .headers()
        .get(reqwest::header::ETAG)
        .cloned()
        .context("materialized chart ETag")?;
    assert!(
        resp.headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .is_some(),
        "materialized artifact should carry Content-Length"
    );

    let not_modified = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .header(reqwest::header::IF_NONE_MATCH, etag)
        .send()
        .await?;
    assert_eq!(not_modified.status(), 304);
    Ok(())
}

#[tokio::test]
async fn chart_api_reports_virtual_history_for_bounded_and_full_windows() -> Result<()> {
    let server = Server::start().await?;
    seed_long_history(&server, 125).await?;

    let slug = pick_chart_slug(&server, |s| s == "Random Access").await?;
    let client = reqwest::Client::new();

    let bounded: Value = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(
        bounded["commits"].as_array().map(Vec::len),
        Some(100),
        "default materialized chart payload should stay latest-100"
    );
    assert_eq!(bounded["history"]["total_commits"].as_u64(), Some(125));
    assert_eq!(bounded["history"]["start_index"].as_u64(), Some(25));
    assert_eq!(bounded["history"]["loaded_commits"].as_u64(), Some(100));
    assert_eq!(bounded["history"]["complete"].as_bool(), Some(false));

    let all: Value = client
        .get(server.url(&format!("/api/chart/{slug}?n=all")))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(all["commits"].as_array().map(Vec::len), Some(125));
    assert_eq!(all["history"]["total_commits"].as_u64(), Some(125));
    assert_eq!(all["history"]["start_index"].as_u64(), Some(0));
    assert_eq!(all["history"]["loaded_commits"].as_u64(), Some(125));
    assert_eq!(all["history"]["complete"].as_bool(), Some(true));
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

/// Every chart payload must declare a structured `unit_kind` so the client
/// can pick a display unit without guessing from the values. The taxonomy
/// lives on [`vortex_bench_server::api::UnitKind`]; this test pins the wire
/// classification of every fact-table family currently emitted.
#[tokio::test]
async fn chart_payload_declares_unit_kind_per_family() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;
    let client = reqwest::Client::new();

    // Each (group-name needle, expected unit_kind) pair pins the
    // classification for one fact-table family. If the underlying base unit
    // ever changes, this test fails loudly and the unit picker in
    // `chart-init.js` must be updated to match. Group names are matched as
    // `name == needle || name.starts_with(needle)` so `"TPC-H"` covers every
    // SF/storage variant and `"Compression"` exact-matches the sole
    // compression-time group.
    let cases = [
        ("query_measurement", "TPC-H", "time_ns"),
        ("compression_time", "Compression", "time_ns"),
        ("compression_size", "Compression Size", "bytes"),
        ("random_access_time", "Random Access", "time_ns"),
    ];
    for (label, needle, expected) in cases {
        let slug =
            pick_chart_slug(&server, |name| name == needle || name.starts_with(needle)).await?;
        let body: Value = client
            .get(server.url(&format!("/api/chart/{slug}")))
            .send()
            .await?
            .json()
            .await?;
        assert_eq!(
            body["unit_kind"].as_str(),
            Some(expected),
            "{label} chart must report unit_kind={expected}, got {:?}",
            body["unit_kind"],
        );
        // The legacy free-form `unit` string was removed; only `unit_kind`
        // travels on the wire now.
        assert!(
            body.get("unit").is_none(),
            "{label} chart wire payload must not carry the legacy `unit` field, got {:?}",
            body.get("unit"),
        );
    }
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
