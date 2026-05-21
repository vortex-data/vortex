// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for `/group/{slug}` and `/api/group/{slug}` plus the
//! v2-compatible group summary contract on `/api/groups`.

mod common;

use anyhow::Context as _;
use anyhow::Result;
use serde_json::Value;

use self::common::Server;
use self::common::assert_close;
use self::common::attr_value;
use self::common::group_by_name;
use self::common::insta_settings;
use self::common::pick_group_slug;
use self::common::seed;
use self::common::seed_long_history;

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
        !body.contains(r#"id="chart-data-0""#),
        "group page should hydrate through materialized shards, not inline chart payloads"
    );
    assert!(
        body.contains(r#"data-artifact-generation="#)
            && body.contains(r#"data-group-shard-prefix="#)
            && body.contains(r#"open"#),
        "group page should render an open shard-hydrated group shell"
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
async fn group_api_respects_commit_window() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let slug = pick_group_slug(&server, |s| s.starts_with("TPC-H")).await?;

    let client = reqwest::Client::new();
    let resp = client
        .get(server.url(&format!("/api/group/{slug}?n=1")))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await?;
    let charts = body["charts"].as_array().context("charts is array")?;
    assert!(!charts.is_empty(), "group must have at least one chart");
    for chart in charts {
        let commits = chart["commits"].as_array().context("commits is array")?;
        assert!(
            commits.len() <= 1,
            "group hydration should honor the requested commit window"
        );
        assert!(chart["series"].as_object().is_some(), "series is present");
    }
    Ok(())
}

#[tokio::test]
async fn group_shard_artifact_returns_bounded_chart_payloads() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let landing = client.get(server.url("/")).send().await?.text().await?;
    let generation = attr_value(&landing, "data-artifact-generation")
        .context("landing exposes artifact generation")?;
    let group_slug =
        attr_value(&landing, "data-group-slug").context("landing exposes group slug")?;

    let resp = client
        .get(server.url(&format!(
            "/api/artifacts/{generation}/groups/{group_slug}/shards/0"
        )))
        .send()
        .await?;
    assert_eq!(resp.status(), 200);
    let etag = resp
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(!etag.is_empty(), "artifact response should carry an ETag");
    let vary = resp
        .headers()
        .get(reqwest::header::VARY)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    assert!(
        vary.contains("accept-encoding"),
        "artifact response should vary on Accept-Encoding"
    );

    let body: Value = resp.json().await?;
    assert_eq!(body["window"].as_u64(), Some(100));
    assert_eq!(body["shard_index"].as_u64(), Some(0));
    assert!(
        body["shard_count"].as_u64().unwrap_or_default() >= 1,
        "shard count should be present"
    );
    let charts = body["charts"].as_array().context("charts is array")?;
    assert!(!charts.is_empty(), "first shard should include charts");
    assert!(charts.len() <= 8, "shards should carry at most 8 charts");
    for chart in charts {
        let commits = chart["commits"].as_array().context("commits is array")?;
        assert!(
            commits.len() <= 100,
            "artifact hydration should use the latest-100 window"
        );
        assert!(chart["series"].as_object().is_some(), "series is present");
    }
    Ok(())
}

#[tokio::test]
async fn group_shard_artifact_carries_chart_history_metadata() -> Result<()> {
    let server = Server::start().await?;
    seed_long_history(&server, 125).await?;

    let client = reqwest::Client::new();
    let landing = client.get(server.url("/")).send().await?.text().await?;
    let generation = attr_value(&landing, "data-artifact-generation")
        .context("landing exposes artifact generation")?;
    let group_slug =
        attr_value(&landing, "data-group-slug").context("landing exposes group slug")?;

    let body: Value = client
        .get(server.url(&format!(
            "/api/artifacts/{generation}/groups/{group_slug}/shards/0"
        )))
        .send()
        .await?
        .json()
        .await?;
    let first_chart = body["charts"][0]
        .as_object()
        .context("first chart in shard")?;
    assert_eq!(
        first_chart["commits"].as_array().map(Vec::len),
        Some(100),
        "group shard remains a latest-100 payload"
    );
    assert_eq!(first_chart["history"]["total_commits"].as_u64(), Some(125));
    assert_eq!(first_chart["history"]["start_index"].as_u64(), Some(25));
    assert_eq!(first_chart["history"]["loaded_commits"].as_u64(), Some(100));
    assert_eq!(first_chart["history"]["complete"].as_bool(), Some(false));
    Ok(())
}

#[tokio::test]
async fn unknown_group_shard_artifact_returns_404() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let resp = client
        .get(server.url("/api/artifacts/not-a-generation/groups/not-a-group/shards/0"))
        .send()
        .await?;
    assert_eq!(resp.status(), 404);
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
