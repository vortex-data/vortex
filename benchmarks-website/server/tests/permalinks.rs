// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for `/chart/{slug}` and `/group/{slug}` permalink
//! behaviour: full-history payloads, embedded filter state, 404s on
//! unknown slugs.

mod common;

use anyhow::Context as _;
use anyhow::Result;

use self::common::Server;
use self::common::attr_value;
use self::common::extract_chart_data;
use self::common::pick_chart_slug;
use self::common::pick_group_slug;
use self::common::seed;
use self::common::seed_long_history;

/// `/chart/{slug}` defaults to the materialized latest-100 window and
/// upgrades to full raw history only with `?n=all`. `/group/{slug}` renders
/// shell markup and hydrates the same latest-100 materialized shards as the
/// landing page.
#[tokio::test]
async fn permalink_pages_default_to_latest_100_and_opt_into_full_history() -> Result<()> {
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
        100,
        "/chart permalink should default to the latest-100 materialized window",
    );

    let chart_all_body = client
        .get(server.url(&format!("/chart/{chart_slug}?n=all")))
        .send()
        .await?
        .text()
        .await?;
    let chart_all_payload =
        extract_chart_data(&chart_all_body, 0).context("chart all inline payload present")?;
    assert_eq!(
        chart_all_payload["commits"]
            .as_array()
            .context("all commits is array")?
            .len(),
        200,
        "/chart?n=all should inline the full raw history",
    );

    let group_body = client
        .get(server.url(&format!("/group/{group_slug}")))
        .send()
        .await?
        .text()
        .await?;
    assert!(
        !group_body.contains(r#"id="chart-data-0""#),
        "/group permalink should not inline chart payloads"
    );
    let generation = attr_value(&group_body, "data-artifact-generation")
        .context("group page exposes artifact generation")?;
    let shard_prefix = attr_value(&group_body, "data-group-shard-prefix")
        .context("group page exposes shard prefix")?;
    let shard_path = shard_prefix
        .strip_prefix('/')
        .map(|s| format!("/{s}0"))
        .context("absolute shard prefix")?;
    let shard: serde_json::Value = client
        .get(server.url(&shard_path))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(
        shard["charts"][0]["commits"]
            .as_array()
            .context("shard commits is array")?
            .len(),
        100,
        "/group permalink should hydrate the latest-100 shard",
    );
    assert!(
        shard_path.contains(&generation),
        "group shard URL should be versioned by generation"
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
