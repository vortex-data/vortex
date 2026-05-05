// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for `/chart/{slug}` and `/group/{slug}` permalink
//! behaviour: full-history payloads, embedded filter state, 404s on
//! unknown slugs.

mod common;

use anyhow::Context as _;
use anyhow::Result;

use self::common::Server;
use self::common::extract_chart_data;
use self::common::pick_chart_slug;
use self::common::pick_group_slug;
use self::common::seed;
use self::common::seed_long_history;

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
