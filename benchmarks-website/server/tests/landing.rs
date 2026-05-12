// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for the landing page (`GET /`).

mod common;

use anyhow::Context as _;
use anyhow::Result;
use serde_json::Value;

use self::common::Server;
use self::common::attr_value;
use self::common::extract_chip;
use self::common::filter_bar_section;
use self::common::filter_section;
use self::common::insta_settings;
use self::common::pick_chart_slug;
use self::common::pick_group_slug;
use self::common::seed;
use self::common::seed_long_history;

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

    // Canvas shells render immediately, but chart data comes from
    // versioned group shard artifacts instead of inline JSON.
    assert!(
        body.contains("<canvas"),
        "landing page must render at least one <canvas>"
    );
    assert!(
        !body.contains(r#"id="chart-data-0""#),
        "landing page should not inline chart payloads"
    );
    assert!(
        body.contains(r#"data-chart-slug="#),
        "every chart card carries data-chart-slug for the lazy-fetch path"
    );
    assert!(
        body.contains(r#"data-group-slug="#),
        "every group carries data-group-slug as stable metadata"
    );
    assert!(
        body.contains(r#"data-artifact-generation="#)
            && body.contains(r#"data-group-shard-count="#)
            && body.contains(r#"data-group-shard-prefix="#),
        "every group should carry versioned shard hydration metadata"
    );
    assert!(
        attr_value(&body, "data-artifact-generation").is_some_and(|v| !v.is_empty()),
        "artifact generation should be non-empty"
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
        body.contains(r#"Vortex_Black_NoBG.png"#) && body.contains(r#"Vortex_White_NoBG.png"#),
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
/// to expand. Chart payloads are intentionally not inlined; the disclosure
/// carries shard metadata so JS can fetch the materialized latest-100
/// artifact on intent/open.
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
    assert!(
        !body.contains(r#"id="chart-data-0""#),
        "landing page should hydrate charts from materialized artifacts",
    );
    assert!(
        body.contains(r#"data-group-shard-count="#),
        "closed groups should still carry shard metadata",
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
/// vector-search group. The first three are in `api::GROUP_ORDER` in the
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

/// The landing page does not inline chart JSON. Its first materialized shard
/// caps chart payloads at 100 commits regardless of `?n=`; power users get
/// full history via the explicit `/api/chart/{slug}?n=all` refetch.
#[tokio::test]
async fn landing_first_group_shard_caps_commits() -> Result<()> {
    // 250 commits is comfortably above the 100-commit artifact cap so the
    // cap actually kicks in. `seed_long_history` only seeds the Random-Access
    // group; with the canonical group ordering Random Access sorts first.
    let server = Server::start().await?;
    seed_long_history(&server, 250).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;
    assert!(
        !body.contains(r#"id="chart-data-0""#),
        "landing page should not inline chart JSON"
    );

    let generation = attr_value(&body, "data-artifact-generation")
        .context("landing exposes artifact generation")?;
    let group_slug = attr_value(&body, "data-group-slug").context("landing exposes group slug")?;
    let shard: Value = client
        .get(server.url(&format!(
            "/api/artifacts/{generation}/groups/{group_slug}/shards/0"
        )))
        .send()
        .await?
        .json()
        .await?;
    let commits = shard["charts"][0]["commits"]
        .as_array()
        .context("shard chart commits array")?;
    assert!(
        commits.len() <= 100,
        "landing shard must cap commits at 100, \
         got {}",
        commits.len(),
    );
    // Sanity check: the cap actually fired on this fixture (≥ 100 commits
    // seeded). Without this we'd silently regress to "always small fixture".
    assert_eq!(
        commits.len(),
        100,
        "with 250 seeded commits the shard payload should be exactly the \
         100-commit cap; got {}",
        commits.len(),
    );

    // ?n=all on the URL still parses without panicking and still leaves the
    // landing page as shell-only metadata.
    let body_all = client
        .get(server.url("/?n=all"))
        .send()
        .await?
        .text()
        .await?;
    assert!(
        !body_all.contains(r#"id="chart-data-0""#),
        "?n=all on the landing page must not inline full-history chart data"
    );
    Ok(())
}

/// Sanity smoke test: round-trip every chart slug `/api/groups` returns
/// through `/chart/{slug}` to make sure each slug shape's HTML route is
/// wired up.
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
