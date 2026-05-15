// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests for the v3 web UI features that span the API and
//! HTML routes:
//!
//! - **Per-group hover descriptions** (Task A): editorial blurbs port from
//!   v2's `BENCHMARK_DESCRIPTIONS` + `getBenchmarkDescription`. Asserted on
//!   the landing page and on the `/group/{slug}` permalink.
//! - **Partial-coverage commits** (Task B): a chart's x-axis includes
//!   commits that have NO row in the fact table for this chart, so
//!   missing measurements render as visible gaps rather than silently
//!   bridged lines.

mod common;

use anyhow::Context as _;
use anyhow::Result;
use serde_json::Value;
use serde_json::json;

use self::common::Server;
use self::common::TOKEN;
use self::common::pick_chart_slug;
use self::common::pick_group_slug;
use self::common::seed;
use self::common::wait_for_materialized_first_chart_commits;

// =============================================================================
// Task A — per-group hover descriptions
// =============================================================================

/// The landing page renders a small ⓘ icon next to every group title that
/// has a canonical description, with the description surfaced via the
/// `data-tooltip` attribute (CSS-only hover/focus tooltip). The description
/// also appears on `/api/groups`.
#[tokio::test]
async fn landing_page_renders_group_descriptions() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;

    // Random Access — verbatim v2 description.
    assert!(
        body.contains(r#"data-tooltip="Tests performance of selecting arbitrary row indices from a file on NVMe storage""#),
        "Random Access description must appear as a hover tooltip on the landing page"
    );
    // Compression — verbatim v2 description (the longer wording, not the
    // shorter `getBenchmarkDescription` fallback).
    assert!(
        body.contains(r#"data-tooltip="Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet files (with zstd page compression)""#),
        "Compression description must appear as a hover tooltip on the landing page"
    );
    // Compression Size — verbatim v2 description.
    assert!(
        body.contains(r#"data-tooltip="Compares compressed file sizes and compression ratios across different encoding strategies""#),
        "Compression Size description must appear as a hover tooltip on the landing page"
    );
    // TPC-H NVMe SF=1 — derived description with scale-bytes annotation.
    assert!(
        body.contains(
            r#"data-tooltip="TPC-H benchmark queries on local NVMe storage at SF=1 (~1GB of data)""#
        ),
        "TPC-H description with scale-bytes annotation must appear on the landing page"
    );

    // The icon itself is keyboard-focusable + role-annotated for a11y.
    assert!(
        body.contains(r#"<span class="group-info-icon""#),
        "info icon must render with the group-info-icon class"
    );
    assert!(
        body.contains(r#"tabindex="0""#) && body.contains(r#"role="note""#),
        "info icon must be keyboard-focusable with role=\"note\""
    );
    assert!(
        body.contains(r#"aria-label="Tests performance of selecting arbitrary row indices from a file on NVMe storage""#),
        "info icon must duplicate the description as aria-label"
    );

    Ok(())
}

/// The `/group/{slug}` permalink page renders the same description as the
/// landing page tooltip. Render through the chart-meta header so it's
/// discoverable next to the chart count.
#[tokio::test]
async fn group_page_renders_description() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let slug = pick_group_slug(&server, |s| s == "Random Access").await?;
    let body = client
        .get(server.url(&format!("/group/{slug}")))
        .send()
        .await?
        .text()
        .await?;

    assert!(
        body.contains(r#"data-tooltip="Tests performance of selecting arbitrary row indices from a file on NVMe storage""#),
        "group permalink page must render the same description tooltip as the landing page"
    );
    assert!(
        body.contains(r#"<span class="group-info-icon""#),
        "group permalink page must render the info-icon span"
    );
    Ok(())
}

/// Vector-search groups have no canonical description in v2, so the v3
/// renderer leaves the icon out (rather than making one up).
#[tokio::test]
async fn vector_search_group_has_no_description_icon() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let body = client.get(server.url("/")).send().await?.text().await?;

    // Locate the vector-search section and assert no info-icon inside its
    // disclosure summary.
    let needle = r#"data-group-name="cohere-large-10m / partitioned""#;
    let start = body.find(needle).context("vector-search section present")?;
    // The `<summary>` tag is the disclosure header; we want the slice
    // between this section's start and the end of its `<summary>`.
    let summary_end = body[start..]
        .find("</summary>")
        .map(|p| start + p)
        .context("section contains </summary>")?;
    let summary = &body[start..summary_end];
    assert!(
        !summary.contains("group-info-icon"),
        "vector-search group should not render an info-icon (no canonical description), got: {summary}"
    );
    Ok(())
}

/// `/api/groups` carries the description on every group entry as a `description`
/// field, so external API consumers can render their own UI without having to
/// hard-code v2's description list.
#[tokio::test]
async fn groups_api_carries_description_field() -> Result<()> {
    let server = Server::start().await?;
    seed(&server).await?;

    let client = reqwest::Client::new();
    let groups: Value = client
        .get(server.url("/api/groups"))
        .send()
        .await?
        .json()
        .await?;
    let arr = groups["groups"].as_array().context("groups[] array")?;
    let by_name = |n: &str| {
        arr.iter()
            .find(|g| g["name"].as_str() == Some(n))
            .with_context(|| format!("group {n:?} present"))
    };
    assert_eq!(
        by_name("Random Access")?["description"].as_str(),
        Some("Tests performance of selecting arbitrary row indices from a file on NVMe storage"),
    );
    assert_eq!(
        by_name("TPC-H (NVMe) (SF=1)")?["description"].as_str(),
        Some("TPC-H benchmark queries on local NVMe storage at SF=1 (~1GB of data)"),
    );
    // Vector-search group has no canonical description; the `description`
    // key should be absent (skip_serializing_if).
    let vsg = by_name("cohere-large-10m / partitioned")?;
    assert!(
        vsg.get("description").is_none(),
        "vector-search group should not carry a description field, got: {vsg}"
    );
    Ok(())
}

// =============================================================================
// Task B — partial-coverage commits
// =============================================================================

/// Build an envelope that records a `random_access_time` measurement only
/// for the listed `(format, value_ns)` pairs. The fixture commits' SHAs are
/// deterministic so tests can assert on them.
fn ra_envelope(sha: &str, ts: &str, msg: &str, rows: &[(&str, i64)]) -> Value {
    json!({
        "run_meta": {
            "benchmark_id": "partial-coverage-fixture",
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
        "records": rows.iter().map(|(format, value_ns)| json!({
            "kind": "random_access_time",
            "commit_sha": sha,
            "dataset": "taxi",
            "format": format,
            "value_ns": value_ns,
            "all_runtimes_ns": [value_ns]
        })).collect::<Vec<_>>()
    })
}

/// Regression test for "charts have invisible gaps where commits should be."
///
/// Seed three commits A, B, C in chronological order:
///   * A — series X and Y both have data
///   * B — only series Y has data (X crashed; this is the partial-coverage case)
///   * C — series X and Y both have data
///
/// The chart's `commits[]` must include all three commits (B included),
/// and series X's value at B must be `null`. Before the fix the chart
/// silently dropped B because `SeriesAccumulator::ensure_commit` only
/// registered commits that had at least one row in the fact table.
#[tokio::test]
async fn chart_includes_commits_with_partial_series_coverage() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    let envelopes = [
        ra_envelope(
            "aaaa111111111111111111111111111111111111",
            "2026-04-23T12:00:00Z",
            "A: both series",
            &[("vortex-file-compressed", 500), ("parquet", 1_000)],
        ),
        ra_envelope(
            "bbbb222222222222222222222222222222222222",
            "2026-04-24T12:00:00Z",
            "B: only parquet (vortex crashed)",
            &[("parquet", 1_100)],
        ),
        ra_envelope(
            "cccc333333333333333333333333333333333333",
            "2026-04-25T12:00:00Z",
            "C: both series",
            &[("vortex-file-compressed", 600), ("parquet", 1_200)],
        ),
    ];
    for env in &envelopes {
        let resp = client
            .post(server.url("/api/ingest"))
            .bearer_auth(TOKEN)
            .json(env)
            .send()
            .await?;
        anyhow::ensure!(
            resp.status().is_success(),
            "seed ingest failed: {}",
            resp.status()
        );
    }
    wait_for_materialized_first_chart_commits(&server, 3).await?;

    let slug = pick_chart_slug(&server, |s| s == "Random Access").await?;
    let chart: Value = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?
        .json()
        .await?;

    let commits = chart["commits"].as_array().context("commits[] array")?;
    let shas: Vec<&str> = commits.iter().filter_map(|c| c["sha"].as_str()).collect();
    assert_eq!(
        shas,
        vec![
            "aaaa111111111111111111111111111111111111",
            "bbbb222222222222222222222222222222222222",
            "cccc333333333333333333333333333333333333",
        ],
        "all three commits must appear in commits[], including the partial-coverage commit B"
    );

    // Series X (vortex-file-compressed) has data at A and C, NULL at B.
    let vortex = chart["series"]["vortex-file-compressed"]
        .as_array()
        .context("vortex-file-compressed series array")?;
    assert_eq!(vortex.len(), 3, "series array aligns with commits[]");
    assert_eq!(vortex[0].as_f64(), Some(500.0));
    assert!(
        vortex[1].is_null(),
        "vortex-file-compressed must be null at the partial-coverage commit, got {:?}",
        vortex[1],
    );
    assert_eq!(vortex[2].as_f64(), Some(600.0));

    // Series Y (parquet) has data at all three commits.
    let parquet = chart["series"]["parquet"]
        .as_array()
        .context("parquet series array")?;
    assert_eq!(parquet[0].as_f64(), Some(1_000.0));
    assert_eq!(parquet[1].as_f64(), Some(1_100.0));
    assert_eq!(parquet[2].as_f64(), Some(1_200.0));

    Ok(())
}

/// A commit with NO row in the chart's fact table (every benchmark crashed
/// for that commit) still appears on the chart's x-axis as long as it falls
/// within the chart's window — i.e. ≥ the earliest commit that has data.
///
/// Seed two commits with random-access data and one commit that only has a
/// `compression_size` row. The compression-size-only commit is in the
/// `commits` dim but has nothing in `random_access_times`, so the random-
/// access chart should still place it on the x-axis with NULL for every
/// series.
#[tokio::test]
async fn chart_includes_commits_with_zero_rows_in_fact_table() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    // Commit A: random-access only.
    let env_a = ra_envelope(
        "aaaa111111111111111111111111111111111111",
        "2026-04-23T12:00:00Z",
        "A",
        &[("parquet", 1_000)],
    );
    // Commit B (chronologically between A and C): a compression_size row,
    // nothing in random_access_times.
    let env_b = json!({
        "run_meta": {
            "benchmark_id": "partial-coverage-fixture",
            "schema_version": 1,
            "started_at": "2026-04-24T12:00:00Z"
        },
        "commit": {
            "sha": "bbbb222222222222222222222222222222222222",
            "timestamp": "2026-04-24T12:00:00Z",
            "message": "B: random-access did not run (only compression_size emitted)",
            "author_name": "Test Author",
            "author_email": "author@example.com",
            "committer_name": "Test Committer",
            "committer_email": "committer@example.com",
            "tree_sha": "fedcba9876543210fedcba9876543210fedcba98",
            "url": "https://github.com/vortex-data/vortex/commit/bbbb222222222222222222222222222222222222"
        },
        "records": [
            {
                "kind": "compression_size",
                "commit_sha": "bbbb222222222222222222222222222222222222",
                "dataset": "tpch-lineitem",
                "format": "parquet",
                "value_bytes": 4_000,
            },
        ],
    });
    // Commit C: random-access again.
    let env_c = ra_envelope(
        "cccc333333333333333333333333333333333333",
        "2026-04-25T12:00:00Z",
        "C",
        &[("parquet", 1_200)],
    );

    for env in [&env_a, &env_b, &env_c] {
        let resp = client
            .post(server.url("/api/ingest"))
            .bearer_auth(TOKEN)
            .json(env)
            .send()
            .await?;
        anyhow::ensure!(
            resp.status().is_success(),
            "seed ingest failed: {}",
            resp.status()
        );
    }
    wait_for_materialized_first_chart_commits(&server, 3).await?;

    let slug = pick_chart_slug(&server, |s| s == "Random Access").await?;
    let chart: Value = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?
        .json()
        .await?;

    let shas: Vec<&str> = chart["commits"]
        .as_array()
        .context("commits[] array")?
        .iter()
        .filter_map(|c| c["sha"].as_str())
        .collect();
    assert_eq!(
        shas,
        vec![
            "aaaa111111111111111111111111111111111111",
            "bbbb222222222222222222222222222222222222",
            "cccc333333333333333333333333333333333333",
        ],
        "the commit with zero rows in the fact table must still appear in commits[]"
    );

    // The parquet series has data only at A and C.
    let parquet = chart["series"]["parquet"]
        .as_array()
        .context("parquet series array")?;
    assert_eq!(parquet.len(), 3);
    assert_eq!(parquet[0].as_f64(), Some(1_000.0));
    assert!(
        parquet[1].is_null(),
        "parquet must be null at the zero-rows commit"
    );
    assert_eq!(parquet[2].as_f64(), Some(1_200.0));

    Ok(())
}

/// Commits older than the earliest fact-table row for this chart are NOT
/// included on the x-axis. Without this lower bound a chart's first commit
/// could be from before the benchmark even existed — the spec calls this
/// out explicitly as "Beware: don't accidentally include EVERY commit ever."
#[tokio::test]
async fn chart_excludes_commits_before_first_fact_row() -> Result<()> {
    let server = Server::start().await?;
    let client = reqwest::Client::new();

    // Commit A: a `compression_time` row (random-access does not exist for A).
    let env_a = json!({
        "run_meta": {
            "benchmark_id": "partial-coverage-fixture",
            "schema_version": 1,
            "started_at": "2026-04-22T12:00:00Z"
        },
        "commit": {
            "sha": "aaaa111111111111111111111111111111111111",
            "timestamp": "2026-04-22T12:00:00Z",
            "message": "A: pre-history of the random-access bench",
            "author_name": "Test Author",
            "author_email": "author@example.com",
            "committer_name": "Test Committer",
            "committer_email": "committer@example.com",
            "tree_sha": "fedcba9876543210fedcba9876543210fedcba98",
            "url": "https://github.com/vortex-data/vortex/commit/aaaa111111111111111111111111111111111111"
        },
        "records": [
            {
                "kind": "compression_time",
                "commit_sha": "aaaa111111111111111111111111111111111111",
                "dataset": "tpch-lineitem",
                "format": "parquet",
                "op": "encode",
                "value_ns": 9_000,
                "all_runtimes_ns": [9_000],
            },
        ],
    });
    // Commit B: first random-access row appears.
    let env_b = ra_envelope(
        "bbbb222222222222222222222222222222222222",
        "2026-04-23T12:00:00Z",
        "B: random-access bench begins",
        &[("parquet", 1_000)],
    );

    for env in [&env_a, &env_b] {
        let resp = client
            .post(server.url("/api/ingest"))
            .bearer_auth(TOKEN)
            .json(env)
            .send()
            .await?;
        anyhow::ensure!(
            resp.status().is_success(),
            "seed ingest failed: {}",
            resp.status()
        );
    }
    wait_for_materialized_first_chart_commits(&server, 1).await?;

    let slug = pick_chart_slug(&server, |s| s == "Random Access").await?;
    let chart: Value = client
        .get(server.url(&format!("/api/chart/{slug}")))
        .send()
        .await?
        .json()
        .await?;

    let shas: Vec<&str> = chart["commits"]
        .as_array()
        .context("commits[] array")?
        .iter()
        .filter_map(|c| c["sha"].as_str())
        .collect();
    assert_eq!(
        shas,
        vec!["bbbb222222222222222222222222222222222222"],
        "commit A predates the first random-access row, so it must not be on the x-axis"
    );
    Ok(())
}
