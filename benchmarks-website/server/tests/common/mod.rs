// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared fixtures for the integration tests.
//!
//! Each test file in `server/tests/` is its own crate; this module is
//! `mod common;`'d into each so they share the [`Server`] harness, the
//! seed envelopes, and the response-extraction helpers without duplicating
//! the in-process startup boilerplate.

#![allow(dead_code)]

use std::net::SocketAddr;

use anyhow::Context as _;
use anyhow::Result;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::Duration;
use vortex_bench_server::app::AppState;
use vortex_bench_server::app::router;

/// Bearer token wired into the in-process server. Test ingest calls send
/// this in `Authorization: Bearer …`.
pub(crate) const TOKEN: &str = "test-bearer-token";

/// In-process axum server bound to a random port. Drops cleanly on `Drop`.
pub(crate) struct Server {
    /// Loopback address the server is listening on.
    pub(crate) addr: SocketAddr,
    _tmp: TempDir,
    handle: JoinHandle<()>,
}

impl Server {
    /// Spin up an in-process server backed by a fresh temp DuckDB.
    pub(crate) async fn start() -> Result<Self> {
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

    /// Build an absolute URL for `path` against the in-process server.
    pub(crate) fn url(&self, path: &str) -> String {
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
pub(crate) fn commits() -> &'static [(&'static str, &'static str, &'static str)] {
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
pub(crate) fn envelope_for(sha: &str, ts: &str, msg: &str, value_bias: i64) -> Value {
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

/// POST every fixture commit through `/api/ingest` so the test DB is
/// pre-populated before the test exercises the read API.
pub(crate) async fn seed(server: &Server) -> Result<()> {
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
    wait_for_materialized_first_chart_commits(server, commits().len()).await?;
    Ok(())
}

/// Slim ingest envelope carrying just a `random_access_time` pair so we can
/// drive a long-history fixture cheaply (the full envelope is ~12 records;
/// this is two). Used by the downsample tests.
pub(crate) fn ra_envelope_for(sha: &str, ts: &str, msg: &str, bias: i64) -> Value {
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
pub(crate) async fn seed_long_history(server: &Server, n: usize) -> Result<()> {
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
    wait_for_materialized_first_chart_commits(server, n).await?;
    Ok(())
}

/// Wait until the background read-model rebuild has made the first group's
/// first shard visible with at least `min_commits` commits on its first chart.
pub(crate) async fn wait_for_materialized_first_chart_commits(
    server: &Server,
    min_commits: usize,
) -> Result<()> {
    wait_for_materialized_group_chart_commits(server, "Random Access", min_commits).await
}

/// Wait until a named group is visible in the active read generation and its
/// first shard's first chart has at least `min_commits` commits.
pub(crate) async fn wait_for_materialized_group_chart_commits(
    server: &Server,
    group_name: &str,
    min_commits: usize,
) -> Result<()> {
    let client = reqwest::Client::new();
    let mut last_count = 0usize;
    for _ in 0..100 {
        let landing = client.get(server.url("/")).send().await?.text().await?;
        let Some(generation) = attr_value(&landing, "data-artifact-generation") else {
            tokio::time::sleep(Duration::from_millis(25)).await;
            continue;
        };
        let groups: Value = client
            .get(server.url("/api/groups"))
            .send()
            .await?
            .json()
            .await?;
        let Some(group_slug) = groups["groups"]
            .as_array()
            .and_then(|groups| {
                groups
                    .iter()
                    .find(|group| group["name"].as_str() == Some(group_name))
            })
            .and_then(|group| group["slug"].as_str())
            .map(str::to_string)
        else {
            tokio::time::sleep(Duration::from_millis(25)).await;
            continue;
        };
        let resp = client
            .get(server.url(&format!(
                "/api/artifacts/{generation}/groups/{group_slug}/shards/0"
            )))
            .send()
            .await?;
        if resp.status().is_success() {
            let body: Value = resp.json().await?;
            let chart = &body["charts"][0];
            last_count = chart["history"]["total_commits"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .or_else(|| chart["commits"].as_array().map(Vec::len))
                .unwrap_or_default();
            if last_count >= min_commits {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    anyhow::bail!(
        "read model did not reach {min_commits} commits on {group_name:?}; last count {last_count}"
    )
}

/// Pull the inline `<script id="chart-data-N">…</script>` JSON out of an
/// HTML body. Returns `None` if the script tag isn't present.
pub(crate) fn extract_chart_data(body: &str, idx: usize) -> Option<Value> {
    let needle = format!(r#"<script id="chart-data-{idx}" type="application/json">"#);
    let start = body.find(&needle)? + needle.len();
    let end = body[start..].find("</script>")? + start;
    // Reverse the `</` neutralisation done by `escape_json_for_script`.
    let json = body[start..end].replace(r"<\/", "</");
    serde_json::from_str(&json).ok()
}

/// Pull a simple double-quoted HTML attribute value from `body`.
pub(crate) fn attr_value(body: &str, attr: &str) -> Option<String> {
    let needle = format!(r#"{attr}=""#);
    let start = body.find(&needle)? + needle.len();
    let end = body[start..].find('"')? + start;
    Some(body[start..end].to_string())
}

/// Configure `insta` to look for snapshots in `tests/snapshots/` keyed by
/// just the explicit name (no module prefix). Every test in this crate uses
/// these settings so the snapshot file layout is path-independent.
pub(crate) fn insta_settings() -> insta::Settings {
    let mut s = insta::Settings::clone_current();
    s.set_snapshot_path("snapshots");
    s.set_prepend_module_to_snapshot(false);
    s
}

/// Lift a single chart slug from `/api/groups`, picking from a group whose
/// name matches `predicate`. Used by tests that need a real slug to drive
/// `/chart/{slug}` and `/api/chart/{slug}` round-trips.
pub(crate) async fn pick_chart_slug(
    server: &Server,
    predicate: impl Fn(&str) -> bool,
) -> Result<String> {
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

/// Lift a single group slug from `/api/groups`, picking the first group
/// whose name matches `predicate`.
pub(crate) async fn pick_group_slug(
    server: &Server,
    predicate: impl Fn(&str) -> bool,
) -> Result<String> {
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

/// Look up a group entry by its `name` field inside an `/api/groups`
/// response.
pub(crate) fn group_by_name<'a>(groups: &'a Value, name: &str) -> Result<&'a Value> {
    groups["groups"]
        .as_array()
        .context("groups is array")?
        .iter()
        .find(|g| g["name"].as_str() == Some(name))
        .with_context(|| format!("group {name:?} exists"))
}

/// Fuzzy `f64` equality for test assertions. The summary rollups round-trip
/// through SQL so exact equality isn't safe even on integer-valued inputs.
pub(crate) fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(
        delta < 0.000_001,
        "expected {actual} to be close to {expected}"
    );
}

/// Pull just the `<div class="filter-dropdown" …>…</div>` substring of the
/// filter dropdown — its trigger button and the chip panel. Keeps the
/// snapshot focused on the chip markup and stable against changes elsewhere
/// on the page.
pub(crate) fn filter_bar_section(body: &str) -> String {
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
pub(crate) fn filter_section(body: &str, dim: &str) -> String {
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
pub(crate) fn extract_chip(section: &str, value: &str) -> String {
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
