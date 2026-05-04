// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-chart payload assembly + the shared `SeriesAccumulator` glue.
//!
//! `chart_payload` dispatches on [`ChartKey`] to one of five
//! `collect_*_chart` functions, each of which runs one SQL query against
//! its fact table, threads the rows through a `SeriesAccumulator`, and
//! returns a [`ChartResponse`].

use std::collections::BTreeMap;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use duckdb::ToSql;
use duckdb::params_from_iter;

use super::dto::ChartResponse;
use super::dto::CommitPoint;
use super::dto::GroupChartsResponse;
use super::dto::NamedChartResponse;
use super::dto::SeriesTag;
use super::groups::collect_groups;
use super::window::CommitWindow;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// Build the JSON payload for one chart by key. This is the shared
/// implementation behind `GET /api/chart/{slug}`, the inline `<script>` JSON
/// rendered into the HTML pages, and `collect_group_charts`.
///
/// `window` caps the number of recent commits returned. `y` / `mode` are not
/// inputs here — they're rendering hints applied client-side, so the SQL is
/// unaffected and the cached payload is identical across hint values.
pub(crate) fn chart_payload(
    conn: &Connection,
    key: &ChartKey,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    match key {
        ChartKey::QueryMeasurement {
            dataset,
            dataset_variant,
            scale_factor,
            storage,
            query_idx,
        } => collect_query_chart(
            conn,
            dataset,
            dataset_variant,
            scale_factor,
            storage,
            *query_idx,
            window,
        ),
        ChartKey::CompressionTime {
            dataset,
            dataset_variant,
        } => collect_compression_time_chart(conn, dataset, dataset_variant, window),
        ChartKey::CompressionSize {
            dataset,
            dataset_variant,
        } => collect_compression_size_chart(conn, dataset, dataset_variant, window),
        ChartKey::RandomAccess { dataset } => collect_random_access_chart(conn, dataset, window),
        ChartKey::VectorSearch {
            dataset,
            layout,
            threshold,
        } => collect_vector_search_chart(conn, dataset, layout, *threshold, window),
    }
}

/// Collect every chart inside one group. Returns `None` if the group has no
/// data at all (callers should render a 404).
// TODO: this currently re-runs the entire `collect_groups` discovery pass
// (api.rs) per call before fetching each chart, which makes the landing page
// O(groups * charts_per_group) DB queries plus the discovery scan. Fine for
// the current dataset; revisit when chart counts grow.
pub(crate) fn collect_group_charts(
    conn: &Connection,
    key: &GroupKey,
    window: &CommitWindow,
) -> Result<Option<GroupChartsResponse>> {
    let groups = collect_groups(conn)?;
    let group = groups.into_iter().find(|g| g.slug == key.to_slug());
    let Some(group) = group else {
        return Ok(None);
    };
    let mut charts = Vec::with_capacity(group.charts.len());
    for link in group.charts {
        let chart_key = ChartKey::from_slug(&link.slug)
            .with_context(|| format!("invalid chart slug in group: {}", link.slug))?;
        let Some(chart) = chart_payload(conn, &chart_key, window)? else {
            continue;
        };
        charts.push(NamedChartResponse {
            name: link.name,
            slug: link.slug,
            chart,
        });
    }
    if charts.is_empty() {
        return Ok(None);
    }
    Ok(Some(GroupChartsResponse {
        name: group.name,
        summary: group.summary,
        charts,
    }))
}

/// Time series rows are gathered keyed by `(commit_sha, series_key)` and then
/// reshaped into the `commits[] / series{}` response shape.
struct SeriesAccumulator {
    commits: Vec<CommitPoint>,
    commit_index: BTreeMap<String, usize>,
    series: BTreeMap<String, Vec<Option<f64>>>,
    tags: BTreeMap<String, SeriesTag>,
}

impl SeriesAccumulator {
    fn new() -> Self {
        Self {
            commits: Vec::new(),
            commit_index: BTreeMap::new(),
            series: BTreeMap::new(),
            tags: BTreeMap::new(),
        }
    }

    fn ensure_commit(&mut self, sha: &str, timestamp: &str, message: &str, url: &str) -> usize {
        if let Some(&idx) = self.commit_index.get(sha) {
            return idx;
        }
        let idx = self.commits.len();
        self.commits.push(CommitPoint {
            sha: sha.to_string(),
            timestamp: timestamp.to_string(),
            message: message.to_string(),
            url: url.to_string(),
        });
        self.commit_index.insert(sha.to_string(), idx);
        idx
    }

    fn record(&mut self, series_key: &str, commit_idx: usize, value: f64) {
        let total_commits = self.commits.len();
        let entry = self
            .series
            .entry(series_key.to_string())
            .or_insert_with(|| vec![None; total_commits]);
        if entry.len() < total_commits {
            entry.resize(total_commits, None);
        }
        entry[commit_idx] = Some(value);
    }

    /// Record an engine/format classification for a series. Repeat calls with
    /// the same `series_key` are idempotent — every row of a given series
    /// shares the same engine/format by construction of the SQL.
    fn tag(&mut self, series_key: &str, engine: Option<&str>, format: Option<&str>) {
        if engine.is_none() && format.is_none() {
            return;
        }
        let entry = self.tags.entry(series_key.to_string()).or_default();
        if let Some(e) = engine {
            entry.engine = Some(e.to_string());
        }
        if let Some(f) = format {
            entry.format = Some(f.to_string());
        }
    }

    fn finish(self, display_name: String, unit: &'static str) -> ChartResponse {
        let total = self.commits.len();
        let mut series_map = serde_json::Map::new();
        for (k, mut v) in self.series {
            if v.len() < total {
                v.resize(total, None);
            }
            series_map.insert(k, serde_json::to_value(v).expect("Vec<Option<f64>>"));
        }
        ChartResponse {
            display_name,
            unit,
            commits: self.commits,
            series: series_map,
            series_meta: self.tags,
        }
    }
}

/// Append the commit-window `LIMIT` bind value to a parameter list, when the
/// window is bounded. Pairs with [`CommitWindow::sql_filter`] which emits
/// the matching `?` placeholder.
fn push_window_limit(binds: &mut Vec<Box<dyn ToSql>>, window: &CommitWindow) {
    if let Some(limit) = window.limit_param() {
        binds.push(Box::new(limit));
    }
}

fn collect_query_chart(
    conn: &Connection,
    dataset: &str,
    dataset_variant: &Option<String>,
    scale_factor: &Option<String>,
    storage: &str,
    query_idx: i32,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let sql = format!(
        r#"
        SELECT q.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               COALESCE(c.message, '') AS message, c.url,
               q.engine, q.format, q.value_ns
          FROM query_measurements q
          JOIN commits c USING (commit_sha)
         WHERE q.dataset = ?
           AND q.dataset_variant IS NOT DISTINCT FROM ?
           AND q.scale_factor    IS NOT DISTINCT FROM ?
           AND q.storage = ?
           AND q.query_idx = ?{filter}
         ORDER BY c.timestamp, q.engine, q.format
        "#,
        filter = window.sql_filter(),
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut acc = SeriesAccumulator::new();
    let mut binds: Vec<Box<dyn ToSql>> = vec![
        Box::new(dataset.to_string()),
        Box::new(dataset_variant.clone()),
        Box::new(scale_factor.clone()),
        Box::new(storage.to_string()),
        Box::new(query_idx),
    ];
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
            row.get::<_, i64>(6)?,
        ))
    })?;
    let mut any = false;
    for row in rows {
        any = true;
        let (sha, ts, msg, url, engine, format, value_ns) = row?;
        let idx = acc.ensure_commit(&sha, &ts, &msg, &url);
        let series_key = format!("{engine}:{format}");
        acc.record(&series_key, idx, value_ns as f64);
        acc.tag(&series_key, Some(&engine), Some(&format));
    }
    if !any {
        return Ok(None);
    }
    let mut name = dataset.to_string();
    if let Some(v) = dataset_variant {
        name.push('/');
        name.push_str(v);
    }
    if let Some(sf) = scale_factor {
        name.push_str(" sf=");
        name.push_str(sf);
    }
    name.push_str(&format!(" Q{query_idx} [{storage}]"));
    Ok(Some(acc.finish(name, "ns")))
}

fn collect_compression_time_chart(
    conn: &Connection,
    dataset: &str,
    dataset_variant: &Option<String>,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let sql = format!(
        r#"
        SELECT t.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               COALESCE(c.message, '') AS message, c.url,
               t.format, t.op, t.value_ns
          FROM compression_times t
          JOIN commits c USING (commit_sha)
         WHERE t.dataset = ?
           AND t.dataset_variant IS NOT DISTINCT FROM ?{filter}
         ORDER BY c.timestamp, t.format, t.op
        "#,
        filter = window.sql_filter(),
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut acc = SeriesAccumulator::new();
    let mut binds: Vec<Box<dyn ToSql>> = vec![
        Box::new(dataset.to_string()),
        Box::new(dataset_variant.clone()),
    ];
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
            row.get::<_, i64>(6)?,
        ))
    })?;
    let mut any = false;
    for row in rows {
        any = true;
        let (sha, ts, msg, url, format, op, value_ns) = row?;
        let idx = acc.ensure_commit(&sha, &ts, &msg, &url);
        let series_key = format!("{format}:{op}");
        acc.record(&series_key, idx, value_ns as f64);
        acc.tag(&series_key, None, Some(&format));
    }
    if !any {
        return Ok(None);
    }
    let mut name = dataset.to_string();
    if let Some(v) = dataset_variant {
        name.push('/');
        name.push_str(v);
    }
    Ok(Some(acc.finish(name, "ns")))
}

fn collect_compression_size_chart(
    conn: &Connection,
    dataset: &str,
    dataset_variant: &Option<String>,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let sql = format!(
        r#"
        SELECT s.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               COALESCE(c.message, '') AS message, c.url,
               s.format, s.value_bytes
          FROM compression_sizes s
          JOIN commits c USING (commit_sha)
         WHERE s.dataset = ?
           AND s.dataset_variant IS NOT DISTINCT FROM ?{filter}
         ORDER BY c.timestamp, s.format
        "#,
        filter = window.sql_filter(),
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut acc = SeriesAccumulator::new();
    let mut binds: Vec<Box<dyn ToSql>> = vec![
        Box::new(dataset.to_string()),
        Box::new(dataset_variant.clone()),
    ];
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, i64>(5)?,
        ))
    })?;
    let mut any = false;
    for row in rows {
        any = true;
        let (sha, ts, msg, url, format, value_bytes) = row?;
        let idx = acc.ensure_commit(&sha, &ts, &msg, &url);
        acc.record(&format, idx, value_bytes as f64);
        acc.tag(&format, None, Some(&format));
    }
    if !any {
        return Ok(None);
    }
    let mut name = dataset.to_string();
    if let Some(v) = dataset_variant {
        name.push('/');
        name.push_str(v);
    }
    Ok(Some(acc.finish(name, "bytes")))
}

fn collect_random_access_chart(
    conn: &Connection,
    dataset: &str,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let sql = format!(
        r#"
        SELECT r.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               COALESCE(c.message, '') AS message, c.url,
               r.format, r.value_ns
          FROM random_access_times r
          JOIN commits c USING (commit_sha)
         WHERE r.dataset = ?{filter}
         ORDER BY c.timestamp, r.format
        "#,
        filter = window.sql_filter(),
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut acc = SeriesAccumulator::new();
    let mut binds: Vec<Box<dyn ToSql>> = vec![Box::new(dataset.to_string())];
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, i64>(5)?,
        ))
    })?;
    let mut any = false;
    for row in rows {
        any = true;
        let (sha, ts, msg, url, format, value_ns) = row?;
        let idx = acc.ensure_commit(&sha, &ts, &msg, &url);
        acc.record(&format, idx, value_ns as f64);
        acc.tag(&format, None, Some(&format));
    }
    if !any {
        return Ok(None);
    }
    Ok(Some(acc.finish(dataset.to_string(), "ns")))
}

fn collect_vector_search_chart(
    conn: &Connection,
    dataset: &str,
    layout: &str,
    threshold: f64,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let sql = format!(
        r#"
        SELECT v.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               COALESCE(c.message, '') AS message, c.url,
               v.flavor, v.value_ns
          FROM vector_search_runs v
          JOIN commits c USING (commit_sha)
         WHERE v.dataset = ?
           AND v.layout = ?
           AND v.threshold = ?{filter}
         ORDER BY c.timestamp, v.flavor
        "#,
        filter = window.sql_filter(),
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut acc = SeriesAccumulator::new();
    let mut binds: Vec<Box<dyn ToSql>> = vec![
        Box::new(dataset.to_string()),
        Box::new(layout.to_string()),
        Box::new(threshold),
    ];
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, i64>(5)?,
        ))
    })?;
    let mut any = false;
    for row in rows {
        any = true;
        let (sha, ts, msg, url, flavor, value_ns) = row?;
        let idx = acc.ensure_commit(&sha, &ts, &msg, &url);
        acc.record(&flavor, idx, value_ns as f64);
    }
    if !any {
        return Ok(None);
    }
    Ok(Some(acc.finish(
        format!("{dataset} / {layout} (threshold={threshold})"),
        "ns",
    )))
}
