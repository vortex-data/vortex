// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-chart payload assembly + the shared `SeriesAccumulator` glue.
//!
//! `chart_payload` dispatches on [`ChartKey`] to one of five
//! `collect_*_chart` functions, each of which runs one SQL query against
//! its fact table, threads the rows through a `SeriesAccumulator`, and
//! returns a [`ChartResponse`].

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use duckdb::ToSql;
use duckdb::params_from_iter;

use super::dto::ChartHistory;
use super::dto::ChartResponse;
use super::dto::CommitPoint;
use super::dto::GroupChartsResponse;
use super::dto::NamedChartResponse;
use super::dto::SeriesTag;
use super::dto::UnitKind;
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
///
/// TODO(#7812): this re-runs the entire [`collect_groups`] discovery pass
/// per call before fetching each chart, so the landing page is
/// O(groups * charts_per_group) DB queries plus the discovery scan. Fine
/// for the current dataset; tracked for the refactor that collapses it
/// into a single query.
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
            chart: Arc::new(chart),
        });
    }
    if charts.is_empty() {
        return Ok(None);
    }
    Ok(Some(GroupChartsResponse {
        name: group.name,
        summary: group.summary,
        description: group.description,
        charts,
    }))
}

/// Time series rows are gathered keyed by `(commit_sha, series_key)` and then
/// reshaped into the `commits[] / series{}` response shape.
///
/// **The accumulator is seeded with the canonical commits-in-window list
/// before any fact rows are recorded.** That list is the chart's x-axis: it
/// includes every commit in the requested [`CommitWindow`] whose timestamp
/// is at or after the earliest commit that has a row in the fact table for
/// this chart. Commits with zero rows in the fact table still appear in
/// `commits[]`; their per-series slot stays `None` and renders as a visible
/// gap in the line. Without seeding, commits absent from the fact table
/// would be silently dropped from the chart's x-axis, making partial-coverage
/// runs (a benchmark crashed; a series only runs nightly) look like
/// continuous lines when they should break.
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

    /// Seed the chart's commit list, oldest-first by timestamp. Must be
    /// called before [`Self::record`] / [`Self::tag`] so series allocations
    /// are sized correctly and missing-value slots stay `None`.
    fn seed_commits(&mut self, commits: Vec<CommitPoint>) {
        self.commit_index.clear();
        for (i, c) in commits.iter().enumerate() {
            self.commit_index.insert(c.sha.clone(), i);
        }
        self.commits = commits;
    }

    /// Index of `sha` in the seeded commits list, or `None` if the sha
    /// was not part of the window. Returning `None` rather than panicking
    /// keeps `collect_*_chart` resilient to an unseeded sha showing up in
    /// the fact table (e.g. a transient race in concurrent ingest); we
    /// just drop the row.
    fn commit_idx(&self, sha: &str) -> Option<usize> {
        self.commit_index.get(sha).copied()
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

    fn finish(
        self,
        display_name: String,
        unit_kind: UnitKind,
        history: ChartHistory,
    ) -> ChartResponse {
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
            unit_kind,
            history,
            commits: self.commits,
            series: series_map,
            series_meta: self.tags,
        }
    }
}

struct SeededCommits {
    commits: Vec<CommitPoint>,
    history: ChartHistory,
}

/// Resolve a chart's x-axis: every commit in the requested commit-window
/// whose timestamp is at or after the earliest commit that has a row in the
/// fact table for this chart. Returns the list oldest-first; an empty list
/// means the fact table has no rows at all for this chart, and the caller
/// should return `None` (404).
///
/// `earliest_subquery` is spliced into the outer query as
/// `c.timestamp >= ({earliest_subquery})`, so it must SELECT a single
/// `MIN(timestamp)` row scoped to this chart's fact-table predicates. Its
/// bound parameters appear first in `subquery_binds`; the window's `LIMIT`
/// placeholder is appended after.
///
/// The bounds matter: without the timestamp lower bound a chart's x-axis
/// would include every commit ever, including pre-history before the
/// benchmark even existed. Without the [`CommitWindow`] cap a chart with a
/// long history would always render the entire timeline regardless of the
/// caller's `?n=` request.
fn seeded_commits_in_window(
    conn: &Connection,
    earliest_subquery: &str,
    subquery_binds: Vec<Box<dyn ToSql>>,
    window: &CommitWindow,
) -> Result<SeededCommits> {
    let window_filter = match window {
        CommitWindow::All => "",
        CommitWindow::Last(_) => "WHERE rn > total_commits - ?",
    };
    let sql = format!(
        r#"
        WITH eligible AS (
            SELECT c.commit_sha,
                   c.timestamp,
                   COALESCE(c.message, '') AS message,
                   c.url,
                   row_number() OVER (ORDER BY c.timestamp ASC, c.commit_sha ASC) AS rn,
                   count(*) OVER () AS total_commits
              FROM commits c
             WHERE c.timestamp >= ({earliest_subquery})
        )
        SELECT commit_sha,
               CAST(timestamp AS VARCHAR),
               message,
               url,
               total_commits
          FROM eligible
         {window_filter}
         ORDER BY timestamp ASC, commit_sha ASC
        "#,
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut binds = subquery_binds;
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            CommitPoint {
                sha: row.get(0)?,
                timestamp: row.get(1)?,
                message: row.get(2)?,
                url: row.get(3)?,
            },
            row.get::<_, i64>(4)?,
        ))
    })?;
    let rows: Vec<(CommitPoint, i64)> = rows.collect::<Result<_, _>>()?;
    let total_commits = rows
        .first()
        .map(|(_, total)| usize::try_from(*total))
        .transpose()?
        .unwrap_or_default();
    let commits: Vec<CommitPoint> = rows.into_iter().map(|(commit, _)| commit).collect();
    let loaded_commits = commits.len();
    let start_index = total_commits.saturating_sub(loaded_commits);
    Ok(SeededCommits {
        commits,
        history: ChartHistory {
            total_commits,
            start_index,
            loaded_commits,
            complete: loaded_commits == total_commits,
        },
    })
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
    // x-axis pre-pass. `IS NOT DISTINCT FROM` matches NULL == NULL so charts
    // with a NULL `dataset_variant` or `scale_factor` still pin the right
    // earliest-commit timestamp.
    let seeded = seeded_commits_in_window(
        conn,
        "SELECT MIN(c2.timestamp) \
           FROM query_measurements q2 \
           JOIN commits c2 ON c2.commit_sha = q2.commit_sha \
          WHERE q2.dataset = ? \
            AND q2.dataset_variant IS NOT DISTINCT FROM ? \
            AND q2.scale_factor    IS NOT DISTINCT FROM ? \
            AND q2.storage = ? \
            AND q2.query_idx = ?",
        vec![
            Box::new(dataset.to_string()),
            Box::new(dataset_variant.clone()),
            Box::new(scale_factor.clone()),
            Box::new(storage.to_string()),
            Box::new(query_idx),
        ],
        window,
    )?;
    if seeded.commits.is_empty() {
        return Ok(None);
    }
    let history = seeded.history;
    let mut acc = SeriesAccumulator::new();
    acc.seed_commits(seeded.commits);

    let sql = format!(
        r#"
        SELECT q.commit_sha,
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
            row.get::<_, i64>(3)?,
        ))
    })?;
    for row in rows {
        let (sha, engine, format, value_ns) = row?;
        let Some(idx) = acc.commit_idx(&sha) else {
            continue;
        };
        let series_key = format!("{engine}:{format}");
        acc.record(&series_key, idx, value_ns as f64);
        acc.tag(&series_key, Some(&engine), Some(&format));
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
    Ok(Some(acc.finish(name, UnitKind::TimeNs, history)))
}

fn collect_compression_time_chart(
    conn: &Connection,
    dataset: &str,
    dataset_variant: &Option<String>,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let seeded = seeded_commits_in_window(
        conn,
        "SELECT MIN(c2.timestamp) \
           FROM compression_times t2 \
           JOIN commits c2 ON c2.commit_sha = t2.commit_sha \
          WHERE t2.dataset = ? \
            AND t2.dataset_variant IS NOT DISTINCT FROM ?",
        vec![
            Box::new(dataset.to_string()),
            Box::new(dataset_variant.clone()),
        ],
        window,
    )?;
    if seeded.commits.is_empty() {
        return Ok(None);
    }
    let history = seeded.history;
    let mut acc = SeriesAccumulator::new();
    acc.seed_commits(seeded.commits);

    let sql = format!(
        r#"
        SELECT t.commit_sha,
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
            row.get::<_, i64>(3)?,
        ))
    })?;
    for row in rows {
        let (sha, format, op, value_ns) = row?;
        let Some(idx) = acc.commit_idx(&sha) else {
            continue;
        };
        let series_key = format!("{format}:{op}");
        acc.record(&series_key, idx, value_ns as f64);
        acc.tag(&series_key, None, Some(&format));
    }
    let mut name = dataset.to_string();
    if let Some(v) = dataset_variant {
        name.push('/');
        name.push_str(v);
    }
    Ok(Some(acc.finish(name, UnitKind::TimeNs, history)))
}

fn collect_compression_size_chart(
    conn: &Connection,
    dataset: &str,
    dataset_variant: &Option<String>,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let seeded = seeded_commits_in_window(
        conn,
        "SELECT MIN(c2.timestamp) \
           FROM compression_sizes s2 \
           JOIN commits c2 ON c2.commit_sha = s2.commit_sha \
          WHERE s2.dataset = ? \
            AND s2.dataset_variant IS NOT DISTINCT FROM ?",
        vec![
            Box::new(dataset.to_string()),
            Box::new(dataset_variant.clone()),
        ],
        window,
    )?;
    if seeded.commits.is_empty() {
        return Ok(None);
    }
    let history = seeded.history;
    let mut acc = SeriesAccumulator::new();
    acc.seed_commits(seeded.commits);

    let sql = format!(
        r#"
        SELECT s.commit_sha,
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
    let mut binds: Vec<Box<dyn ToSql>> = vec![
        Box::new(dataset.to_string()),
        Box::new(dataset_variant.clone()),
    ];
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;
    for row in rows {
        let (sha, format, value_bytes) = row?;
        let Some(idx) = acc.commit_idx(&sha) else {
            continue;
        };
        acc.record(&format, idx, value_bytes as f64);
        acc.tag(&format, None, Some(&format));
    }
    let mut name = dataset.to_string();
    if let Some(v) = dataset_variant {
        name.push('/');
        name.push_str(v);
    }
    Ok(Some(acc.finish(name, UnitKind::Bytes, history)))
}

fn collect_random_access_chart(
    conn: &Connection,
    dataset: &str,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let seeded = seeded_commits_in_window(
        conn,
        "SELECT MIN(c2.timestamp) \
           FROM random_access_times r2 \
           JOIN commits c2 ON c2.commit_sha = r2.commit_sha \
          WHERE r2.dataset = ?",
        vec![Box::new(dataset.to_string())],
        window,
    )?;
    if seeded.commits.is_empty() {
        return Ok(None);
    }
    let history = seeded.history;
    let mut acc = SeriesAccumulator::new();
    acc.seed_commits(seeded.commits);

    let sql = format!(
        r#"
        SELECT r.commit_sha,
               r.format, r.value_ns
          FROM random_access_times r
          JOIN commits c USING (commit_sha)
         WHERE r.dataset = ?{filter}
         ORDER BY c.timestamp, r.format
        "#,
        filter = window.sql_filter(),
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut binds: Vec<Box<dyn ToSql>> = vec![Box::new(dataset.to_string())];
    push_window_limit(&mut binds, window);
    let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;
    for row in rows {
        let (sha, format, value_ns) = row?;
        let Some(idx) = acc.commit_idx(&sha) else {
            continue;
        };
        acc.record(&format, idx, value_ns as f64);
        acc.tag(&format, None, Some(&format));
    }
    Ok(Some(acc.finish(
        dataset.to_string(),
        UnitKind::TimeNs,
        history,
    )))
}

fn collect_vector_search_chart(
    conn: &Connection,
    dataset: &str,
    layout: &str,
    threshold: f64,
    window: &CommitWindow,
) -> Result<Option<ChartResponse>> {
    let seeded = seeded_commits_in_window(
        conn,
        "SELECT MIN(c2.timestamp) \
           FROM vector_search_runs v2 \
           JOIN commits c2 ON c2.commit_sha = v2.commit_sha \
          WHERE v2.dataset = ? \
            AND v2.layout = ? \
            AND v2.threshold = ?",
        vec![
            Box::new(dataset.to_string()),
            Box::new(layout.to_string()),
            Box::new(threshold),
        ],
        window,
    )?;
    if seeded.commits.is_empty() {
        return Ok(None);
    }
    let history = seeded.history;
    let mut acc = SeriesAccumulator::new();
    acc.seed_commits(seeded.commits);

    let sql = format!(
        r#"
        SELECT v.commit_sha,
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
            row.get::<_, i64>(2)?,
        ))
    })?;
    for row in rows {
        let (sha, flavor, value_ns) = row?;
        let Some(idx) = acc.commit_idx(&sha) else {
            continue;
        };
        acc.record(&flavor, idx, value_ns as f64);
    }
    Ok(Some(acc.finish(
        format!("{dataset} / {layout} (threshold={threshold})"),
        UnitKind::TimeNs,
        history,
    )))
}
