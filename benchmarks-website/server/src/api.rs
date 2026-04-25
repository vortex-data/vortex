// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-side API: `/api/groups`, `/api/chart/:slug`, `/health`.
//!
//! Group / chart / series fit follows
//! `benchmarks-website/planning/01-schema.md`. Slugs round-trip through
//! [`crate::slug::ChartKey`].

use anyhow::{Context as _, Result};
use axum::Json;
use axum::extract::{Path, State};
use axum::response::IntoResponse;
use duckdb::{Connection, params};
use serde::Serialize;
use serde_json::Value as JsonValue;

use crate::app::AppState;
use crate::db;
use crate::error::ApiError;
use crate::slug::ChartKey;

#[derive(Debug, Serialize)]
pub struct GroupsResponse {
    pub groups: Vec<Group>,
}

#[derive(Debug, Serialize)]
pub struct Group {
    pub name: String,
    pub charts: Vec<ChartLink>,
}

#[derive(Debug, Serialize)]
pub struct ChartLink {
    pub name: String,
    pub slug: String,
}

#[derive(Debug, Serialize)]
pub struct ChartResponse {
    pub display_name: String,
    pub unit: &'static str,
    pub commits: Vec<CommitPoint>,
    pub series: serde_json::Map<String, JsonValue>,
}

#[derive(Debug, Serialize)]
pub struct CommitPoint {
    pub sha: String,
    pub timestamp: String,
    pub message: String,
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub db_path: String,
    pub schema_version: i32,
    pub latest_commit_timestamp: Option<String>,
    pub row_counts: RowCounts,
}

#[derive(Debug, Serialize)]
pub struct RowCounts {
    pub commits: i64,
    pub query_measurements: i64,
    pub compression_times: i64,
    pub compression_sizes: i64,
    pub random_access_times: i64,
    pub vector_search_runs: i64,
}

/// Handler for `GET /api/groups`.
pub async fn groups(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let groups = db::run_blocking(&state.db, |conn| collect_groups(conn)).await?;
    Ok(Json(GroupsResponse { groups }))
}

/// Handler for `GET /api/chart/:slug`.
pub async fn chart(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let key = ChartKey::from_slug(&slug)
        .map_err(|e| ApiError::BadRequest(format!("invalid slug: {e}")))?;
    let response = db::run_blocking(&state.db, move |conn| collect_chart(conn, &key)).await?;
    let response =
        response.ok_or_else(|| ApiError::NotFound(format!("no data for slug {slug:?}")))?;
    Ok(Json(response))
}

/// Handler for `GET /health`.
pub async fn health(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let path = state.db_path.display().to_string();
    let response = db::run_blocking(&state.db, move |conn| collect_health(conn, path)).await?;
    Ok(Json(response))
}

fn collect_health(conn: &Connection, db_path: String) -> Result<HealthResponse> {
    let count = |table: &str| -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        let n: i64 = conn.query_row(&sql, [], |r| r.get(0))?;
        Ok(n)
    };
    let row_counts = RowCounts {
        commits: count("commits")?,
        query_measurements: count("query_measurements")?,
        compression_times: count("compression_times")?,
        compression_sizes: count("compression_sizes")?,
        random_access_times: count("random_access_times")?,
        vector_search_runs: count("vector_search_runs")?,
    };
    let latest_commit_timestamp: Option<String> = conn
        .query_row(
            "SELECT CAST(timestamp AS VARCHAR) FROM commits ORDER BY timestamp DESC LIMIT 1",
            [],
            |r| r.get::<_, String>(0),
        )
        .ok();
    Ok(HealthResponse {
        status: "ok",
        db_path,
        schema_version: crate::schema::SCHEMA_VERSION,
        latest_commit_timestamp,
        row_counts,
    })
}

fn collect_groups(conn: &Connection) -> Result<Vec<Group>> {
    let mut groups: Vec<Group> = Vec::new();

    let qm_groups = collect_query_groups(conn).context("collect_query_groups")?;
    groups.extend(qm_groups);

    if let Some(g) = collect_compression_time_group(conn)? {
        groups.push(g);
    }
    if let Some(g) = collect_compression_size_group(conn)? {
        groups.push(g);
    }
    if let Some(g) = collect_random_access_group(conn)? {
        groups.push(g);
    }
    let vsr_groups = collect_vector_search_groups(conn)?;
    groups.extend(vsr_groups);

    Ok(groups)
}

fn collect_query_groups(conn: &Connection) -> Result<Vec<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant, scale_factor, storage, query_idx
          FROM query_measurements
         GROUP BY dataset, dataset_variant, scale_factor, storage, query_idx
         ORDER BY dataset, dataset_variant NULLS FIRST,
                  scale_factor NULLS FIRST, storage, query_idx
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i32>(4)?,
        ))
    })?;

    let mut groups: Vec<Group> = Vec::new();
    let mut current: Option<(String, Option<String>, Option<String>, String)> = None;
    for row in rows {
        let (dataset, dataset_variant, scale_factor, storage, query_idx) = row?;
        let key = (
            dataset.clone(),
            dataset_variant.clone(),
            scale_factor.clone(),
            storage.clone(),
        );
        let need_new_group = current.as_ref() != Some(&key);
        if need_new_group {
            groups.push(Group {
                name: group_name_query(&dataset, &dataset_variant, &scale_factor, &storage),
                charts: Vec::new(),
            });
            current = Some(key);
        }
        let slug = ChartKey::QueryMeasurement {
            dataset,
            dataset_variant,
            scale_factor,
            storage,
            query_idx,
        }
        .to_slug();
        groups
            .last_mut()
            .expect("just pushed")
            .charts
            .push(ChartLink {
                name: format!("Q{query_idx}"),
                slug,
            });
    }
    Ok(groups)
}

fn group_name_query(
    dataset: &str,
    dataset_variant: &Option<String>,
    scale_factor: &Option<String>,
    storage: &str,
) -> String {
    let mut name = dataset.to_string();
    if let Some(v) = dataset_variant {
        name.push('/');
        name.push_str(v);
    }
    if let Some(sf) = scale_factor {
        name.push_str(" sf=");
        name.push_str(sf);
    }
    name.push_str(" [");
    name.push_str(storage);
    name.push(']');
    name
}

fn collect_compression_time_group(conn: &Connection) -> Result<Option<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant
          FROM compression_times
         GROUP BY dataset, dataset_variant
         ORDER BY dataset, dataset_variant NULLS FIRST
        "#,
    )?;
    let charts: Vec<ChartLink> = stmt
        .query_map([], |row| {
            let dataset: String = row.get(0)?;
            let dataset_variant: Option<String> = row.get(1)?;
            Ok((dataset, dataset_variant))
        })?
        .map(|r| {
            r.map(|(dataset, dataset_variant)| {
                let key = ChartKey::CompressionTime {
                    dataset: dataset.clone(),
                    dataset_variant: dataset_variant.clone(),
                };
                let mut name = dataset;
                if let Some(v) = &dataset_variant {
                    name.push('/');
                    name.push_str(v);
                }
                ChartLink {
                    name,
                    slug: key.to_slug(),
                }
            })
        })
        .collect::<Result<_, _>>()?;
    if charts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Group {
            name: "Compression".into(),
            charts,
        }))
    }
}

fn collect_compression_size_group(conn: &Connection) -> Result<Option<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant
          FROM compression_sizes
         GROUP BY dataset, dataset_variant
         ORDER BY dataset, dataset_variant NULLS FIRST
        "#,
    )?;
    let charts: Vec<ChartLink> = stmt
        .query_map([], |row| {
            let dataset: String = row.get(0)?;
            let dataset_variant: Option<String> = row.get(1)?;
            Ok((dataset, dataset_variant))
        })?
        .map(|r| {
            r.map(|(dataset, dataset_variant)| {
                let key = ChartKey::CompressionSize {
                    dataset: dataset.clone(),
                    dataset_variant: dataset_variant.clone(),
                };
                let mut name = dataset;
                if let Some(v) = &dataset_variant {
                    name.push('/');
                    name.push_str(v);
                }
                ChartLink {
                    name,
                    slug: key.to_slug(),
                }
            })
        })
        .collect::<Result<_, _>>()?;
    if charts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Group {
            name: "Compression Size".into(),
            charts,
        }))
    }
}

fn collect_random_access_group(conn: &Connection) -> Result<Option<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT DISTINCT dataset
          FROM random_access_times
         ORDER BY dataset
        "#,
    )?;
    let charts: Vec<ChartLink> = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .map(|r| {
            r.map(|dataset| ChartLink {
                name: dataset.clone(),
                slug: ChartKey::RandomAccess { dataset }.to_slug(),
            })
        })
        .collect::<Result<_, _>>()?;
    if charts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Group {
            name: "Random Access".into(),
            charts,
        }))
    }
}

fn collect_vector_search_groups(conn: &Connection) -> Result<Vec<Group>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, layout, threshold
          FROM vector_search_runs
         GROUP BY dataset, layout, threshold
         ORDER BY dataset, layout, threshold
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })?;

    let mut groups: Vec<Group> = Vec::new();
    let mut current: Option<(String, String)> = None;
    for row in rows {
        let (dataset, layout, threshold) = row?;
        let key = (dataset.clone(), layout.clone());
        if current.as_ref() != Some(&key) {
            groups.push(Group {
                name: format!("{dataset} / {layout}"),
                charts: Vec::new(),
            });
            current = Some(key);
        }
        let slug = ChartKey::VectorSearch {
            dataset,
            layout,
            threshold,
        }
        .to_slug();
        groups
            .last_mut()
            .expect("just pushed")
            .charts
            .push(ChartLink {
                name: format!("threshold={threshold}"),
                slug,
            });
    }
    Ok(groups)
}

fn collect_chart(conn: &Connection, key: &ChartKey) -> Result<Option<ChartResponse>> {
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
        ),
        ChartKey::CompressionTime {
            dataset,
            dataset_variant,
        } => collect_compression_time_chart(conn, dataset, dataset_variant),
        ChartKey::CompressionSize {
            dataset,
            dataset_variant,
        } => collect_compression_size_chart(conn, dataset, dataset_variant),
        ChartKey::RandomAccess { dataset } => collect_random_access_chart(conn, dataset),
        ChartKey::VectorSearch {
            dataset,
            layout,
            threshold,
        } => collect_vector_search_chart(conn, dataset, layout, *threshold),
    }
}

/// Time series rows are gathered keyed by `(commit_sha, series_key)` and then
/// reshaped into the `commits[] / series{}` response shape.
struct SeriesAccumulator {
    commits: Vec<CommitPoint>,
    commit_index: std::collections::BTreeMap<String, usize>,
    series: std::collections::BTreeMap<String, Vec<Option<f64>>>,
}

impl SeriesAccumulator {
    fn new() -> Self {
        Self {
            commits: Vec::new(),
            commit_index: std::collections::BTreeMap::new(),
            series: std::collections::BTreeMap::new(),
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
        }
    }
}

fn collect_query_chart(
    conn: &Connection,
    dataset: &str,
    dataset_variant: &Option<String>,
    scale_factor: &Option<String>,
    storage: &str,
    query_idx: i32,
) -> Result<Option<ChartResponse>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT q.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               c.message, c.url,
               q.engine, q.format, q.value_ns
          FROM query_measurements q
          JOIN commits c USING (commit_sha)
         WHERE q.dataset = ?
           AND q.dataset_variant IS NOT DISTINCT FROM ?
           AND q.scale_factor    IS NOT DISTINCT FROM ?
           AND q.storage = ?
           AND q.query_idx = ?
         ORDER BY c.timestamp, q.engine, q.format
        "#,
    )?;
    let mut acc = SeriesAccumulator::new();
    let rows = stmt.query_map(
        params![
            dataset,
            dataset_variant.as_deref(),
            scale_factor.as_deref(),
            storage,
            query_idx,
        ],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
            ))
        },
    )?;
    let mut any = false;
    for row in rows {
        any = true;
        let (sha, ts, msg, url, engine, format, value_ns) = row?;
        let idx = acc.ensure_commit(&sha, &ts, &msg, &url);
        acc.record(&format!("{engine}:{format}"), idx, value_ns as f64);
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
) -> Result<Option<ChartResponse>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT t.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               c.message, c.url,
               t.format, t.op, t.value_ns
          FROM compression_times t
          JOIN commits c USING (commit_sha)
         WHERE t.dataset = ?
           AND t.dataset_variant IS NOT DISTINCT FROM ?
         ORDER BY c.timestamp, t.format, t.op
        "#,
    )?;
    let mut acc = SeriesAccumulator::new();
    let rows = stmt.query_map(params![dataset, dataset_variant.as_deref()], |row| {
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
        acc.record(&format!("{format}:{op}"), idx, value_ns as f64);
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
) -> Result<Option<ChartResponse>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT s.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               c.message, c.url,
               s.format, s.value_bytes
          FROM compression_sizes s
          JOIN commits c USING (commit_sha)
         WHERE s.dataset = ?
           AND s.dataset_variant IS NOT DISTINCT FROM ?
         ORDER BY c.timestamp, s.format
        "#,
    )?;
    let mut acc = SeriesAccumulator::new();
    let rows = stmt.query_map(params![dataset, dataset_variant.as_deref()], |row| {
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

fn collect_random_access_chart(conn: &Connection, dataset: &str) -> Result<Option<ChartResponse>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT r.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               c.message, c.url,
               r.format, r.value_ns
          FROM random_access_times r
          JOIN commits c USING (commit_sha)
         WHERE r.dataset = ?
         ORDER BY c.timestamp, r.format
        "#,
    )?;
    let mut acc = SeriesAccumulator::new();
    let rows = stmt.query_map(params![dataset], |row| {
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
) -> Result<Option<ChartResponse>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT v.commit_sha,
               CAST(c.timestamp AS VARCHAR),
               c.message, c.url,
               v.flavor, v.value_ns
          FROM vector_search_runs v
          JOIN commits c USING (commit_sha)
         WHERE v.dataset = ?
           AND v.layout = ?
           AND v.threshold = ?
         ORDER BY c.timestamp, v.flavor
        "#,
    )?;
    let mut acc = SeriesAccumulator::new();
    let rows = stmt.query_map(params![dataset, layout, threshold], |row| {
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
