// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-side API: `/api/groups`, `/api/chart/{slug}`, `/api/group/{slug}`,
//! `/health`.
//!
//! Group / chart / series fit follows the layout in [`crate::schema`]:
//! one fact table per measurement family, each with a known group / chart /
//! series tuple. Slugs round-trip through [`crate::slug::ChartKey`] and
//! [`crate::slug::GroupKey`].
//!
//! Submodules:
//! - [`mod@dto`]          — every wire-shape struct (`Group`, `ChartResponse`, …).
//! - [`mod@window`]       — [`CommitWindow`] + [`ChartQuery`].
//! - [`mod@groups`]       — discovery passes that build the group / chart-link tree.
//! - [`mod@summary`]      — v2-compatible per-group rollups.
//! - [`mod@charts`]       — `chart_payload` + the per-fact-table `collect_*_chart`
//!   functions and their shared `SeriesAccumulator`.
//! - [`mod@filter`]       — chip-universe collection for the global filter bar.
//! - [`mod@descriptions`] — editorial blurbs surfaced as hover tooltips.

pub mod charts;
pub mod descriptions;
pub mod dto;
pub mod filter;
pub mod groups;
pub mod summary;
pub mod window;

use anyhow::Result;
use axum::Json;
use axum::extract::Path;
use axum::extract::Query;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::response::Response;
use duckdb::Connection;

pub(crate) use self::charts::chart_payload;
pub(crate) use self::charts::collect_group_charts;
pub use self::dto::ChartHistory;
pub use self::dto::ChartLink;
pub use self::dto::ChartResponse;
pub use self::dto::CommitPoint;
pub use self::dto::DEFAULT_COMMIT_WINDOW;
pub use self::dto::FilterUniverse;
pub use self::dto::GROUP_ORDER;
pub use self::dto::Group;
pub use self::dto::GroupChartsResponse;
pub use self::dto::GroupsResponse;
pub use self::dto::HealthResponse;
pub use self::dto::NamedChartResponse;
pub use self::dto::QueryRanking;
pub use self::dto::RandomAccessRanking;
pub use self::dto::RowCounts;
pub use self::dto::SeriesTag;
pub use self::dto::Summary;
pub use self::dto::UnitKind;
pub use self::dto::group_sort_key;
pub use self::filter::collect_filter_universe;
pub(crate) use self::groups::collect_groups;
pub use self::window::ChartQuery;
pub use self::window::CommitWindow;
use crate::app::AppState;
use crate::db;
use crate::error::ApiError;
use crate::read_model::ArtifactCachePolicy;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

pub(crate) fn read_transaction<T>(
    conn: &mut Connection,
    f: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    conn.execute_batch("BEGIN TRANSACTION")?;
    let result = f(conn);
    match result {
        Ok(value) => {
            conn.execute_batch("COMMIT")?;
            Ok(value)
        }
        Err(err) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(err)
        }
    }
}

/// Handler for `GET /api/groups`.
pub async fn groups(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let generation = state.read_store.active();
    Ok(generation
        .groups_artifact()
        .response(&headers, ArtifactCachePolicy::Revalidate))
}

/// Handler for `GET /api/chart/{slug}`.
pub async fn chart(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(q): Query<ChartQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let key = ChartKey::from_slug(&slug)
        .map_err(|e| ApiError::BadRequest(format!("invalid slug: {e}")))?;
    let window = q.window();
    if is_materialized_window(&window) {
        let generation = state.read_store.active();
        let response = generation
            .chart_artifact(&slug)
            .ok_or_else(|| ApiError::NotFound(format!("no data for slug {slug:?}")))?;
        return Ok(response.response(&headers, ArtifactCachePolicy::Revalidate));
    }
    let response = cached_chart_payload(&state, &slug, &key, &window).await?;
    let response =
        response.ok_or_else(|| ApiError::NotFound(format!("no data for slug {slug:?}")))?;
    Ok(Json(response).into_response())
}

/// Handler for `GET /api/group/{slug}`.
pub async fn group(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(q): Query<ChartQuery>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let key = GroupKey::from_slug(&slug)
        .map_err(|e| ApiError::BadRequest(format!("invalid group slug: {e}")))?;
    let window = q.window();
    if is_materialized_window(&window) {
        let generation = state.read_store.active();
        let response = generation
            .group_artifact(&slug)
            .ok_or_else(|| ApiError::NotFound(format!("no data for group slug {slug:?}")))?;
        return Ok(response.response(&headers, ArtifactCachePolicy::Revalidate));
    }
    let response = cached_group_charts(&state, &slug, &key, &window).await?;
    let response =
        response.ok_or_else(|| ApiError::NotFound(format!("no data for group slug {slug:?}")))?;
    Ok(Json(response).into_response())
}

/// Handler for versioned latest-100 group shard artifacts.
pub async fn group_shard_artifact(
    State(state): State<AppState>,
    Path((generation_id, group_slug, index)): Path<(String, String, usize)>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let generation = state
        .read_store
        .generation(&generation_id)
        .ok_or_else(|| ApiError::NotFound(format!("unknown generation {generation_id:?}")))?;
    let artifact = generation
        .group_shard_artifact(&group_slug, index)
        .ok_or_else(|| ApiError::NotFound(format!("unknown group shard {group_slug:?}#{index}")))?;
    Ok(artifact.response(&headers, ArtifactCachePolicy::Immutable))
}

fn is_materialized_window(window: &CommitWindow) -> bool {
    matches!(window, CommitWindow::Last(n) if n.get() == DEFAULT_COMMIT_WINDOW)
}

/// Cache-aware wrapper around `collect_groups`.
pub async fn cached_groups(state: &AppState) -> Result<std::sync::Arc<Vec<Group>>> {
    let db = state.db.clone();
    state
        .cache
        .groups(move || async move {
            db::run_read_blocking(&db, |conn| read_transaction(conn, collect_groups)).await
        })
        .await
}

/// Cache-aware wrapper around [`collect_filter_universe`].
pub async fn cached_filter_universe(state: &AppState) -> Result<std::sync::Arc<FilterUniverse>> {
    let db = state.db.clone();
    state
        .cache
        .filter_universe(move || async move {
            db::run_read_blocking(&db, |conn| read_transaction(conn, collect_filter_universe)).await
        })
        .await
}

/// Cache-aware wrapper around `chart_payload`.
pub async fn cached_chart_payload(
    state: &AppState,
    slug: &str,
    key: &ChartKey,
    window: &CommitWindow,
) -> Result<Option<std::sync::Arc<ChartResponse>>> {
    let db = state.db.clone();
    let key_for_compute = key.clone();
    let window_for_compute = *window;
    state
        .cache
        .chart_payload(slug, window, move || async move {
            db::run_read_blocking(&db, move |conn| {
                read_transaction(conn, |conn| {
                    chart_payload(conn, &key_for_compute, &window_for_compute)
                })
            })
            .await
        })
        .await
}

/// Cache-aware wrapper around `collect_group_charts`.
pub async fn cached_group_charts(
    state: &AppState,
    slug: &str,
    key: &GroupKey,
    window: &CommitWindow,
) -> Result<Option<std::sync::Arc<GroupChartsResponse>>> {
    let db = state.db.clone();
    let key_for_compute = key.clone();
    let window_for_compute = *window;
    state
        .cache
        .group_charts(slug, window, move || async move {
            db::run_read_blocking(&db, move |conn| {
                read_transaction(conn, |conn| {
                    collect_group_charts(conn, &key_for_compute, &window_for_compute)
                })
            })
            .await
        })
        .await
}

/// Handler for `GET /health`.
pub async fn health(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let path = state.db_path.display().to_string();
    let response = db::run_read_blocking(&state.db, move |conn| {
        read_transaction(conn, |conn| collect_health(conn, path))
    })
    .await?;
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
