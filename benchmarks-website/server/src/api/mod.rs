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
use axum::response::IntoResponse;
use duckdb::Connection;

pub(crate) use self::charts::chart_payload;
pub(crate) use self::charts::collect_group_charts;
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
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// Handler for `GET /api/groups`.
pub async fn groups(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let groups = db::run_blocking(&state.db, |conn| collect_groups(conn)).await?;
    Ok(Json(GroupsResponse { groups }))
}

/// Handler for `GET /api/chart/{slug}`.
pub async fn chart(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(q): Query<ChartQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let key = ChartKey::from_slug(&slug)
        .map_err(|e| ApiError::BadRequest(format!("invalid slug: {e}")))?;
    let window = q.window();
    let response =
        db::run_blocking(&state.db, move |conn| chart_payload(conn, &key, &window)).await?;
    let response =
        response.ok_or_else(|| ApiError::NotFound(format!("no data for slug {slug:?}")))?;
    Ok(Json(response))
}

/// Handler for `GET /api/group/{slug}`.
pub async fn group(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(q): Query<ChartQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let key = GroupKey::from_slug(&slug)
        .map_err(|e| ApiError::BadRequest(format!("invalid group slug: {e}")))?;
    let window = q.window();
    let response = db::run_blocking(&state.db, move |conn| {
        collect_group_charts(conn, &key, &window)
    })
    .await?;
    let response =
        response.ok_or_else(|| ApiError::NotFound(format!("no data for group slug {slug:?}")))?;
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
