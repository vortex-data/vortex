// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! HTML routes for the bench.vortex.dev v3 web UI.
//!
//! Three pages, all backed by the same per-chart UX:
//! - `GET /` — landing page. Every group is a collapsible `<details>`,
//!   all collapsed by default; the user picks which to expand. Every group
//!   ships chart-card shells plus versioned shard metadata, and JS hydrates
//!   the latest-100 payloads from materialized artifacts on intent/open.
//! - `GET /chart/{slug}` — single chart page; permalink for sharing.
//! - `GET /group/{slug}` — every chart shell in one group on a single page,
//!   opened by default and hydrated through the same shard path.
//!
//! Each chart card owns its own compact toolbar (scope slider + Y-axis). There
//! is no page-level toolbar — every chart is independent. Scope is
//! **zoom-as-scope**: each chart fetches a generous window once, then the
//! toolbar manipulates `chart.options.scales.x.min`/`max` to set the visible
//! window. No refetches on scope change.
//!
//! HTML routes default to the latest-100 materialized window. Users who
//! pan/zoom beyond that window trigger an explicit `?n=all` chart fetch.
//! Visual downsampling happens client-side in `chart-init.js`, applied only
//! to the currently visible commit range.
//!
//! URL query param `?n=` is accepted as a power-user override on the
//! initial fetch but is not written back from the toolbar. Per-chart UI
//! state is intentionally not persisted in the URL — the user feedback
//! emphasised that this UX should feel local-and-immediate, not "share a
//! perfect view via URL". Permalinks (`/chart/{slug}`, `/group/{slug}`)
//! are the sharing mechanism, not query strings.
//!
//! Slugs are opaque strings the server received from `/api/groups`; the
//! handler echoes them straight into [`crate::slug::ChartKey::from_slug`]
//! (or [`crate::slug::GroupKey::from_slug`]) without parsing.
//!
//! Static assets (Chart.js + zoom plugin + CSS + the small hydration
//! script) are served from `/static/...` via [`include_bytes!`] so the
//! binary is fully self-contained.
//!
//! Submodules (all crate-private):
//! - `render`        — page chrome (header, theme bootstrap, error page,
//!   `escape_json_for_script`).
//! - `landing`       — landing-page body + chart-card shell rendering.
//! - `chart`         — chart and group permalink page bodies.
//! - `summary`       — group summary card rendering.
//! - `filter`        — filter dropdown + on-page filter-state JSON.
//! - `toolbar`       — per-chart scope slider, Y-axis switch, range strip.
//! - `static_assets` — `include_bytes!`'d JS/CSS/PNG handlers.

mod chart;
mod filter;
mod landing;
mod render;
mod static_assets;
mod summary;
mod toolbar;

use axum::Router;
use axum::extract::Path;
use axum::extract::Query;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use serde::Deserialize;

use self::chart::chart_body;
use self::landing::LandingGroup;
use self::landing::landing_body;
use self::render::PageScripts;
use self::render::error_page;
use self::render::render_page;
use self::static_assets::serve_chart_init_js;
use self::static_assets::serve_chart_js;
use self::static_assets::serve_chart_zoom_js;
use self::static_assets::serve_style_css;
use self::static_assets::serve_vortex_black_png;
use self::static_assets::serve_vortex_white_png;
use crate::api;
use crate::api::CommitWindow;
use crate::api::Group;
use crate::app::AppState;
use crate::read_model::ReadGeneration;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// HTML routes mounted under `/`.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(landing))
        .route("/chart/{slug}", get(chart_page))
        .route("/group/{slug}", get(group_page))
        .route("/static/chart.umd.js", get(serve_chart_js))
        .route(
            "/static/chartjs-plugin-zoom.umd.min.js",
            get(serve_chart_zoom_js),
        )
        .route("/static/chart-init.js", get(serve_chart_init_js))
        .route("/static/style.css", get(serve_style_css))
        .route("/Vortex_Black_NoBG.png", get(serve_vortex_black_png))
        .route("/Vortex_White_NoBG.png", get(serve_vortex_white_png))
}

/// Query string for HTML routes. `?n=` overrides the commit window;
/// `?engine=` and `?format=` carry the global filter bar's selection so a
/// shared link or refresh preserves which engines/formats are visible. The
/// per-chart toolbar (Y axis, scope slider) remains local-only — its state
/// is intentionally not in the URL.
#[derive(Debug, Default, Deserialize)]
pub struct UiQuery {
    /// Override for the per-chart fetch size. Accepts `25|50|100|250|all`.
    pub n: Option<String>,
    /// Comma-separated list of engines to keep visible across every chart.
    /// Empty / unset means no engine filter is active. Unknown engines are
    /// preserved verbatim so a stale URL still survives a chip-universe
    /// expansion.
    pub engine: Option<String>,
    /// Comma-separated list of formats to keep visible across every chart.
    /// Same shape as `engine`.
    pub format: Option<String>,
}

impl UiQuery {
    /// Resolve the [`CommitWindow`] for HTML routes. Defaults to the
    /// materialized latest-100 window; `?n=all` opts into the slower
    /// full-history fallback.
    fn fetch_window(&self) -> CommitWindow {
        match self.n.as_deref() {
            Some(_) => CommitWindow::parse(self.n.as_deref()),
            None => CommitWindow::default(),
        }
    }

    /// Parse `?engine=`/`?format=` into a deduplicated, trimmed [`FilterState`].
    /// Empty entries (e.g. trailing commas) are dropped; an entirely empty
    /// param means "no filter active" and is encoded as an empty `Vec`.
    fn filter_state(&self) -> FilterState {
        FilterState {
            engines: parse_csv(self.engine.as_deref()),
            formats: parse_csv(self.format.as_deref()),
        }
    }
}

/// Parsed filter selection from `?engine=` / `?format=`.
///
/// An empty `Vec` means "all chips active" (no filter); a non-empty `Vec`
/// is the explicit allowlist for that dimension.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct FilterState {
    /// Engine names the URL allowlists; empty means no filter.
    pub engines: Vec<String>,
    /// Format names the URL allowlists; empty means no filter.
    pub formats: Vec<String>,
}

fn parse_csv(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw else { return Vec::new() };
    let mut seen = std::collections::BTreeSet::new();
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter(|s| seen.insert(s.to_string()))
        .map(str::to_string)
        .collect()
}

async fn landing(State(state): State<AppState>, Query(ui): Query<UiQuery>) -> Response {
    // The landing page intentionally ignores `?n=` for group hydration. It
    // always starts from the materialized latest-100 shards, and
    // `chart-init.js` fetches `/api/chart/{slug}?n=all` only after a user asks
    // for history beyond that loaded window.
    let filter = ui.filter_state();
    let generation = state.read_store.active();
    let groups = generation.groups();
    let universe = generation.filter_universe();
    let landing_groups = collect_landing_groups(&generation, &groups, None);
    let scripts = if landing_groups.is_empty() {
        PageScripts::Empty
    } else {
        PageScripts::Chart
    };
    render_page(
        "bench.vortex.dev",
        "Vortex benchmarks (v3 alpha)",
        landing_body(&landing_groups, universe.as_ref()),
        scripts,
        Some(universe.as_ref()),
        &filter,
    )
    .into_response()
}

/// Build a landing/group-page shell view. Each group carries the active
/// generation id and shard prefix; no chart payload JSON is inlined.
fn collect_landing_groups(
    generation: &ReadGeneration,
    groups: &[Group],
    open_slug: Option<&str>,
) -> Vec<LandingGroup> {
    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let shard_prefix = format!(
            "/api/artifacts/{}/groups/{}/shards/",
            generation.id(),
            group.slug
        );
        out.push(LandingGroup {
            name: group.name.clone(),
            slug: group.slug.clone(),
            generation: generation.id().to_string(),
            shard_count: generation.group_shard_count(&group.slug),
            shard_prefix,
            open: open_slug == Some(group.slug.as_str()),
            description: group.description.clone(),
            summary: group.summary.clone(),
            chart_links: group.charts.clone(),
        });
    }
    out
}

async fn chart_page(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(ui): Query<UiQuery>,
) -> Response {
    let key = match ChartKey::from_slug(&slug) {
        Ok(key) => key,
        Err(err) => {
            tracing::warn!(error = ?err, slug, "chart_page: invalid slug");
            return error_page(StatusCode::NOT_FOUND, "chart not found").into_response();
        }
    };

    let window = ui.fetch_window();
    let (chart, payload_json) = if is_materialized_window(&window) {
        let generation = state.read_store.active();
        let Some(chart) = generation.chart_payload(&slug) else {
            return error_page(StatusCode::NOT_FOUND, "chart not found").into_response();
        };
        let Some(artifact) = generation.chart_artifact(&slug) else {
            return error_page(StatusCode::NOT_FOUND, "chart not found").into_response();
        };
        let payload_json = match std::str::from_utf8(artifact.identity()) {
            Ok(s) => s.to_string(),
            Err(err) => {
                tracing::error!(error = ?err, "chart_page: artifact utf8 failed");
                return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
                    .into_response();
            }
        };
        (chart, payload_json)
    } else {
        let chart = match api::cached_chart_payload(&state, &slug, &key, &window).await {
            Ok(Some(c)) => c,
            Ok(None) => {
                return error_page(StatusCode::NOT_FOUND, "chart not found").into_response();
            }
            Err(err) => {
                tracing::error!(error = ?err, "chart_page: chart_payload failed");
                return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
                    .into_response();
            }
        };

        let payload_json = match serde_json::to_string(&*chart) {
            Ok(s) => s,
            Err(err) => {
                tracing::error!(error = ?err, "chart_page: serialize failed");
                return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
                    .into_response();
            }
        };
        (chart, payload_json)
    };

    let title = format!("{} — bench.vortex.dev", chart.display_name);
    let subtitle = chart.display_name.clone();
    let filter = ui.filter_state();
    let universe = api::cached_filter_universe(&state).await.ok();
    render_page(
        &title,
        &subtitle,
        chart_body(&chart, &slug, &payload_json, &window),
        PageScripts::Chart,
        universe.as_deref(),
        &filter,
    )
    .into_response()
}

async fn group_page(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(ui): Query<UiQuery>,
) -> Response {
    let key = match GroupKey::from_slug(&slug) {
        Ok(k) => k,
        Err(err) => {
            tracing::warn!(error = ?err, slug, "group_page: invalid slug");
            return error_page(StatusCode::NOT_FOUND, "group not found").into_response();
        }
    };
    let group_slug = key.to_slug();
    let generation = state.read_store.active();
    let groups = generation.groups();
    let Some(group) = groups.iter().find(|group| group.slug == group_slug) else {
        return error_page(StatusCode::NOT_FOUND, "group not found").into_response();
    };
    let title = format!("{} — bench.vortex.dev", group.name);
    let subtitle = group.name.clone();
    let filter = ui.filter_state();
    let universe = generation.filter_universe();
    let group_shell =
        collect_landing_groups(&generation, std::slice::from_ref(group), Some(&group_slug));
    render_page(
        &title,
        &subtitle,
        landing_body(&group_shell, universe.as_ref()),
        PageScripts::Chart,
        Some(universe.as_ref()),
        &filter,
    )
    .into_response()
}

fn is_materialized_window(window: &CommitWindow) -> bool {
    matches!(window, CommitWindow::Last(n) if n.get() == api::DEFAULT_COMMIT_WINDOW)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_window_default_is_latest_100() {
        let ui = UiQuery::default();
        match ui.fetch_window() {
            CommitWindow::Last(n) => assert_eq!(n.get(), api::DEFAULT_COMMIT_WINDOW),
            CommitWindow::All => panic!(),
        }
    }

    #[test]
    fn fetch_window_respects_n_override() {
        let ui = UiQuery {
            n: Some("25".into()),
            ..Default::default()
        };
        match ui.fetch_window() {
            CommitWindow::Last(n) => assert_eq!(n.get(), 25),
            CommitWindow::All => panic!(),
        }
        let ui = UiQuery {
            n: Some("all".into()),
            ..Default::default()
        };
        assert!(matches!(ui.fetch_window(), CommitWindow::All));
    }

    #[test]
    fn filter_state_parses_csv_and_dedupes() {
        let ui = UiQuery {
            engine: Some("duckdb, datafusion ,duckdb".into()),
            format: Some(",, vortex-file-compressed ,".into()),
            ..Default::default()
        };
        let f = ui.filter_state();
        assert_eq!(f.engines, vec!["duckdb", "datafusion"]);
        assert_eq!(f.formats, vec!["vortex-file-compressed"]);
    }

    #[test]
    fn filter_state_default_is_empty() {
        let f = UiQuery::default().filter_state();
        assert!(f.engines.is_empty());
        assert!(f.formats.is_empty());
    }
}
