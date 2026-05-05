// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! HTML routes for the bench.vortex.dev v3 web UI.
//!
//! Three pages, all backed by the same per-chart UX:
//! - `GET /` — landing page. Every group is a collapsible `<details>`,
//!   all collapsed by default; the user picks which to expand. The
//!   *first* group's chart payloads are still pre-inlined in the HTML
//!   so opening it skips the JS fetch round-trip; every other group
//!   ships only chart-card shells and is fetched on first toggle.
//! - `GET /chart/{slug}` — single chart page; permalink for sharing.
//! - `GET /group/{slug}` — every chart in one group on a single page.
//!
//! Each chart card owns its own compact toolbar (scope slider + Y-axis). There
//! is no page-level toolbar — every chart is independent. Scope is
//! **zoom-as-scope**: each chart fetches a generous window once, then the
//! toolbar manipulates `chart.options.scales.x.min`/`max` to set the visible
//! window. No refetches on scope change.
//!
//! Every HTML route defaults to the unbounded commit window
//! ([`CommitWindow::All`]) so users can pan/zoom all the way back to the
//! very first commit. The chart payload is sent **raw** — any visual
//! downsampling happens client-side in `chart-init.js`, applied only to
//! the currently visible commit range. The common case (a chart zoomed in
//! to the last ~100 commits) renders raw with no LTTB at all.
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

use anyhow::Result;
use axum::Router;
use axum::extract::Path;
use axum::extract::Query;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use duckdb::Connection;
use serde::Deserialize;

use self::chart::chart_body;
use self::chart::group_body;
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
use crate::app::AppState;
use crate::db;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// Commits to inline for the first group's pre-fetched chart payloads.
/// The chart's initial visible window is ~100 commits; bigger windows
/// just bloat the cold-page HTML. Users who zoom out trigger a refetch
/// with `?n=all` via `chart-init.js`.
const LANDING_INLINE_N: u32 = 100;

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
    /// Resolve the [`CommitWindow`] for HTML routes. Defaults to
    /// [`CommitWindow::All`] so users can pan/zoom all the way back to
    /// the very first commit on every chart. Visual downsampling
    /// happens client-side on the visible commit range only.
    fn fetch_window(&self) -> CommitWindow {
        match self.n.as_deref() {
            Some(_) => CommitWindow::parse(self.n.as_deref()),
            None => CommitWindow::All,
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
    // The landing page intentionally ignores `?n=` for the inline payloads —
    // they are always capped at [`LANDING_INLINE_N`] commits (see
    // [`collect_landing_groups`]) so the cold HTML stays small. Power users
    // with `?n=all` in the URL still get the unbounded view: `chart-init.js`
    // refetches via `/api/chart/{slug}?n=all` when they zoom past the
    // inlined range.
    let filter = ui.filter_state();
    let result = db::run_blocking(&state.db, move |conn| {
        let groups = collect_landing_groups(conn)?;
        let universe = api::collect_filter_universe(conn)?;
        Ok::<_, anyhow::Error>((groups, universe))
    })
    .await;
    let (groups, universe) = match result {
        Ok(g) => g,
        Err(err) => {
            tracing::error!(error = ?err, "landing: collect_landing_groups failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };
    let scripts = if groups.is_empty() {
        PageScripts::Empty
    } else {
        PageScripts::Chart
    };
    render_page(
        "bench.vortex.dev",
        "Vortex benchmarks (v3 alpha)",
        landing_body(&groups),
        scripts,
        Some(&universe),
        &filter,
    )
    .into_response()
}

/// Build a landing-page view: every group, with the first group's payloads
/// inlined and the rest left as shells. Groups whose discovery query
/// returns no data are dropped, but a group whose charts simply have no data
/// inside the inlined window is preserved as a shell so the user can see
/// it (and the lazy-fetch retry path can populate it on toggle).
///
/// Inline payloads are always capped at [`LANDING_INLINE_N`] commits — the
/// chart's initial visible range is ~100 commits anyway, and the bytes
/// saved on the cold-page HTML matter much more than a slightly different
/// fully-zoomed view that the user only sees if they zoom out (at which
/// point `chart-init.js` refetches `?n=all` from the API).
fn collect_landing_groups(conn: &Connection) -> Result<Vec<LandingGroup>> {
    let groups = api::collect_groups(conn)?;
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let inline_window = CommitWindow::Last(
        std::num::NonZeroU32::new(LANDING_INLINE_N).expect("LANDING_INLINE_N is non-zero"),
    );
    let mut out = Vec::with_capacity(groups.len());
    for (i, group) in groups.into_iter().enumerate() {
        let inlined = if i == 0 {
            // First group in canonical order: pre-fetch every chart so
            // the moment the user expands it the chart hydrates from
            // the inline JSON without a JS round-trip.
            let mut v = Vec::with_capacity(group.charts.len());
            for link in &group.charts {
                let key = ChartKey::from_slug(&link.slug)?;
                let payload = api::chart_payload(conn, &key, &inline_window)?;
                v.push(payload.map(|chart| api::NamedChartResponse {
                    name: link.name.clone(),
                    slug: link.slug.clone(),
                    chart,
                }));
            }
            v
        } else {
            // Other groups: ship only the shells. The client fetches on
            // first `details.toggle`.
            (0..group.charts.len()).map(|_| None).collect()
        };
        out.push(LandingGroup {
            name: group.name,
            description: group.description,
            summary: group.summary,
            chart_links: group.charts,
            inlined,
        });
    }
    Ok(out)
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
    let result = db::run_blocking(&state.db, move |conn| {
        api::chart_payload(conn, &key, &window)
    })
    .await;
    let chart = match result {
        Ok(Some(c)) => c,
        Ok(None) => return error_page(StatusCode::NOT_FOUND, "chart not found").into_response(),
        Err(err) => {
            tracing::error!(error = ?err, "chart_page: chart_payload failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };

    let payload_json = match serde_json::to_string(&chart) {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(error = ?err, "chart_page: serialize failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };

    let title = format!("{} — bench.vortex.dev", chart.display_name);
    let subtitle = chart.display_name.clone();
    let filter = ui.filter_state();
    let universe_result =
        db::run_blocking(&state.db, |conn| api::collect_filter_universe(conn)).await;
    let universe = universe_result.ok();
    render_page(
        &title,
        &subtitle,
        chart_body(&chart, &slug, &payload_json),
        PageScripts::Chart,
        universe.as_ref(),
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
    let window = ui.fetch_window();
    let result = db::run_blocking(&state.db, move |conn| {
        api::collect_group_charts(conn, &key, &window)
    })
    .await;
    let group = match result {
        Ok(Some(g)) => g,
        Ok(None) => return error_page(StatusCode::NOT_FOUND, "group not found").into_response(),
        Err(err) => {
            tracing::error!(error = ?err, "group_page: collect_group_charts failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };
    let title = format!("{} — bench.vortex.dev", group.name);
    let subtitle = group.name.clone();
    let filter = ui.filter_state();
    let universe_result =
        db::run_blocking(&state.db, |conn| api::collect_filter_universe(conn)).await;
    let universe = universe_result.ok();
    render_page(
        &title,
        &subtitle,
        group_body(&group),
        PageScripts::Chart,
        universe.as_ref(),
        &filter,
    )
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_window_default_is_all() {
        let ui = UiQuery::default();
        assert!(matches!(ui.fetch_window(), CommitWindow::All));
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
