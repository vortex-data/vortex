// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! HTML routes for the bench.vortex.dev v3 web UI.
//!
//! Three pages, all backed by the same per-chart UX:
//! - `GET /` — landing page. Every group is a collapsible `<details>`. The
//!   first group is open by default and its charts pre-inline their JSON
//!   payload for a fast first paint; closed groups carry only the chart-card
//!   shell and their payloads are fetched on first toggle (`details.open`).
//! - `GET /chart/{slug}` — single chart page; permalink for sharing.
//! - `GET /group/{slug}` — every chart in one group on a single page.
//!
//! Each chart card owns its own compact toolbar (scope slider + Y-axis). There
//! is no page-level toolbar — every chart is independent. Scope is
//! **zoom-as-scope**: each chart fetches up to [`api::MAX_COMMIT_WINDOW`]
//! commits once, then the toolbar manipulates `chart.options.scales.x.min`/
//! `max` to set the visible window. No refetches on scope change.
//!
//! URL query params (`?n=`) are accepted as power-user overrides on the
//! initial fetch but are not written back from the toolbar. Per-chart UI
//! state is intentionally not persisted in the URL — the user feedback
//! emphasised that this UX should feel local-and-immediate, not "share a
//! perfect view via URL". Permalinks (`/chart/{slug}`, `/group/{slug}`) are
//! the sharing mechanism, not query strings.
//!
//! Slugs are opaque strings the server received from `/api/groups`; the
//! handler echoes them straight into [`crate::slug::ChartKey::from_slug`]
//! (or [`crate::slug::GroupKey::from_slug`]) without parsing.
//!
//! Static assets (Chart.js + zoom plugin + CSS + the small hydration
//! script) are served from `/static/...` via [`include_bytes!`] so the
//! binary is fully self-contained.

use std::num::NonZeroU32;

use anyhow::Result;
use axum::Router;
use axum::extract::Path;
use axum::extract::Query;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use duckdb::Connection;
use maud::DOCTYPE;
use maud::Markup;
use maud::PreEscaped;
use maud::html;
use serde::Deserialize;

use crate::api;
use crate::api::ChartResponse;
use crate::api::CommitWindow;
use crate::api::GroupChartsResponse;
use crate::api::NamedChartResponse;
use crate::api::Summary;
use crate::app::AppState;
use crate::db;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// How many commits each chart pre-fetches. The toolbar's scope slider zooms
/// into smaller windows of this slice; we never refetch on scope change.
/// Capped at the API ceiling so a future bigger ceiling is picked up here too.
const PER_CHART_FETCH_N: u32 = api::MAX_COMMIT_WINDOW;

const CHART_JS: &[u8] = include_bytes!("../static/chart.umd.js");
const CHART_ZOOM_JS: &[u8] = include_bytes!("../static/chartjs-plugin-zoom.umd.min.js");
const CHART_INIT_JS: &[u8] = include_bytes!("../static/chart-init.js");
const STYLE_CSS: &[u8] = include_bytes!("../static/style.css");
const VORTEX_BLACK_SVG: &[u8] = include_bytes!("../../public/vortex_black_nobg.svg");
const VORTEX_WHITE_SVG: &[u8] = include_bytes!("../../public/vortex_white_nobg.svg");
const STATIC_ASSET_VERSION: &str = "bench-v3-ui-10";

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
        .route("/vortex_black_nobg.svg", get(serve_vortex_black_svg))
        .route("/vortex_white_nobg.svg", get(serve_vortex_white_svg))
}

/// Query string for HTML routes. `?n=` overrides the per-chart fetch size;
/// `?engine=` and `?format=` carry the global filter bar's selection so a
/// shared link or refresh preserves which engines/formats are visible. The
/// per-chart toolbar (Y axis, scope slider) remains local-only — its state
/// is intentionally not in the URL.
#[derive(Debug, Default, Deserialize)]
pub struct UiQuery {
    /// Override for the per-chart fetch size. Defaults to `PER_CHART_FETCH_N`.
    /// Accepts `25|50|100|250|all`.
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
    /// Resolve the [`CommitWindow`] for the initial fetch. When `?n=` is
    /// unset, falls back to [`PER_CHART_FETCH_N`].
    fn fetch_window(&self) -> CommitWindow {
        match self.n.as_deref() {
            Some(_) => CommitWindow::parse(self.n.as_deref()),
            None => {
                CommitWindow::Last(NonZeroU32::new(PER_CHART_FETCH_N).expect("non-zero default"))
            }
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
    pub engines: Vec<String>,
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
    let window = ui.fetch_window();
    let filter = ui.filter_state();
    let result = db::run_blocking(&state.db, move |conn| {
        let groups = collect_landing_groups(conn, &window)?;
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

/// One group's worth of data for the landing page.
///
/// The first group (in canonical order) ships with `charts` populated so the
/// open-by-default `<details>` paints immediately. Subsequent groups ship
/// with `charts` empty and only their chart-card shells — payloads are
/// fetched client-side on first `details.toggle` to keep the cold landing
/// HTML small.
struct LandingGroup {
    name: String,
    summary: Option<Summary>,
    /// Chart links for every chart in the group. Always present — we need
    /// the slugs server-side so the chart-card shell can carry
    /// `data-chart-slug` for the lazy fetch.
    chart_links: Vec<api::ChartLink>,
    /// Pre-fetched payloads. Populated only for the open-by-default group.
    /// `Vec` indices line up with `chart_links`.
    inlined: Vec<Option<NamedChartResponse>>,
}

/// Build a landing-page view: every group, with the first group's payloads
/// inlined and the rest left as shells. Groups whose discovery query
/// returns no data are dropped, but a group whose charts simply have no data
/// inside the requested window is preserved as a shell so the user can see
/// it (and the lazy-fetch retry path can populate it on toggle).
fn collect_landing_groups(conn: &Connection, window: &CommitWindow) -> Result<Vec<LandingGroup>> {
    let groups = api::collect_groups(conn)?;
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(groups.len());
    for (i, group) in groups.into_iter().enumerate() {
        let inlined = if i == 0 {
            // First (open-by-default) group: pre-fetch every chart so the
            // first paint is fast and there is no JS round-trip.
            let mut v = Vec::with_capacity(group.charts.len());
            for link in &group.charts {
                let key = ChartKey::from_slug(&link.slug)?;
                let payload = api::chart_payload(conn, &key, window)?;
                v.push(payload.map(|chart| NamedChartResponse {
                    name: link.name.clone(),
                    slug: link.slug.clone(),
                    chart,
                }));
            }
            v
        } else {
            // Closed groups: ship only the shells. The client fetches on
            // first `details.toggle`.
            (0..group.charts.len()).map(|_| None).collect()
        };
        out.push(LandingGroup {
            name: group.name,
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
        api::collect_chart(conn, &key, &window)
    })
    .await;
    let chart = match result {
        Ok(Some(c)) => c,
        Ok(None) => return error_page(StatusCode::NOT_FOUND, "chart not found").into_response(),
        Err(err) => {
            tracing::error!(error = ?err, "chart_page: collect_chart failed");
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

/// Which scripts the page wants pulled in.
enum PageScripts {
    /// Empty database — skip Chart.js entirely.
    Empty,
    /// Any page with at least one chart-card. Pulls Chart.js + zoom plugin.
    Chart,
}

fn render_page(
    title: &str,
    _header_subtitle: &str,
    body: Markup,
    scripts: PageScripts,
    universe: Option<&api::FilterUniverse>,
    filter: &FilterState,
) -> Markup {
    let style_href = versioned_asset("/static/style.css");
    let chart_js_src = versioned_asset("/static/chart.umd.js");
    let chart_zoom_src = versioned_asset("/static/chartjs-plugin-zoom.umd.min.js");
    let chart_init_src = versioned_asset("/static/chart-init.js");
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { (title) }
                (theme_bootstrap_script())
                link rel="stylesheet" href=(style_href);
            }
            body {
                (filter_state_script(filter))
                (site_header(universe, filter))
                main { (body) }
                @match scripts {
                    PageScripts::Empty => {
                        script src=(chart_init_src) defer {}
                    },
                    PageScripts::Chart => {
                        script src=(chart_js_src) defer {}
                        script src=(chart_zoom_src) defer {}
                        script src=(chart_init_src) defer {}
                    },
                }
            }
        }
    }
}

fn theme_bootstrap_script() -> Markup {
    html! {
        script {
            (PreEscaped(
                r#"(function(){try{var t=localStorage.getItem("bench-theme");if(t==="light"||t==="dark"){document.documentElement.dataset.theme=t;}}catch(e){}})();"#
            ))
        }
    }
}

fn site_header(universe: Option<&api::FilterUniverse>, filter: &FilterState) -> Markup {
    let black_logo = versioned_asset("/vortex_black_nobg.svg");
    let white_logo = versioned_asset("/vortex_white_nobg.svg");
    let show_filters = universe
        .map(|u| !u.engines.is_empty() || !u.formats.is_empty())
        .unwrap_or(false);
    let active_count = filter.engines.len() + filter.formats.len();
    html! {
        header.sticky-header {
            div.header-content {
                div.header-left {
                    a.logo-link href="/" aria-label="bench.vortex.dev home" {
                        img.site-logo.logo-light src=(black_logo) alt="Vortex";
                        img.site-logo.logo-dark src=(white_logo) alt="Vortex";
                    }
                    h1.site-title { "Vortex Benchmarks" }
                }
                div.header-center {
                    div.nav-controls aria-label="Benchmark group controls" {
                        button.control-btn type="button" data-action="expand-all" {
                            (chevrons_down_icon())
                            span { "Expand All" }
                        }
                        button.control-btn type="button" data-action="collapse-all" {
                            (chevrons_up_icon())
                            span { "Collapse All" }
                        }
                        @if show_filters {
                            (filter_dropdown(universe.expect("show_filters guard"), filter, active_count))
                        }
                    }
                }
                div.header-right {
                    a.repo-link href="https://github.com/vortex-data/vortex" rel="noopener noreferrer" target="_blank" {
                        svg.github-logo viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true" {
                            path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" {}
                        }
                        span { "GitHub" }
                    }
                    button.control-btn.theme-toggle type="button" data-role="theme-toggle" data-next-theme="light" aria-label="Toggle color theme" {
                        (sun_icon())
                        (moon_icon())
                        span.theme-toggle-label { "Light" }
                    }
                }
            }
        }
    }
}

fn filter_icon() -> Markup {
    html! {
        svg.btn-icon viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" {
            polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" {}
        }
    }
}

fn chevrons_down_icon() -> Markup {
    html! {
        svg.btn-icon viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" {
            path d="m7 6 5 5 5-5" {}
            path d="m7 13 5 5 5-5" {}
        }
    }
}

fn chevrons_up_icon() -> Markup {
    html! {
        svg.btn-icon viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" {
            path d="m17 18-5-5-5 5" {}
            path d="m17 11-5-5-5 5" {}
        }
    }
}

fn sun_icon() -> Markup {
    html! {
        svg.btn-icon.theme-icon.theme-icon-light viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" {
            circle cx="12" cy="12" r="4" {}
            path d="M12 2v2" {}
            path d="M12 20v2" {}
            path d="m4.93 4.93 1.41 1.41" {}
            path d="m17.66 17.66 1.41 1.41" {}
            path d="M2 12h2" {}
            path d="M20 12h2" {}
            path d="m6.34 17.66-1.41 1.41" {}
            path d="m19.07 4.93-1.41 1.41" {}
        }
    }
}

fn moon_icon() -> Markup {
    html! {
        svg.btn-icon.theme-icon.theme-icon-dark viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" {
            path d="M20.99 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 20.99 12.79z" {}
        }
    }
}

fn landing_body(groups: &[LandingGroup]) -> Markup {
    if groups.is_empty() {
        return html! {
            p.empty { "No data ingested yet." }
        };
    }
    let total_charts: usize = groups.iter().map(|g| g.chart_links.len()).sum();
    // Index every chart globally so `<canvas data-chart-index="N">` and
    // `<script id="chart-data-N">` agree across groups.
    let mut idx_iter = 0usize..total_charts;
    html! {
        @for (group_idx, group) in groups.iter().enumerate() {
            section.group-details data-group-name=(group.name) {
                details.group-disclosure open[group_idx == 0] {
                    summary.group-summary {
                        span.group-summary-row {
                            span.group-name { (group.name) }
                            span.group-count {
                                (group.chart_links.len()) " chart" @if group.chart_links.len() != 1 { "s" }
                            }
                        }
                    }
                }
                (summary_markup(group.summary.as_ref()))
                div.chart-grid {
                    @for (chart_idx, link) in group.chart_links.iter().enumerate() {
                        @let idx = idx_iter.next().expect("indices match charts");
                        @let inlined = group.inlined.get(chart_idx).and_then(|o| o.as_ref());
                        (chart_card(link, idx, inlined))
                    }
                }
            }
        }
        noscript {
            p.no-script { "JavaScript is required to render the charts." }
        }
    }
}

/// Render one chart-card. `inlined` carries the JSON payload when the
/// server pre-fetched it; absent on closed-by-default landing groups, where
/// the JS fetches on first `details.toggle`.
fn chart_card(link: &api::ChartLink, idx: usize, inlined: Option<&NamedChartResponse>) -> Markup {
    let permalink = format!("/chart/{}", link.slug);
    html! {
        section.chart-card data-chart-index=(idx) data-chart-slug=(link.slug) {
            h3.chart-card-title {
                a href=(permalink) { (link.name) }
            }
            (per_chart_toolbar(idx))
            div.chart-tooltip-host {}
            div.chart-wrap {
                canvas data-chart-index=(idx) {}
            }
            (range_strip(idx))
            @if let Some(item) = inlined {
                script id={ "chart-data-" (idx) } type="application/json" {
                    (PreEscaped(escape_json_for_script(
                        &serde_json::to_string(&item.chart)
                            .expect("ChartResponse serialises"),
                    )))
                }
            }
        }
    }
}

fn chart_body(chart: &ChartResponse, slug: &str, payload_json: &str) -> Markup {
    let series_count = chart.series.len();
    let commit_count = chart.commits.len();
    html! {
        p.chart-meta {
            "unit: " code { (chart.unit) }
            " · "
            (series_count) " series · "
            (commit_count) " commit" @if commit_count != 1 { "s" }
        }
        section.chart-card data-chart-index="0" data-chart-slug=(slug) {
            (per_chart_toolbar(0))
            div.chart-tooltip-host {}
            div.chart-wrap {
                canvas data-chart-index="0" {}
            }
            (range_strip(0))
            // Embedded JSON; rendered as text content so JSON `<` / `>` are HTML-escaped.
            script id="chart-data-0" type="application/json" {
                (PreEscaped(escape_json_for_script(payload_json)))
            }
        }
        noscript {
            p.no-script { "JavaScript is required to render the chart." }
        }
    }
}

fn group_body(group: &GroupChartsResponse) -> Markup {
    let chart_count = group.charts.len();
    html! {
        p.chart-meta {
            (chart_count) " chart" @if chart_count != 1 { "s" }
        }
        (summary_markup(group.summary.as_ref()))
        div.chart-grid {
            @for (i, item) in group.charts.iter().enumerate() {
                @let permalink = format!("/chart/{}", item.slug);
                section.chart-card data-chart-index=(i) data-chart-slug=(item.slug) {
                    h3.chart-card-title {
                        a href=(permalink) { (item.name) }
                    }
                    (per_chart_toolbar(i))
                    div.chart-tooltip-host {}
                    div.chart-wrap {
                        canvas data-chart-index=(i) {}
                    }
                    (range_strip(i))
                    script id={ "chart-data-" (i) } type="application/json" {
                        (PreEscaped(escape_json_for_script(
                            &serde_json::to_string(&item.chart)
                                .expect("ChartResponse serialises"),
                        )))
                    }
                }
            }
        }
        noscript {
            p.no-script { "JavaScript is required to render the charts." }
        }
    }
}

fn summary_markup(summary: Option<&Summary>) -> Markup {
    let Some(summary) = summary else {
        return html! {};
    };
    match summary {
        Summary::RandomAccess {
            title,
            rankings,
            explanation,
        } if !rankings.is_empty() => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    @for (idx, item) in rankings.iter().enumerate() {
                        div.score-item {
                            span.score-rank { "#" (idx + 1) }
                            span.score-series title=(item.name) { (item.name) }
                            span.score-metrics {
                                span.score-value { (format_time_ns(item.time)) }
                                span.score-runtime { (format!("{:.2}x", item.ratio)) }
                            }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        Summary::Compression {
            title,
            compress_ratio,
            decompress_ratio,
            dataset_count: _,
            explanation,
        } if compress_ratio.is_some() || decompress_ratio.is_some() => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    @if let Some(v) = compress_ratio {
                        div.score-item {
                            span.score-rank { "⚡" }
                            span.score-series { "Write Speed (Compression)" }
                            span.score-metrics {
                                span.score-value { (format!("{v:.2}x")) }
                            }
                        }
                    }
                    @if let Some(v) = decompress_ratio {
                        div.score-item {
                            span.score-rank { "📤" }
                            span.score-series { "Scan Speed (Decompression)" }
                            span.score-metrics {
                                span.score-value { (format!("{v:.2}x")) }
                            }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        Summary::CompressionSize {
            title,
            min_ratio,
            mean_ratio,
            max_ratio,
            dataset_count: _,
            explanation,
        } => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    div.score-item {
                        span.score-rank { "⬇️" }
                        span.score-series { "Min Size Ratio" }
                        span.score-metrics {
                            span.score-value { (format!("{min_ratio:.2}x")) }
                        }
                    }
                    div.score-item {
                        span.score-rank { "📊" }
                        span.score-series { "Mean Size Ratio" }
                        span.score-metrics {
                            span.score-value { (format!("{mean_ratio:.2}x")) }
                        }
                    }
                    div.score-item {
                        span.score-rank { "⬆️" }
                        span.score-series { "Max Size Ratio" }
                        span.score-metrics {
                            span.score-value { (format!("{max_ratio:.2}x")) }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        Summary::QueryBenchmark {
            title,
            rankings,
            explanation,
        } if !rankings.is_empty() => html! {
            section.benchmark-scores-summary aria-label=(title) {
                h3.scores-title { (title) }
                div.scores-list {
                    @for (idx, item) in rankings.iter().enumerate() {
                        div.score-item {
                            span.score-rank { "#" (idx + 1) }
                            span.score-series title=(item.name) { (item.name) }
                            span.score-metrics {
                                span.score-value { (format!("{:.2}x", item.score)) }
                                span.score-runtime { (format_time_ns(item.total_runtime)) }
                            }
                        }
                    }
                }
                div.scores-explanation { (explanation) }
            }
        },
        _ => html! {},
    }
}

fn format_time_ns(ns: f64) -> String {
    let abs = ns.abs();
    if abs >= 1_000_000_000.0 {
        format!("{:.2} s", ns / 1_000_000_000.0)
    } else if abs >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{:.2} us", ns / 1_000.0)
    } else {
        format!("{ns:.0} ns")
    }
}

/// Filter dropdown rendered inside the sticky header. The trigger button
/// shows an icon + active-count badge; clicking it opens a panel with two
/// rows of toggle chips that hide/show series by `engine` or `format`
/// across every chart on the page.
///
/// Chip toggle semantics (driven by `chart-init.js`):
/// - The chip's active state mirrors the visibility of that engine/format.
///   With every chip in a row active, no filter is applied for that
///   dimension. Click any chip to toggle it independently.
/// - The "all" chip is a one-shot reset: clicking it forces every chip in
///   that row back to active.
///
/// Chip universes are sourced from [`api::collect_filter_universe`] so a new
/// engine or format showing up in ingest automatically grows the panel;
/// nothing is hard-coded.
fn filter_dropdown(
    universe: &api::FilterUniverse,
    filter: &FilterState,
    active_count: usize,
) -> Markup {
    html! {
        div.filter-dropdown data-role="global-filter-bar" {
            button.control-btn.filter-trigger
                type="button"
                data-role="filter-trigger"
                aria-haspopup="true"
                aria-expanded="false" {
                (filter_icon())
                span { "Filters" }
                @if active_count > 0 {
                    span.filter-badge data-role="filter-badge" { (active_count) }
                }
            }
            div.filter-panel data-role="filter-panel" hidden {
                (filter_row("Engine", "engine", &universe.engines, &filter.engines))
                (filter_row("Format", "format", &universe.formats, &filter.formats))
            }
        }
    }
}

/// Render one row of chips inside the filter panel. `active_list` is empty
/// when no filter is applied for this dimension — every chip renders active.
/// When non-empty, only chips whose value is in the list render active.
fn filter_row(label: &str, dim: &str, universe: &[String], active_list: &[String]) -> Markup {
    let dim_filtered = !active_list.is_empty();
    html! {
        div.global-filter-row {
            span.global-filter-label { (label) }
            button.filter-chip.filter-chip--all
                type="button"
                data-filter=(dim)
                data-value="*"
                aria-pressed="false" {
                "all"
            }
            @for value in universe {
                @let active = !dim_filtered || active_list.iter().any(|v| v == value);
                button.filter-chip
                    type="button"
                    data-filter=(dim)
                    data-value=(value)
                    .filter-chip--active[active]
                    aria-pressed=(active) {
                    (value)
                }
            }
        }
    }
}

/// Embed the active filter state as a small JSON payload the client picks up
/// on hydration. Cheaper to read than re-parsing `location.search` and
/// guarantees the server- and client-side decoders agree.
fn filter_state_script(filter: &FilterState) -> Markup {
    let json = serde_json::to_string(filter).unwrap_or_else(|_| "{}".into());
    html! {
        script id="bench-filter-state" type="application/json" {
            (PreEscaped(escape_json_for_script(&json)))
        }
    }
}

/// Render the per-chart toolbar. `idx` namespaces input ids so multiple
/// charts on the same page don't collide on `<input id="...">`.
///
/// All buttons are `<button type="button">` (not `<a>`): this toolbar does
/// not navigate or rewrite the URL, it manipulates Chart.js state in place.
fn per_chart_toolbar(idx: usize) -> Markup {
    let slider_id = format!("scope-slider-{idx}");
    html! {
        div.toolbar.toolbar--card aria-label="Chart controls" {
            div.toolbar-group role="group" aria-label="Visible commits" {
                span.toolbar-label { "Show" }
                input id=(slider_id).toolbar-slider type="range"
                    min="5" max="1000" step="5" value="100"
                    data-role="scope-slider"
                    aria-label="Custom commit window";
            }
            div.toolbar-group role="group" aria-label="Y-axis scale" {
                span.toolbar-label { "Y" }
                button.toolbar-btn.toolbar-btn--active type="button" data-y="linear" { "linear" }
                button.toolbar-btn type="button" data-y="log" { "log" }
            }
        }
    }
}

/// Render the per-chart range scrollbar strip. A thin track that spans the
/// full chart width and shows which slice of the fetched commit history is
/// currently visible. `chart-init.js` hydrates the strip on chart construction
/// and wires bidirectional drag/resize to the chart's pan/zoom state.
fn range_strip(idx: usize) -> Markup {
    html! {
        div.chart-range-strip data-chart-index=(idx)
            data-role="range-strip"
            aria-label="Visible commit range"
            role="slider" {
            div.chart-range-strip-track {
                div.chart-range-strip-window data-role="range-window" {
                    span.chart-range-strip-handle.chart-range-strip-handle--left
                        data-role="range-handle-left" aria-hidden="true" {}
                    span.chart-range-strip-handle.chart-range-strip-handle--right
                        data-role="range-handle-right" aria-hidden="true" {}
                }
            }
        }
    }
}

/// Make a JSON string safe to embed inside a `<script>` element.
///
/// HTML parsers terminate `<script>` early on a literal `</`. Replacing the
/// `/` with its escaped form keeps the JSON valid while neutering the
/// terminator. `<!--` is similarly neutralised.
fn escape_json_for_script(s: &str) -> String {
    s.replace("</", r"<\/")
        .replace("<!--", r"<\!--")
        .replace("<script", r"<\script")
}

fn error_page(status: StatusCode, message: &str) -> Response {
    let style_href = versioned_asset("/static/style.css");
    let body = html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { (status.as_u16()) " — bench.vortex.dev" }
                (theme_bootstrap_script())
                link rel="stylesheet" href=(style_href);
            }
            body {
                (site_header(None, &FilterState::default()))
                main {
                    p.empty { (message) }
                }
            }
        }
    };
    (status, body).into_response()
}

fn versioned_asset(path: &str) -> String {
    format!("{path}?v={STATIC_ASSET_VERSION}")
}

async fn serve_chart_js() -> impl IntoResponse {
    static_response(CHART_JS, "application/javascript; charset=utf-8")
}

async fn serve_chart_zoom_js() -> impl IntoResponse {
    static_response(CHART_ZOOM_JS, "application/javascript; charset=utf-8")
}

async fn serve_chart_init_js() -> impl IntoResponse {
    static_response(CHART_INIT_JS, "application/javascript; charset=utf-8")
}

async fn serve_style_css() -> impl IntoResponse {
    static_response(STYLE_CSS, "text/css; charset=utf-8")
}

async fn serve_vortex_black_svg() -> impl IntoResponse {
    static_response(VORTEX_BLACK_SVG, "image/svg+xml; charset=utf-8")
}

async fn serve_vortex_white_svg() -> impl IntoResponse {
    static_response(VORTEX_WHITE_SVG, "image/svg+xml; charset=utf-8")
}

fn static_response(bytes: &'static [u8], content_type: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (
                header::CACHE_CONTROL,
                "no-cache, max-age=0, must-revalidate",
            ),
        ],
        bytes,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_json_neutralises_script_terminators() {
        let input = r#"{"x":"</script><script>alert(1)</script>"}"#;
        let out = escape_json_for_script(input);
        assert!(!out.contains("</script"));
        assert!(!out.contains("<script"));
        assert!(out.contains(r"<\/script"));
    }

    #[test]
    fn escape_json_passes_through_safe_strings() {
        let s = r#"{"a":1,"b":"hello"}"#;
        assert_eq!(escape_json_for_script(s), s);
    }

    #[test]
    fn fetch_window_default_is_max() {
        let ui = UiQuery::default();
        match ui.fetch_window() {
            CommitWindow::Last(n) => assert_eq!(n.get(), PER_CHART_FETCH_N),
            CommitWindow::All => panic!("default should be Last(N)"),
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
