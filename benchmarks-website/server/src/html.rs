// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! HTML routes for the bench.vortex.dev v3 web UI.
//!
//! Three pages:
//! - `GET /` — landing page listing every group + chart derived from the
//!   current data. Each group name links to `/group/{slug}` and each chart
//!   link goes to `/chart/{slug}`.
//! - `GET /chart/{slug}` — single Chart.js line chart, payload fetched
//!   server-side and embedded inline as a JSON `<script>` block so there is
//!   no client-side round-trip after page load.
//! - `GET /group/{slug}` — every chart in one group on a single page. Each
//!   chart's payload is embedded inline; chart construction itself is
//!   deferred until the canvas scrolls into view (mobile-friendly + cheap
//!   for big groups like the 22 TPC-H queries).
//!
//! All three pages share the same toolbar: scope (`?n=`), Y-axis (`?y=`),
//! mode (`?mode=`), and hidden series (`?hidden=`). The URL query string is
//! the source of truth for state; clicking through the toolbar is just plain
//! `<a>` navigation, while the JS rewrites the URL via
//! `history.replaceState` for client-only changes (legend toggles).
//!
//! Slugs are opaque strings the server received from `/api/groups`; the
//! handler echoes them straight into [`crate::slug::ChartKey::from_slug`]
//! (or [`crate::slug::GroupKey::from_slug`]) without parsing.
//!
//! Static assets (Chart.js + CSS + the small hydration script) are served
//! from `/static/...` via [`include_bytes!`] so the binary is fully
//! self-contained.

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
use crate::app::AppState;
use crate::db;
use crate::slug::ChartKey;
use crate::slug::GroupKey;

/// Default commit window for the landing page. Smaller than the default for
/// `/chart/{slug}` and `/group/{slug}` (100) because the landing page renders
/// every chart inline — a smaller default keeps the cold payload cheap, and
/// users can ask for more data via the `?n=` toolbar.
const LANDING_DEFAULT_N: u32 = 50;

const CHART_JS: &[u8] = include_bytes!("../static/chart.umd.js");
const CHART_INIT_JS: &[u8] = include_bytes!("../static/chart-init.js");
const STYLE_CSS: &[u8] = include_bytes!("../static/style.css");

/// HTML routes mounted under `/`.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(landing))
        .route("/chart/{slug}", get(chart_page))
        .route("/group/{slug}", get(group_page))
        .route("/static/chart.umd.js", get(serve_chart_js))
        .route("/static/chart-init.js", get(serve_chart_init_js))
        .route("/static/style.css", get(serve_style_css))
}

/// Toolbar/UI state parsed from the query string.
///
/// Server only consumes `n`; the rest are echoed into the rendered toolbar
/// and the URL the JS rewrites for legend toggles. Storing them as
/// strings keeps the round-trip lossless for unknown values.
#[derive(Debug, Default, Deserialize)]
pub struct UiQuery {
    pub n: Option<String>,
    pub y: Option<String>,
    pub mode: Option<String>,
    pub hidden: Option<String>,
}

impl UiQuery {
    fn window(&self) -> CommitWindow {
        CommitWindow::parse(self.n.as_deref())
    }

    /// Like [`Self::window`] but falls back to `default_n` when `?n=` is
    /// unset rather than to [`CommitWindow::default`]. Used by the landing
    /// page which has a different default (50) from the per-chart routes.
    fn window_or_default(&self, default_n: u32) -> CommitWindow {
        match self.n.as_deref() {
            Some(_) => self.window(),
            None => CommitWindow::Last(NonZeroU32::new(default_n).expect("non-zero default")),
        }
    }

    /// `?y=linear|log`, default `linear`.
    fn y_axis(&self) -> &'static str {
        match self.y.as_deref() {
            Some("log") => "log",
            _ => "linear",
        }
    }

    /// `?mode=abs|rel`, default `abs`. Unknown values fall through to `abs`.
    fn mode(&self) -> &'static str {
        match self.mode.as_deref() {
            Some("rel") => "rel",
            _ => "abs",
        }
    }

    /// Build a query string with the given override applied. Only retains
    /// non-empty / non-default values so URLs stay short.
    fn with_override(&self, key: &str, value: &str) -> String {
        let mut pairs: Vec<(&str, String)> = Vec::new();
        let mut add = |k: &'static str, v: Option<String>| {
            if let Some(v) = v
                && !v.is_empty()
            {
                pairs.push((k, v));
            }
        };
        let n = if key == "n" {
            Some(value.to_string())
        } else {
            self.n.clone()
        };
        let y = if key == "y" {
            Some(value.to_string())
        } else {
            self.y.clone()
        };
        let mode = if key == "mode" {
            Some(value.to_string())
        } else {
            self.mode.clone()
        };
        let hidden = if key == "hidden" {
            Some(value.to_string())
        } else {
            self.hidden.clone()
        };
        add("n", n);
        add("y", y);
        add("mode", mode);
        add("hidden", hidden);
        if pairs.is_empty() {
            String::new()
        } else {
            let body: Vec<String> = pairs
                .into_iter()
                .map(|(k, v)| format!("{k}={}", urlencode(&v)))
                .collect();
            format!("?{}", body.join("&"))
        }
    }
}

/// Minimal URL-encoder for query string values. Only chars that need escaping
/// inside an `application/x-www-form-urlencoded` value are touched; the
/// alphanumeric plus a few path-safe symbols pass through verbatim. We avoid
/// pulling in a crate for this — values are short (`100`, `log`, `rel`,
/// `engine:format|engine:format`). `|` is the `?hidden=` series delimiter
/// (see `chart-init.js`); kept unescaped so URLs stay readable in the bar.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b':' | b'|' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

async fn landing(State(state): State<AppState>, Query(ui): Query<UiQuery>) -> Response {
    // Landing-only default: 50 commits. Cheap by default; users opt into
    // bigger windows via the toolbar.
    let window = ui.window_or_default(LANDING_DEFAULT_N);
    // NOTE: payload size on the landing page is the next thing to watch if
    // chart counts grow — every chart's series is inlined as a JSON
    // `<script>` block, so the cold HTML grows linearly in (groups × charts ×
    // series × commits). If this gets fat, switch to lazy-fetch on
    // intersection (the `data-chart-slug` plumbing is already in place).
    let groups_with_charts = match db::run_blocking(&state.db, move |conn| {
        collect_landing_groups(conn, &window)
    })
    .await
    {
        Ok(g) => g,
        Err(err) => {
            tracing::error!(error = ?err, "landing: collect_landing_groups failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };
    let subtitle = format!(
        "Vortex benchmarks (v3 alpha) · {}",
        toolbar_subtitle_suffix_with_default(&ui, LANDING_DEFAULT_N),
    );
    let scripts = if groups_with_charts.is_empty() {
        PageScripts::LandingEmpty
    } else {
        PageScripts::Chart
    };
    render_page(
        "bench.vortex.dev",
        &subtitle,
        landing_body(&groups_with_charts, &ui, LANDING_DEFAULT_N),
        scripts,
        LANDING_DEFAULT_N,
    )
    .into_response()
}

/// Group + every chart inside it, used by the landing-page render.
struct LandingGroup {
    name: String,
    /// Slug for the `/group/{slug}` permalink.
    slug: String,
    charts: Vec<NamedChartResponse>,
}

/// Collect all groups + every chart inside each group for the landing page.
///
/// Returns groups in the same order as [`api::collect_groups`]; groups whose
/// charts all return empty for the current `window` are dropped so the page
/// doesn't render visually-empty sections.
fn collect_landing_groups(conn: &Connection, window: &CommitWindow) -> Result<Vec<LandingGroup>> {
    let groups = api::collect_groups(conn)?;
    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let mut charts = Vec::with_capacity(group.charts.len());
        for link in group.charts {
            let key = ChartKey::from_slug(&link.slug)?;
            let Some(chart) = api::chart_payload(conn, &key, window)? else {
                continue;
            };
            charts.push(NamedChartResponse {
                name: link.name,
                slug: link.slug,
                chart,
            });
        }
        if charts.is_empty() {
            continue;
        }
        out.push(LandingGroup {
            name: group.name,
            slug: group.slug,
            charts,
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

    let window = ui.window();
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
    let subtitle = format!("{} · {}", chart.display_name, toolbar_subtitle_suffix(&ui));
    render_page(
        &title,
        &subtitle,
        chart_body(&chart, &slug, &payload_json, &ui),
        PageScripts::Chart,
        api::DEFAULT_COMMIT_WINDOW,
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
    let window = ui.window();
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
    let subtitle = format!("{} · {}", group.name, toolbar_subtitle_suffix(&ui));
    render_page(
        &title,
        &subtitle,
        group_body(&group, &ui),
        PageScripts::Chart,
        api::DEFAULT_COMMIT_WINDOW,
    )
    .into_response()
}

/// Which scripts the page wants pulled in.
enum PageScripts {
    /// Landing pages without any charts (empty database). Skips Chart.js.
    LandingEmpty,
    Chart,
}

fn render_page(
    title: &str,
    header_subtitle: &str,
    body: Markup,
    scripts: PageScripts,
    default_n: u32,
) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { (title) }
                link rel="stylesheet" href="/static/style.css";
            }
            body {
                header.page-header {
                    h1 { a href="/" { "bench.vortex.dev" } }
                    p.subtitle { (header_subtitle) }
                }
                main data-default-n=(default_n) { (body) }
                @match scripts {
                    PageScripts::LandingEmpty => {
                        script src="/static/chart-init.js" defer {}
                    },
                    PageScripts::Chart => {
                        script src="/static/chart.umd.js" defer {}
                        script src="/static/chart-init.js" defer {}
                    },
                }
            }
        }
    }
}

fn landing_body(groups: &[LandingGroup], ui: &UiQuery, default_n: u32) -> Markup {
    if groups.is_empty() {
        return html! {
            p.empty { "No data ingested yet." }
        };
    }
    let total_charts: usize = groups.iter().map(|g| g.charts.len()).sum();
    // Index every chart globally so `<canvas data-chart-index="N">` and
    // `<script id="chart-data-N">` agree across groups.
    let mut idx_iter = 0usize..total_charts;
    html! {
        (toolbar(ui, default_n))
        div.landing-search {
            input #group-search type="search" placeholder="Filter groups…"
                autocomplete="off" spellcheck="false";
        }
        @for group in groups {
            section.group data-group-name=(group.name) {
                h2 {
                    a.group-link
                        href={ "/group/" (group.slug) (ui_query_string(ui)) }
                        { (group.name) }
                }
                div.chart-grid {
                    @for item in &group.charts {
                        (inline_chart_card(item, idx_iter.next().expect("indices match charts"), ui))
                    }
                }
            }
        }
        noscript {
            p.no-script { "JavaScript is required to render the charts." }
        }
    }
}

fn inline_chart_card(item: &NamedChartResponse, idx: usize, ui: &UiQuery) -> Markup {
    let payload_json = serde_json::to_string(&item.chart).expect("ChartResponse serialises");
    let permalink = format!("/chart/{}", item.slug);
    html! {
        section.chart-card data-chart-index=(idx) data-chart-slug=(item.slug) {
            h3.chart-card-title {
                a href={ (permalink) (ui_query_string(ui)) }
                    data-permalink=(permalink) { (item.name) }
            }
            div.chart-tooltip-host {}
            div.chart-wrap {
                canvas data-chart-index=(idx) {}
            }
            script id={ "chart-data-" (idx) } type="application/json" {
                (PreEscaped(escape_json_for_script(&payload_json)))
            }
        }
    }
}

fn chart_body(chart: &ChartResponse, slug: &str, payload_json: &str, ui: &UiQuery) -> Markup {
    let series_count = chart.series.len();
    let commit_count = chart.commits.len();
    html! {
        (toolbar(ui, api::DEFAULT_COMMIT_WINDOW))
        p.chart-meta {
            "unit: " code { (chart.unit) }
            " · "
            (series_count) " series · "
            (commit_count) " commit" @if commit_count != 1 { "s" }
        }
        section.chart-card data-chart-index="0" data-chart-slug=(slug) {
            div.chart-tooltip-host {}
            div.chart-wrap {
                canvas data-chart-index="0" {}
            }
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

fn group_body(group: &GroupChartsResponse, ui: &UiQuery) -> Markup {
    let chart_count = group.charts.len();
    html! {
        (toolbar(ui, api::DEFAULT_COMMIT_WINDOW))
        p.chart-meta {
            (chart_count) " chart" @if chart_count != 1 { "s" }
        }
        div.chart-grid {
            @for (i, item) in group.charts.iter().enumerate() {
                @let permalink = format!("/chart/{}", item.slug);
                section.chart-card data-chart-index=(i) data-chart-slug=(item.slug) {
                    h3.chart-card-title {
                        a href={ (permalink) (ui_query_string(ui)) }
                            data-permalink=(permalink) { (item.name) }
                    }
                    div.chart-tooltip-host {}
                    div.chart-wrap {
                        canvas data-chart-index=(i) {}
                    }
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

/// Fragment that captures the current toolbar state as a query string,
/// preserved when navigating between pages.
fn ui_query_string(ui: &UiQuery) -> String {
    ui.with_override("__noop", "")
}

/// Render the chart-page toolbar.
///
/// `default_n` is the route's commit-window default: 100 for `/chart` and
/// `/group`, 50 for `/`. When the URL has no `?n=` it determines which
/// `Scope` button is highlighted as active.
fn toolbar(ui: &UiQuery, default_n: u32) -> Markup {
    let active_scope = ui.window_or_default(default_n).url_value();
    let active_y = ui.y_axis();
    let active_mode = ui.mode();
    html! {
        nav.toolbar aria-label="Chart controls" {
            div.toolbar-group role="group" aria-label="Commit window" {
                span.toolbar-label { "Scope" }
                @for opt in ["25", "50", "100", "250", "all"] {
                    a.toolbar-btn.toolbar-btn--active[opt == active_scope]
                        href=(ui.with_override("n", opt))
                        data-scope=(opt) { (opt) }
                }
                input #scope-slider type="range" min="5" max="500" step="5"
                    value=(slider_value(active_scope.as_str()))
                    aria-label="Custom commit window";
                span #scope-slider-label.toolbar-slider-label { (active_scope) }
            }
            div.toolbar-group role="group" aria-label="Y-axis scale" {
                span.toolbar-label { "Y" }
                a.toolbar-btn.toolbar-btn--active[active_y == "linear"]
                    href=(ui.with_override("y", "linear"))
                    data-y="linear" { "linear" }
                a.toolbar-btn.toolbar-btn--active[active_y == "log"]
                    href=(ui.with_override("y", "log"))
                    data-y="log" { "log" }
            }
            div.toolbar-group role="group" aria-label="Display mode" {
                span.toolbar-label { "Mode" }
                a.toolbar-btn.toolbar-btn--active[active_mode == "abs"]
                    href=(ui.with_override("mode", "abs"))
                    data-mode="abs" { "absolute" }
                a.toolbar-btn.toolbar-btn--active[active_mode == "rel"]
                    href=(ui.with_override("mode", "rel"))
                    data-mode="rel" { "% of baseline" }
            }
        }
    }
}

/// Best-effort default value for the slider when the active scope is
/// non-numeric (e.g. `all`).
fn slider_value(scope: &str) -> String {
    scope
        .parse::<u32>()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "100".into())
}

/// Subtitle suffix that mirrors active toolbar state, e.g.
/// `last 100 commits · log · rel`.
fn toolbar_subtitle_suffix(ui: &UiQuery) -> String {
    toolbar_subtitle_suffix_with_default(ui, api::DEFAULT_COMMIT_WINDOW)
}

/// Like [`toolbar_subtitle_suffix`] but uses `default_n` when the URL has no
/// `?n=`. The landing page passes its smaller default (50) here so the
/// subtitle reads "last 50 commits" rather than the global default of 100.
fn toolbar_subtitle_suffix_with_default(ui: &UiQuery, default_n: u32) -> String {
    let scope = match ui.window_or_default(default_n) {
        CommitWindow::All => "all commits".to_string(),
        CommitWindow::Last(n) => format!("last {} commits", n.get()),
    };
    let mut bits = vec![scope];
    if ui.y_axis() == "log" {
        bits.push("log".into());
    }
    if ui.mode() != "abs" {
        bits.push(ui.mode().into());
    }
    bits.join(" · ")
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
    let body = html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                title { (status.as_u16()) " — bench.vortex.dev" }
                link rel="stylesheet" href="/static/style.css";
            }
            body {
                header.page-header {
                    h1 { a href="/" { "bench.vortex.dev" } }
                    p.subtitle { (status.as_u16()) " " (status.canonical_reason().unwrap_or("")) }
                }
                main {
                    p.empty { (message) }
                }
            }
        }
    };
    (status, body).into_response()
}

async fn serve_chart_js() -> impl IntoResponse {
    static_response(CHART_JS, "application/javascript; charset=utf-8")
}

async fn serve_chart_init_js() -> impl IntoResponse {
    static_response(CHART_INIT_JS, "application/javascript; charset=utf-8")
}

async fn serve_style_css() -> impl IntoResponse {
    static_response(STYLE_CSS, "text/css; charset=utf-8")
}

fn static_response(bytes: &'static [u8], content_type: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "public, max-age=3600"),
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
    fn url_encode_keeps_safe_chars() {
        assert_eq!(urlencode("engine:format|parquet"), "engine:format|parquet");
        // `,` is no longer in the allowlist — it gets percent-encoded.
        assert_eq!(urlencode("a,b"), "a%2Cb");
        assert_eq!(urlencode("hello world"), "hello%20world");
    }

    #[test]
    fn ui_query_with_override_round_trips() {
        let ui = UiQuery {
            n: Some("50".into()),
            y: Some("log".into()),
            mode: None,
            hidden: None,
        };
        let qs = ui.with_override("mode", "rel");
        assert!(qs.contains("n=50"));
        assert!(qs.contains("y=log"));
        assert!(qs.contains("mode=rel"));
    }

    #[test]
    fn ui_query_with_override_drops_empty() {
        let ui = UiQuery::default();
        // Default with no override produces empty string.
        assert_eq!(ui.with_override("__noop", ""), "");
    }

    #[test]
    fn toolbar_subtitle_includes_active_state() {
        let ui = UiQuery {
            n: Some("50".into()),
            y: Some("log".into()),
            mode: Some("rel".into()),
            hidden: None,
        };
        let s = toolbar_subtitle_suffix(&ui);
        assert!(s.contains("last 50 commits"));
        assert!(s.contains("log"));
        assert!(s.contains("rel"));
    }

    #[test]
    fn ui_query_with_override_preserves_pipe_delimited_hidden() {
        // `?hidden=` uses `|` as its delimiter (see chart-init.js). A
        // permalink that arrives at the server with multiple hidden series
        // must round-trip through `with_override` without the pipe being
        // percent-encoded — that pins server and client agreement on the
        // wire shape.
        let ui = UiQuery {
            n: None,
            y: None,
            mode: None,
            hidden: Some("a:b|c:d".into()),
        };
        let qs = ui.with_override("__noop", "");
        assert!(
            qs.contains("hidden=a:b|c:d"),
            "expected literal pipe in {qs}",
        );
        assert!(!qs.contains("%7C"), "pipe should not be percent-encoded");
    }
}
