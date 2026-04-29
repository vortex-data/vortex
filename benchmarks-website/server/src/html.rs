// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! HTML routes for the bench.vortex.dev v3 alpha web UI.
//!
//! Two pages:
//! - `GET /` — landing page listing every group + chart derived from the
//!   current data.
//! - `GET /chart/{slug}` — single Chart.js line chart, payload fetched
//!   server-side and embedded inline as a JSON `<script>` block so there is
//!   no client-side round-trip after page load.
//!
//! Slugs are opaque strings the server received from `/api/groups`; the
//! handler echoes them straight into [`crate::slug::ChartKey::from_slug`]
//! without parsing.
//!
//! Static assets (Chart.js + CSS + the small hydration script) are served
//! from `/static/...` via [`include_bytes!`] so the binary is fully
//! self-contained.

use axum::Router;
use axum::extract::Path;
use axum::extract::Query;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use maud::DOCTYPE;
use maud::Markup;
use maud::PreEscaped;
use maud::html;

use crate::api;
use crate::api::ChartQuery;
use crate::api::ChartResponse;
use crate::api::Group;
use crate::app::AppState;
use crate::db;
use crate::downsample::resolve_max_points;
use crate::slug::ChartKey;

const CHART_JS: &[u8] = include_bytes!("../static/chart.umd.js");
const CHART_INIT_JS: &[u8] = include_bytes!("../static/chart-init.js");
const STYLE_CSS: &[u8] = include_bytes!("../static/style.css");

/// HTML routes mounted under `/`.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(landing))
        .route("/chart/{slug}", get(chart_page))
        .route("/static/chart.umd.js", get(serve_chart_js))
        .route("/static/chart-init.js", get(serve_chart_init_js))
        .route("/static/style.css", get(serve_style_css))
}

async fn landing(State(state): State<AppState>) -> Response {
    let groups = match db::run_blocking(&state.db, |conn| api::collect_groups(conn)).await {
        Ok(g) => g,
        Err(err) => {
            tracing::error!(error = ?err, "landing: collect_groups failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };
    render_page(
        "bench.vortex.dev",
        "Vortex benchmarks (v3 alpha)",
        landing_body(&groups),
        PageScripts::None,
    )
    .into_response()
}

async fn chart_page(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Query(params): Query<ChartQuery>,
) -> Response {
    let key = match ChartKey::from_slug(&slug) {
        Ok(key) => key,
        Err(err) => {
            tracing::warn!(error = ?err, slug, "chart_page: invalid slug");
            return error_page(StatusCode::NOT_FOUND, "chart not found").into_response();
        }
    };

    if let Some(n) = params.n.as_deref() {
        if n != "all" && n.parse::<u32>().is_err() {
            return error_page(
                StatusCode::BAD_REQUEST,
                "invalid `n` query parameter (expected `all` or a positive integer)",
            )
            .into_response();
        }
    }
    let max_points = resolve_max_points(params.max_points);
    let result = db::run_blocking(&state.db, move |conn| api::collect_chart(conn, &key)).await;
    let mut chart = match result {
        Ok(Some(c)) => c,
        Ok(None) => return error_page(StatusCode::NOT_FOUND, "chart not found").into_response(),
        Err(err) => {
            tracing::error!(error = ?err, "chart_page: collect_chart failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };
    if let Some(target) = max_points {
        api::downsample_chart(&mut chart, target);
    }

    let payload_json = match serde_json::to_string(&chart) {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(error = ?err, "chart_page: serialize failed");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response();
        }
    };

    let title = format!("{} — bench.vortex.dev", chart.display_name);
    render_page(
        &title,
        &chart.display_name,
        chart_body(&chart, &payload_json),
        PageScripts::Chart,
    )
    .into_response()
}

/// Which scripts the page wants pulled in.
enum PageScripts {
    None,
    Chart,
}

fn render_page(title: &str, header_subtitle: &str, body: Markup, scripts: PageScripts) -> Markup {
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
                main { (body) }
                @match scripts {
                    PageScripts::None => {},
                    PageScripts::Chart => {
                        script src="/static/chart.umd.js" defer {}
                        script src="/static/chart-init.js" defer {}
                    },
                }
            }
        }
    }
}

fn landing_body(groups: &[Group]) -> Markup {
    html! {
        @if groups.is_empty() {
            p.empty { "No data ingested yet." }
        } @else {
            @for group in groups {
                section.group {
                    h2 { (group.name) }
                    ul.charts {
                        @for chart in &group.charts {
                            li {
                                a href={ "/chart/" (chart.slug) } { (chart.name) }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn chart_body(chart: &ChartResponse, payload_json: &str) -> Markup {
    let series_count = chart.series.len();
    let commit_count = chart.commits.len();
    html! {
        p.chart-meta {
            "unit: " code { (chart.unit) }
            " · "
            (series_count) " series · "
            (commit_count) " commit" @if commit_count != 1 { "s" }
        }
        div.chart-wrap {
            canvas id="chart" {}
        }
        // Embedded JSON; rendered as text content so JSON `<` / `>` are HTML-escaped.
        script id="chart-data" type="application/json" { (PreEscaped(escape_json_for_script(payload_json))) }
        noscript {
            p.no-script { "JavaScript is required to render the chart." }
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
}
