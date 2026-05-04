// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Page chrome: the outer `<html>` shell, sticky header, theme bootstrap,
//! SVG icon helpers, error page.

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use maud::DOCTYPE;
use maud::Markup;
use maud::PreEscaped;
use maud::html;

use super::FilterState;
use super::filter::filter_dropdown;
use super::filter::filter_state_script;
use super::static_assets::versioned_asset;
use crate::api;

/// Which scripts the page wants pulled in.
pub(super) enum PageScripts {
    /// Empty database — skip Chart.js entirely.
    Empty,
    /// Any page with at least one chart-card. Pulls Chart.js + zoom plugin.
    Chart,
}

/// Render the full HTML page wrapper around `body`. The header carries the
/// global filter dropdown when `universe` is non-empty.
pub(super) fn render_page(
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
    let black_logo = versioned_asset("/Vortex_Black_NoBG.png");
    let white_logo = versioned_asset("/Vortex_White_NoBG.png");
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

/// Funnel icon used by the global filter dropdown trigger.
pub(super) fn filter_icon() -> Markup {
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

/// Render an error page with the same chrome as the main pages.
pub(super) fn error_page(status: StatusCode, message: &str) -> Response {
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

/// Make a JSON string safe to embed inside a `<script>` element.
///
/// HTML parsers terminate `<script>` early on a literal `</`. Replacing the
/// `/` with its escaped form keeps the JSON valid while neutering the
/// terminator. `<!--` is similarly neutralised.
pub(super) fn escape_json_for_script(s: &str) -> String {
    s.replace("</", r"<\/")
        .replace("<!--", r"<\!--")
        .replace("<script", r"<\script")
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
