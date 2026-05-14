// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Landing-page body rendering.
//!
//! Every group is wrapped in a collapsed `<details>`; the first group's
//! chart-card shells are hydrated from versioned group shard artifacts on
//! first intent/open.

use maud::Markup;
use maud::html;

use super::render::filter_icon;
use super::summary::summary_markup;
use super::toolbar::per_chart_toolbar;
use super::toolbar::range_strip;
use crate::api;
use crate::api::Summary;

/// One group's worth of data for the landing page.
///
/// Every disclosure renders closed by default on the landing page. Chart
/// payloads are fetched from versioned materialized group shards, so the
/// cold HTML stays metadata-only while first-open hydration is served from
/// memory on the server.
pub(super) struct LandingGroup {
    /// Display name rendered in the disclosure header.
    pub(super) name: String,
    /// Slug for `/api/group/{slug}` exposed as stable group metadata.
    pub(super) slug: String,
    /// Active materialized read generation.
    pub(super) generation: String,
    /// Number of latest-100 shard artifacts available for this group.
    pub(super) shard_count: usize,
    /// URL prefix ending in `/shards/`; JS appends the shard index.
    pub(super) shard_prefix: String,
    /// Whether the group's `<details>` should render open initially.
    pub(super) open: bool,
    /// Optional editorial blurb rendered as a hover tooltip on the
    /// disclosure title's info-icon.
    pub(super) description: Option<String>,
    /// Optional v2-compatible summary card rendered above the chart grid.
    pub(super) summary: Option<Summary>,
    /// Chart links for every chart in the group. Always present — we need
    /// the slugs server-side so the chart-card shell can carry
    /// `data-chart-slug` for the lazy fetch.
    pub(super) chart_links: Vec<api::ChartLink>,
}

/// Render the landing-page body — one `<section>` per group, each wrapping a
/// `<details>` disclosure. The `chart-data-N` script ids are globally
/// indexed so `chart-init.js` can find every payload by integer.
pub(super) fn landing_body(groups: &[LandingGroup], universe: &api::FilterUniverse) -> Markup {
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
        @for group in groups.iter() {
            section.group-details
                data-group-name=(group.name)
                data-group-slug=(group.slug)
                data-artifact-generation=(group.generation)
                data-group-shard-count=(group.shard_count)
                data-group-shard-prefix=(group.shard_prefix) {
                details.group-disclosure open[group.open] {
                    summary.group-summary {
                        span.group-summary-row {
                            span.group-name { (group.name) }
                            (group_description_icon(group.description.as_deref()))
                            span.group-count {
                                (group.chart_links.len()) " chart" @if group.chart_links.len() != 1 { "s" }
                            }
                        }
                    }
                }
                (summary_markup(group.summary.as_ref()))
                (per_group_toolbar(universe))
                div.chart-grid {
                    @for link in &group.chart_links {
                        @let idx = idx_iter.next().expect("indices match charts");
                        (chart_card(link, idx))
                    }
                }
            }
        }
        noscript {
            p.no-script { "JavaScript is required to render the charts." }
        }
    }
}

/// Render the per-group toolbar that lets the user override the global filter
/// and Y-axis scale across every chart in the group. The toolbar is a sibling
/// of `.chart-grid`; CSS hides it when the enclosing `<details>` is closed,
/// mirroring the rule that hides `.chart-grid` itself.
///
/// Layout: Y-axis buttons on the left, a centered "Filter series" dropdown
/// trigger, and a Reset button on the right. The dropdown panel contains
/// engine and format macro chips (which expand to "toggle every series with
/// this engine/format") plus a series row whose chips are populated by JS as
/// charts in the group hydrate and surface their `payload.series_meta`.
///
/// Resolution layering (driven by `chart-init.js`):
/// - Per-card legend overrides win over everything.
/// - The per-group filter (`hiddenSeries`) hides next.
/// - The global filter hides last.
/// - The Y-axis pass skips charts where the user previously clicked the
///   per-chart Y toolbar (`canvas.__bench_y_user_set`).
fn per_group_toolbar(universe: &api::FilterUniverse) -> Markup {
    html! {
        section.group-toolbar data-role="group-toolbar" {
            div.toolbar-group.group-toolbar-y role="group" aria-label="Group Y-axis scale" {
                span.toolbar-label { "Y" }
                // Linear is the resting default (matches each chart's
                // own default) so it ships highlighted; the JS keeps it
                // lit while the per-group Y is unset or explicitly linear.
                button.toolbar-btn.toolbar-btn--active
                    type="button" data-group-y="linear" { "linear" }
                button.toolbar-btn type="button" data-group-y="log" { "log" }
            }
            div.group-filter-dropdown data-role="group-filter-dropdown" {
                button.control-btn.filter-trigger.group-filter-trigger
                    type="button"
                    data-role="group-filter-trigger"
                    aria-haspopup="true"
                    aria-expanded="false" {
                    (filter_icon())
                    span { "Filter series" }
                }
                div.filter-panel.group-filter-panel data-role="group-filter-panel" hidden {
                    (group_macro_row("Engine", "engine", &universe.engines))
                    (group_macro_row("Format", "format", &universe.formats))
                    div.global-filter-row.group-series-row {
                        span.global-filter-label { "Series" }
                        button.filter-chip.filter-chip--all
                            type="button"
                            data-group-filter="series"
                            data-value="*"
                            aria-pressed="false" {
                            "all"
                        }
                        // Series chips hydrate client-side once a chart in this
                        // group exposes its `payload.series_meta`. Until then
                        // the row only shows the Engine/Format macros above.
                        div.group-series-chips data-role="group-series-chips" {}
                    }
                }
            }
            button.group-toolbar-reset
                type="button"
                data-role="group-toolbar-reset" {
                "Reset group"
            }
        }
    }
}

/// Render an engine/format macro row inside the per-group filter panel. The
/// macro chip click bulk-toggles every known series whose `engine`/`format`
/// matches; the chip's active state reflects "every matching series is
/// currently visible". `data-group-filter` distinguishes these chips from the
/// global filter's `data-filter` ones so the click handler can route them to
/// the per-group state.
fn group_macro_row(label: &str, dim: &str, universe: &[String]) -> Markup {
    html! {
        div.global-filter-row.group-macro-row {
            span.global-filter-label { (label) }
            button.filter-chip.filter-chip--all
                type="button"
                data-group-filter=(dim)
                data-value="*"
                aria-pressed="false" {
                "all"
            }
            @for value in universe {
                button.filter-chip.filter-chip--active
                    type="button"
                    data-group-filter=(dim)
                    data-value=(value)
                    aria-pressed="true" {
                    (value)
                }
            }
        }
    }
}

/// Render the small ⓘ info icon that surfaces the group's editorial
/// description on hover and on focus. The CSS-only tooltip uses a
/// `data-tooltip` attribute so it shows below the icon (see `style.css`'s
/// `.group-info-icon` rule). The icon itself is keyboard-focusable and
/// `aria-label`-ed so the description is reachable via the keyboard and to
/// screen readers.
///
/// Returns an empty markup fragment when `description` is `None` so groups
/// without a canonical blurb (e.g. vector-search groups) render unchanged.
pub(super) fn group_description_icon(description: Option<&str>) -> Markup {
    let Some(text) = description else {
        return html! {};
    };
    html! {
        span.group-info-icon
            tabindex="0"
            role="note"
            aria-label=(text)
            data-tooltip=(text) {
            "ⓘ"
        }
    }
}

/// Render one chart-card shell. The payload arrives later from the group's
/// materialized shard artifact.
fn chart_card(link: &api::ChartLink, idx: usize) -> Markup {
    let permalink = format!("/chart/{}", link.slug);
    html! {
        section.chart-card data-chart-index=(idx) data-chart-slug=(link.slug) {
            h3.chart-card-title {
                a href=(permalink) { (link.name) }
                (downsample_badge_slot())
            }
            (per_chart_toolbar(idx))
            div.chart-tooltip-host {}
            div.chart-wrap {
                canvas data-chart-index=(idx) {}
            }
            (range_strip(idx))
        }
    }
}

/// Empty hidden slot for the LTTB badge. `chart-init.js` flips it on when
/// the *currently visible* commit range exceeds the LTTB threshold and the
/// rendered point count is therefore less than the raw point count in that
/// range.
pub(super) fn downsample_badge_slot() -> Markup {
    html! {
        span.chart-badge.chart-badge--downsampled
            data-role="downsample-badge"
            hidden {}
    }
}
