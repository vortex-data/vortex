// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chart and group permalink page bodies.
//!
//! Both pages render a single chart card (or grid of cards) — the same
//! per-chart toolbar / canvas / range-strip shape the landing page renders
//! per-group, just without the `<details>` wrapper.

use maud::Markup;
use maud::PreEscaped;
use maud::html;

use super::landing::downsample_badge_slot;
use super::landing::group_description_icon;
use super::render::escape_json_for_script;
use super::summary::summary_markup;
use super::toolbar::per_chart_toolbar;
use super::toolbar::range_strip;
use crate::api::ChartResponse;
use crate::api::GroupChartsResponse;

/// Body for `/chart/{slug}`: a single chart-card with the payload inlined.
pub(super) fn chart_body(chart: &ChartResponse, slug: &str, payload_json: &str) -> Markup {
    let series_count = chart.series.len();
    let commit_count = chart.commits.len();
    html! {
        p.chart-meta {
            "unit: " code { (chart.unit_kind.label()) }
            " · "
            (series_count) " series · "
            (commit_count) " commit" @if commit_count != 1 { "s" }
            (downsample_badge_slot())
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

/// Body for `/group/{slug}`: every chart in the group with payloads inlined.
pub(super) fn group_body(group: &GroupChartsResponse) -> Markup {
    let chart_count = group.charts.len();
    html! {
        p.chart-meta {
            (chart_count) " chart" @if chart_count != 1 { "s" }
            (group_description_icon(group.description.as_deref()))
        }
        (summary_markup(group.summary.as_ref()))
        div.chart-grid {
            @for (i, item) in group.charts.iter().enumerate() {
                @let permalink = format!("/chart/{}", item.slug);
                section.chart-card data-chart-index=(i) data-chart-slug=(item.slug) {
                    h3.chart-card-title {
                        a href=(permalink) { (item.name) }
                        (downsample_badge_slot())
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
