// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chart and group permalink page bodies.
//!
//! The chart page renders a single chart card with its payload inlined.

use maud::Markup;
use maud::PreEscaped;
use maud::html;

use super::landing::downsample_badge_slot;
use super::render::escape_json_for_script;
use super::toolbar::per_chart_toolbar;
use super::toolbar::range_strip;
use crate::api::ChartResponse;
use crate::api::CommitWindow;

/// Body for `/chart/{slug}`: a single chart-card with the payload inlined.
pub(super) fn chart_body(
    chart: &ChartResponse,
    slug: &str,
    payload_json: &str,
    window: &CommitWindow,
) -> Markup {
    let series_count = chart.series.len();
    let commit_count = chart.commits.len();
    let payload_window = window.url_value();
    html! {
        p.chart-meta {
            "unit: " code { (chart.unit_kind.label()) }
            " · "
            (series_count) " series · "
            (commit_count) " commit" @if commit_count != 1 { "s" }
            (downsample_badge_slot())
        }
        section.chart-card
            data-chart-index="0"
            data-chart-slug=(slug)
            data-payload-window=(payload_window) {
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
