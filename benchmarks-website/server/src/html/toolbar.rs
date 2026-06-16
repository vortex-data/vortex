// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-chart toolbar markup: the scope slider + Y-axis switch above each
//! chart, and the range scrollbar strip below it.

use maud::Markup;
use maud::html;

/// Render the per-chart toolbar. `idx` namespaces input ids so multiple
/// charts on the same page don't collide on `<input id="...">`.
///
/// All buttons are `<button type="button">` (not `<a>`): this toolbar does
/// not navigate or rewrite the URL, it manipulates Chart.js state in place.
pub(super) fn per_chart_toolbar(idx: usize) -> Markup {
    let slider_id = format!("scope-slider-{idx}");
    html! {
        div.toolbar.toolbar--card aria-label="Chart controls" {
            div.toolbar-group role="group" aria-label="Visible commits" {
                span.toolbar-label { "Show" }
                // `max` and `step` are placeholders — `chart-init.js` resets
                // them after constructing the chart so the slider tracks the
                // actual loaded commit count, not the initial markup.
                input id=(slider_id).toolbar-slider type="range"
                    min="5" max="100" step="1" value="100"
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
pub(super) fn range_strip(idx: usize) -> Markup {
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
