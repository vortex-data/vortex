// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Global filter dropdown markup + the on-page filter-state JSON bridge.

use maud::Markup;
use maud::PreEscaped;
use maud::html;

use super::FilterState;
use super::render::escape_json_for_script;
use super::render::filter_icon;
use crate::api;

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
pub(super) fn filter_dropdown(
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
pub(super) fn filter_state_script(filter: &FilterState) -> Markup {
    let json = serde_json::to_string(filter).unwrap_or_else(|_| "{}".into());
    html! {
        script id="bench-filter-state" type="application/json" {
            (PreEscaped(escape_json_for_script(&json)))
        }
    }
}
