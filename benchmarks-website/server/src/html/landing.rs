// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Landing-page body rendering.
//!
//! Every group is wrapped in a collapsed `<details>`; the first group's
//! chart-card shells are hydrated from versioned group shard artifacts on
//! first intent/open.

use maud::Markup;
use maud::html;

use super::current::history_section;
use super::render::filter_icon;
use crate::api;
use crate::api::Summary;
use crate::read_model::ReadGeneration;

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
    /// Optional v2-compatible summary card. Retired on Previous Versions in
    /// favour of the synthesis headline chart; kept on the struct until the v2
    /// summary chain is removed.
    #[allow(dead_code)]
    pub(super) summary: Option<Summary>,
    /// Chart links for every chart in the group. Always present - we need
    /// the slugs server-side so the chart-card shell can carry
    /// `data-chart-slug` for the lazy fetch.
    pub(super) chart_links: Vec<api::ChartLink>,
    /// Scale-factor selector for TPC query suites: the largest SF is current,
    /// the rest link to their group pages. Empty for non-TPC and single-SF groups.
    pub(super) scale_pills: Vec<ScalePill>,
    /// Distinct engines that actually have data in this query group's charts
    /// (e.g. `["datafusion", "duckdb"]` for ClickBench/TPC-H, `["duckdb"]` for
    /// statpopgen, `["datafusion"]` for polarsignals). Empty for non-query
    /// families. Drives which per-engine cards the chart grid emits so a
    /// single-engine group doesn't render hidden placeholders that reflow the
    /// layout.
    pub(super) engines: Vec<String>,
}

/// One scale-factor option in a TPC suite's selector.
pub(super) struct ScalePill {
    /// Display label, e.g. `SF10`.
    pub(super) label: String,
    /// `/group/{slug}` target (ignored when `current`).
    pub(super) slug: String,
    /// Whether this is the SF shown by default (the largest available).
    pub(super) current: bool,
}

/// Render the landing-page body - one `<section>` per group, each wrapping a
/// `<details>` disclosure. Each `<canvas>` carries a unique
/// `data-chart-index` integer (used by `chart-init.js` to wire toolbar
/// controls to canvases by id) and a `data-chart-slug` (used to resolve the
/// per-chart payload via the enclosing `<section.group-details>`'s
/// `data-group-shard-prefix`). The landing page emits NO inline JSON
/// payloads - every chart hydrates from a versioned shard artifact on
/// first intent/open. The `chart-data-N` inline-script id is permalink-page
/// only and lives in `chart.rs`.
pub(super) fn landing_body(
    generation: &ReadGeneration,
    groups: &[LandingGroup],
    universe: &api::FilterUniverse,
) -> Markup {
    if groups.is_empty() {
        return html! {
            p.empty { "No data ingested yet." }
        };
    }
    // One robust card per chart, except for chart kinds that carry an
    // incomparable secondary dimension — query charts (`qm.`) split per engine
    // (DataFusion | DuckDB), compression-time charts (`ct.`) split per op
    // (Encode | Decode). Cards are emitted interleaved so the 2-col grid puts
    // the two facets side by side per chart (df-left/duck-right; encode-left/
    // decode-right), and a 1-col breakpoint interleaves them. RA (`rat.`) and
    // compression-size (`cs.`) charts get one card each. Every canvas gets a
    // unique per-page `data-chart-index`.
    struct CardSpec<'a> {
        link: &'a api::ChartLink,
        idx: usize,
        engine: Option<&'static str>,
        op: Option<&'static str>,
    }
    let mut next_idx = 0usize;
    let mut group_cards: Vec<Vec<CardSpec>> = Vec::with_capacity(groups.len());
    for g in groups.iter() {
        // For query groups, the engine split honours `g.engines` (computed
        // server-side from the data) so a single-engine group like statpopgen
        // (duckdb-only) doesn't emit an empty DataFusion card that would
        // reflow the layout. Fall back to both engines if the engine set is
        // empty (defensive — happens only when the first chart payload was
        // unavailable at shell render).
        let mut cards = Vec::new();
        for link in &g.chart_links {
            let splits: Vec<(Option<&'static str>, Option<&'static str>)> =
                if link.slug.starts_with("qm.") {
                    let mut v: Vec<(Option<&'static str>, Option<&'static str>)> = Vec::new();
                    for &eng in &["datafusion", "duckdb"] {
                        if g.engines.iter().any(|e| e == eng) {
                            v.push((Some(eng), None));
                        }
                    }
                    if v.is_empty() {
                        v.push((Some("datafusion"), None));
                        v.push((Some("duckdb"), None));
                    }
                    v
                } else if link.slug.starts_with("ct.") {
                    vec![(None, Some("encode")), (None, Some("decode"))]
                } else {
                    vec![(None, None)]
                };
            for (engine, op) in &splits {
                cards.push(CardSpec {
                    link,
                    idx: next_idx,
                    engine: *engine,
                    op: *op,
                });
                next_idx += 1;
            }
        }
        group_cards.push(cards);
    }
    html! {
        header.current-intro {
            h2.current-headline { "Vortex vs Parquet, across versions." }
            div.methodology {
                p.methodology-text {
                    "Each group's headline plots one number across develop: the "
                    strong { "geometric mean of its per-item Vortex/Parquet ratios" }
                    " at each commit. Expand a group to see the per-item charts beneath — "
                    "same time axis, plotted in "
                    strong { "raw measurement units" }
                    " (ms, MB/s, MiB) instead of ratios. Every line, headline or per-item, "
                    "is a centred rolling median with a p25–p75 ribbon and the raw "
                    "per-commit points faint behind it, so harness spikes and run-to-run "
                    "jitter show up as ribbon width rather than as kinks in the trend. The "
                    "rightmost point of every headline matches the "
                    a href="/latest" { "Latest Commit" }
                    " headline."
                }
            }
        }
        @for (group, cards) in groups.iter().zip(group_cards.iter()) {
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
                (history_section(generation, group))
                (per_group_toolbar(universe))
                div.chart-grid {
                    @for c in cards.iter() {
                        (chart_card(c.link, c.idx, c.engine, c.op))
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
/// Render the scale-factor selector in a TPC group header: the current
/// (largest) SF is a highlighted label, the others link to their group pages.
/// Empty when there's nothing to switch between.
// Retired on Previous Versions: the headline now carries an in-place SF toggle
// (see `current::history_section`) instead of pills that link away to
// per-SF group pages. Kept for potential reuse.
#[allow(dead_code)]
pub(super) fn scale_pills_markup(pills: &[ScalePill]) -> Markup {
    if pills.len() < 2 {
        return html! {};
    }
    html! {
        span.group-scale-pills aria-label="Scale factor" {
            @for pill in pills {
                @if pill.current {
                    span.scale-pill.scale-pill--current { (pill.label) }
                } @else {
                    a.scale-pill href=(format!("/group/{}", pill.slug)) { (pill.label) }
                }
            }
        }
    }
}

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
fn chart_card(link: &api::ChartLink, idx: usize, engine: Option<&str>, op: Option<&str>) -> Markup {
    let permalink = format!("/chart/{}", link.slug);
    // Robust cards label the facet (engine or op) and drop the pan/zoom-only
    // chrome (toolbar, downsample badge, range strip) — they render a static
    // robust band chart instead. The `data-robust` flag drives the JS branch;
    // `data-engine` / `data-op` carry the per-card series filter.
    let suffix = match (engine, op) {
        (Some("datafusion"), _) => Some("DataFusion"),
        (Some("duckdb"), _) => Some("DuckDB"),
        (_, Some("encode")) => Some("Encode"),
        (_, Some("decode")) => Some("Decode"),
        _ => None,
    };
    let title = match suffix {
        Some(s) => format!("{} · {s}", link.name),
        None => link.name.clone(),
    };
    html! {
        section.chart-card data-robust data-chart-index=(idx) data-chart-slug=(link.slug)
            data-engine=[engine] data-op=[op] {
            h3.chart-card-title {
                a href=(permalink) { (title) }
            }
            div.chart-tooltip-host {}
            div.chart-wrap {
                canvas data-chart-index=(idx) {}
            }
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
