// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Landing-page body rendering.
//!
//! Every group is wrapped in a collapsed `<details>`; the first group's
//! chart-card shells are hydrated from versioned group shard artifacts on
//! first intent/open.

use maud::Markup;
use maud::html;

use super::current::history_headline;
use super::current::history_section;
use super::current::sf_toggle_pills;
use super::current::storage_toggle_pills;
use super::current::strip_tpc_parentheticals;
use super::render::chart_controls;
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
    /// (storage, scale-factor) panels for TPC query suites. One pill per real
    /// (storage, sf) combination present in the data; the default-shown pill is
    /// (preferred storage, largest SF). Empty for non-TPC groups and clusters
    /// with only one combination.
    pub(super) scale_pills: Vec<ScalePill>,
    /// Distinct engines that actually have data in this query group's charts
    /// (e.g. `["datafusion", "duckdb"]` for ClickBench/TPC-H, `["duckdb"]` for
    /// statpopgen, `["datafusion"]` for polarsignals). Empty for non-query
    /// families. Drives which per-engine cards the chart grid emits so a
    /// single-engine group doesn't render hidden placeholders that reflow the
    /// layout.
    pub(super) engines: Vec<String>,
}

/// One (storage, scale-factor) option in a TPC suite's selector. Each pill
/// names one real combination present in the data — the UI derives the storage
/// and scale-factor button rows by collecting distinct values across pills.
pub(super) struct ScalePill {
    /// Scale-factor display label, e.g. `SF10`.
    pub(super) sf_label: String,
    /// Scale-factor raw value used as `data-sf`, e.g. `10`.
    pub(super) sf_value: String,
    /// Storage raw value used as `data-storage`, e.g. `nvme` / `s3`. The
    /// display label is derived at render time via `super::storage_label` so it
    /// stays in sync with the canonical-tiers list.
    pub(super) storage_value: String,
    /// Underlying group slug for this (storage, sf) panel.
    pub(super) slug: String,
    /// Underlying group display name (carried for `data-group-name` on the
    /// panel so the JS hydration log treats each panel as a named group).
    pub(super) name: String,
    /// Versioned shard URL prefix for this panel's group; the JS appends the
    /// shard index. The outer disclosure no longer carries a shard prefix for
    /// TPC clusters — each panel hydrates from its own group's artifacts.
    pub(super) shard_prefix: String,
    /// Number of materialized shards for this panel's group.
    pub(super) shard_count: usize,
    /// Chart links for the panel's group, used to render the per-query grid
    /// inside the panel.
    pub(super) chart_links: Vec<api::ChartLink>,
    /// Whether this is the combination shown by default — exactly one pill per
    /// group is current.
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
    // Non-TPC groups carry one flat list of cards; TPC clusters carry one list
    // per (storage, SF) pill so the panel toggle can swap both headline and
    // per-query grid together.
    enum GroupLayout<'a> {
        Flat(Vec<CardSpec<'a>>),
        PerPanel(Vec<Vec<CardSpec<'a>>>),
    }
    // For query groups, the engine split honours `engines` (computed
    // server-side from the data) so a single-engine group like statpopgen
    // (duckdb-only) doesn't emit an empty DataFusion card that would
    // reflow the layout. Falls back to both engines if the engine set is
    // empty (defensive — happens only when the first chart payload was
    // unavailable at shell render).
    fn build_cards<'a>(
        links: &'a [api::ChartLink],
        engines: &[String],
        next_idx: &mut usize,
    ) -> Vec<CardSpec<'a>> {
        let mut cards = Vec::new();
        for link in links {
            let splits: Vec<(Option<&'static str>, Option<&'static str>)> =
                if link.slug.starts_with("qm.") {
                    let mut v: Vec<(Option<&'static str>, Option<&'static str>)> = Vec::new();
                    for &eng in &["datafusion", "duckdb"] {
                        if engines.iter().any(|e| e == eng) {
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
                    idx: *next_idx,
                    engine: *engine,
                    op: *op,
                });
                *next_idx += 1;
            }
        }
        cards
    }
    let mut next_idx = 0usize;
    let mut layouts: Vec<GroupLayout> = Vec::with_capacity(groups.len());
    for g in groups.iter() {
        // Anything with fewer than two (storage, SF) combinations renders
        // flat — there's no toggle to own a dimension, so the heading keeps
        // its parentheticals and the chart-grid is the rep's.
        if g.scale_pills.len() < 2 {
            layouts.push(GroupLayout::Flat(build_cards(
                &g.chart_links,
                &g.engines,
                &mut next_idx,
            )));
        } else {
            let mut panels: Vec<Vec<CardSpec>> = Vec::with_capacity(g.scale_pills.len());
            for pill in &g.scale_pills {
                panels.push(build_cards(&pill.chart_links, &g.engines, &mut next_idx));
            }
            layouts.push(GroupLayout::PerPanel(panels));
        }
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
        @for (group, layout) in groups.iter().zip(layouts.iter()) {
            // For non-TPC groups the outer section carries the (single) shard
            // prefix; for TPC clusters the shard prefix moves to each panel so
            // the toggle swaps both headline and per-query grid in lockstep
            // (panel attrs include `data-group-shard-prefix`, `-count`,
            // `-slug`, `-name`, `-artifact-generation`). The outer section
            // keeps `data-group-slug` for anchoring and `data-artifact-generation`
            // so chart-init.js can still log a generation, but drops the shard
            // prefix to make `groupShardUrl` look at the panels instead.
            @let is_per_panel = matches!(layout, GroupLayout::PerPanel(_));
            section.group-details
                data-group-name=(group.name)
                data-group-slug=(group.slug)
                data-artifact-generation=(group.generation)
                data-group-shard-count=[if is_per_panel { None } else { Some(group.shard_count) }]
                data-group-shard-prefix=[if is_per_panel { None } else { Some(group.shard_prefix.as_str()) }] {
                details.group-disclosure open[group.open] {
                    summary.group-summary {
                        span.group-summary-row {
                            // Per-panel TPC clusters move the storage and SF
                            // parentheticals onto the toggle buttons, so the
                            // disclosure label drops them. Everything else
                            // (including 1-pill TPC stragglers without a real
                            // dimension to switch) keeps the original name.
                            @let display_name = if is_per_panel {
                                strip_tpc_parentheticals(&group.name)
                            } else {
                                group.name.clone()
                            };
                            span.group-name { (display_name) }
                            (group_description_icon(group.description.as_deref()))
                            span.group-count {
                                (group.chart_links.len()) " chart" @if group.chart_links.len() != 1 { "s" }
                            }
                        }
                    }
                }
                @match layout {
                    GroupLayout::Flat(cards) => {
                        (history_section(generation, group))
                        (per_group_toolbar(universe))
                        div.chart-grid {
                            @for c in cards {
                                (chart_card(c.link, c.idx, c.engine, c.op))
                            }
                        }
                    },
                    GroupLayout::PerPanel(panel_cards) => {
                        div.history-fanout {
                            div.history-controls {
                                (storage_toggle_pills(&group.scale_pills))
                                (sf_toggle_pills(&group.scale_pills))
                            }
                            div.history-sf-sets {
                                @for (pill, cards) in group.scale_pills.iter().zip(panel_cards.iter()) {
                                    section.speedup-sf
                                        data-sf=(pill.sf_value)
                                        data-storage=(pill.storage_value)
                                        data-group-slug=(pill.slug)
                                        data-group-name=(pill.name)
                                        data-artifact-generation=(group.generation)
                                        data-group-shard-prefix=(pill.shard_prefix)
                                        data-group-shard-count=(pill.shard_count)
                                        hidden[!pill.current] {
                                        (history_headline(generation, &pill.chart_links))
                                        div.chart-grid {
                                            @for c in cards {
                                                (chart_card(c.link, c.idx, c.engine, c.op))
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        (per_group_toolbar(universe))
                    },
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
                (chart_controls(true))
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
