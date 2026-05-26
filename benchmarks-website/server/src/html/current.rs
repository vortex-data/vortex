// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Current page body (`/current`): the synthesized Vortex-vs-Parquet view.
//!
//! Every chart here answers one question with one controlled comparison -
//! *holding dataset, engine, operation and scale constant, vary only the
//! format*. The raw, as-collected experiment charts (and their drift over
//! time) live on the Raw-data page.
//!
//! One shape for every group: a **speedup distribution**. For each *item* (a
//! query, a dataset, an access pattern) the chart plots the `B / A` ratio of
//! two formats picked from per-chart dropdowns (default Vortex vs Parquet), so
//! the axis is "how much faster/smaller A is" and the spread is visible. A
//! group splits into one chart per *facet* — the dimension that isn't the
//! format or the item:
//! - query suites → facet is the **engine** (DuckDB and DataFusion never share
//!   a chart, so the comparison is about the format);
//! - compression → facet is the **operation** (encode vs decode);
//! - random access / size → **no facet** (a single chart).
//!
//! The metric verb ("faster" for times, "smaller" for sizes) comes from the
//! unit. All chart data + server-formatted labels are emitted as inline JSON
//! beside a `<canvas>`; `chart-init.js` builds the Chart.js view. Each group
//! section has a stable `id` ([`super::anchor_for`]) so the showcase's links
//! land on - and `:target`-highlight - the right section.

use std::collections::BTreeMap;

use maud::Markup;
use maud::PreEscaped;
use maud::html;
use serde::Serialize;

use super::anchor_for;
use super::landing::LandingGroup;
use super::render::escape_json_for_script;
use super::showcase::format_value;
use super::showcase::latest_value;
use crate::api::ChartLink;
use crate::api::UnitKind;
use crate::read_model::ReadGeneration;

/// The canonical "Vortex" format for Vortex-vs-Parquet comparisons (the
/// heavily-compressed file variant the published numbers use).
const VORTEX_FORMAT: &str = "vortex-file-compressed";
/// The baseline format every comparison races against.
const PARQUET_FORMAT: &str = "parquet";

/// Render the Current body: a lead anchored to the snapshot commit, then one
/// section per group.
pub(super) fn current_body(generation: &ReadGeneration, groups: &[LandingGroup]) -> Markup {
    if groups.is_empty() {
        return html! { p.empty { "No data ingested yet." } };
    }
    html! {
        section.current {
            header.current-intro {
                h2.current-headline { "Vortex vs Parquet, head to head." }
                p.current-lead {
                    "Each chart pits two formats against each other: every bar is one query, "
                    "dataset, or access pattern, plotted as the ratio between them — so you see "
                    "where the win holds and where it doesn't, not just the average. Pick a "
                    "different pair with the dropdowns; the full per-commit history lives under "
                    a href="/raw" { "Previous Versions" }
                    "."
                }
                (snapshot_stamp(generation, groups))
            }
            @for group in groups {
                (speedup_section(generation, group))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Speedup distribution (the synthesis model, used by every group)
// ---------------------------------------------------------------------------

/// One format option for a comparison dropdown.
#[derive(Serialize)]
struct FormatOpt {
    /// Physical format id (`vortex-file-compressed`, `parquet`, …).
    id: String,
    /// Human label shown in the dropdown / axis / headline.
    label: String,
}

/// One item's (query / dataset / pattern) value for every format measured for
/// it under one facet. Keyed by format id; only formats present appear. The
/// client divides the two selected formats to get the per-item ratio.
#[derive(Serialize)]
struct QueryRow {
    query: String,
    /// format id -> raw value (base unit), for the ratio.
    #[serde(rename = "v")]
    values: BTreeMap<String, f64>,
    /// format id -> server-formatted value, for the tooltip.
    #[serde(rename = "d")]
    displays: BTreeMap<String, String>,
}

/// One facet's full per-item/per-format matrix, emitted to the client. The two
/// comparison dropdowns pick any pair of `formats`; the chart shows the `B / A`
/// ratio per item. Defaults reproduce the Vortex-vs-Parquet view.
#[derive(Serialize)]
struct EngineData {
    /// The facet this chart is for: an engine, an operation, or `""` for the
    /// no-facet (single-chart) groups. Rendered as the chart's label.
    facet: String,
    /// Comparison verb from the unit: `"faster"` (times) or `"smaller"` (sizes).
    metric: &'static str,
    formats: Vec<FormatOpt>,
    #[serde(rename = "defaultA")]
    default_a: String,
    #[serde(rename = "defaultB")]
    default_b: String,
    queries: Vec<QueryRow>,
}

/// Comparison verb for a unit. Both times and sizes are lower-is-better, so the
/// `B / A` direction is the same; only the word changes.
fn metric_for(unit: UnitKind) -> &'static str {
    match unit {
        UnitKind::Bytes => "smaller",
        _ => "faster",
    }
}

/// The facet a series belongs to: its engine (query suites), else the operation
/// suffix of a `format:op` name (compression), else `""` (random access, size).
fn facet_of(name: &str, engine: Option<&str>) -> String {
    if let Some(engine) = engine {
        engine.to_string()
    } else if let Some((_, op)) = name.rsplit_once(':') {
        op.to_string()
    } else {
        String::new()
    }
}

/// Sort key for a format in the comparison dropdowns: Vortex variants first,
/// then the baselines.
fn format_order(id: &str) -> usize {
    match id {
        "vortex-file-compressed" => 0,
        "vortex-compact" => 1,
        "parquet" => 2,
        "arrow" => 3,
        "duckdb" => 4,
        "lance" => 5,
        _ => 6,
    }
}

/// Human label for a physical format id.
fn format_label(id: &str) -> &str {
    match id {
        "vortex-file-compressed" => "Vortex",
        "vortex-compact" => "Vortex-compact",
        "parquet" => "Parquet",
        "arrow" => "Arrow",
        "duckdb" => "DuckDB",
        "lance" => "Lance",
        other => other,
    }
}

/// Build the per-facet comparison data for a set of items (chart links): one
/// [`EngineData`] per facet (engine / operation / `""`). Returns the facets and
/// whether any were engine-faceted (so callers can label the natural sort
/// order). A facet needs two distinct formats to be comparable.
fn build_facets(generation: &ReadGeneration, chart_links: &[ChartLink]) -> (Vec<EngineData>, bool) {
    use std::collections::BTreeSet;
    #[derive(Default)]
    struct Acc {
        formats: BTreeSet<String>,
        queries: Vec<QueryRow>,
    }
    let mut per_facet: BTreeMap<String, Acc> = BTreeMap::new();
    let mut unit = UnitKind::TimeNs;
    let mut faceted_by_engine = false;

    for link in chart_links {
        let Some(payload) = generation.chart_payload(&link.slug) else {
            continue;
        };
        unit = payload.unit_kind;
        let mut rows: BTreeMap<String, QueryRow> = BTreeMap::new();
        for (name, tag) in payload.series_meta.iter() {
            let Some(format) = tag.format.as_ref() else {
                continue;
            };
            let Some(v) = payload.series.get(name).and_then(latest_value) else {
                continue;
            };
            if v <= 0.0 {
                continue;
            }
            if tag.engine.is_some() {
                faceted_by_engine = true;
            }
            let facet = facet_of(name, tag.engine.as_deref());
            let row = rows.entry(facet).or_insert_with(|| QueryRow {
                query: link.name.clone(),
                values: BTreeMap::new(),
                displays: BTreeMap::new(),
            });
            row.values.insert(format.clone(), v);
            row.displays.insert(format.clone(), format_value(v, unit));
        }
        for (facet, row) in rows {
            if row.values.is_empty() {
                continue;
            }
            let acc = per_facet.entry(facet).or_default();
            acc.formats.extend(row.values.keys().cloned());
            acc.queries.push(row);
        }
    }

    let metric = metric_for(unit);
    let facets = per_facet
        .into_iter()
        .filter(|(_, acc)| acc.formats.len() >= 2 && !acc.queries.is_empty())
        .map(|(facet, acc)| {
            let mut formats: Vec<String> = acc.formats.into_iter().collect();
            formats.sort_by_key(|f| (format_order(f), f.clone()));
            let default_a = formats
                .iter()
                .find(|f| f.as_str() == VORTEX_FORMAT)
                .or_else(|| formats.iter().find(|f| f.contains("vortex")))
                .or_else(|| formats.first())
                .cloned()
                .unwrap_or_default();
            let default_b = formats
                .iter()
                .find(|f| f.as_str() == PARQUET_FORMAT && f.as_str() != default_a)
                .or_else(|| formats.iter().find(|f| f.as_str() != default_a))
                .cloned()
                .unwrap_or_else(|| default_a.clone());
            let format_opts = formats
                .iter()
                .map(|id| FormatOpt {
                    id: id.clone(),
                    label: format_label(id).to_string(),
                })
                .collect();
            EngineData {
                facet,
                metric,
                formats: format_opts,
                default_a,
                default_b,
                queries: acc.queries,
            }
        })
        .collect();
    (facets, faceted_by_engine)
}

/// Render a group as comparison charts. TPC suites (which carry scale-factor
/// pills) get storage + scale-factor toggles that swap the visible charts in
/// place; everything else is a single set of facet charts.
fn speedup_section(generation: &ReadGeneration, group: &LandingGroup) -> Markup {
    if group.scale_pills.len() >= 2 {
        return query_fanout_section(generation, group);
    }
    let (facets, faceted_by_engine) = build_facets(generation, &group.chart_links);
    if facets.is_empty() {
        return html! {};
    }
    let order_noun = if faceted_by_engine {
        "Query #"
    } else {
        "Dataset"
    };
    html! {
        section.current-group id=(anchor_for(&group.slug)) {
            header.current-group-head {
                (collapsible_name(&group.name))
                (speedup_sort_control(order_noun))
            }
            (group_blurb(&group.slug, group.description.as_deref()))
            div.speedup-grid {
                @for ed in &facets {
                    (speedup_figure(ed))
                }
            }
        }
    }
}

/// One scale factor's pre-built charts for a fan-out section.
struct SfSet {
    /// Display label, e.g. `SF10`.
    label: String,
    /// Numeric scale factor used as the toggle's data attribute, e.g. `10`.
    value: String,
    /// Whether this is the scale factor shown initially (the largest).
    current: bool,
    facets: Vec<EngineData>,
}

/// A TPC suite as one section with a storage toggle and a scale-factor toggle
/// that swap the visible charts in place (no navigation). The heading drops the
/// `(NVMe) (SF=N)` parenthetical the group name carries — those dimensions are
/// the toggles now. Each scale factor's charts are pre-rendered and hidden until
/// selected (`chart-init.js` resizes them on show).
fn query_fanout_section(generation: &ReadGeneration, group: &LandingGroup) -> Markup {
    let all = generation.groups();
    let dataset = group
        .name
        .split(" (")
        .next()
        .unwrap_or(&group.name)
        .to_string();
    let mut sets: Vec<SfSet> = Vec::new();
    for pill in &group.scale_pills {
        let Some(sf_group) = all.iter().find(|g| g.slug == pill.slug) else {
            continue;
        };
        let (facets, _) = build_facets(generation, &sf_group.charts);
        if facets.is_empty() {
            continue;
        }
        sets.push(SfSet {
            label: pill.label.clone(),
            value: pill.label.trim_start_matches("SF").to_string(),
            current: pill.current,
            facets,
        });
    }
    if sets.is_empty() {
        return html! {};
    }

    html! {
        section.current-group id=(anchor_for(&group.slug)) {
            header.current-group-head {
                (collapsible_name(&dataset))
                (storage_toggle())
                (sf_toggle(&sets))
                (speedup_sort_control("Query #"))
            }
            // Strip the "at SF=N (~XGB)" clause — the SF toggle owns that now.
            (group_blurb(&group.slug, group.description.as_deref().map(|d| d.split(" at SF=").next().unwrap_or(d))))
            div.speedup-sf-sets {
                @for set in &sets {
                    div.speedup-sf data-sf=(set.value) hidden[!set.current] {
                        div.speedup-grid {
                            @for ed in &set.facets {
                                (speedup_figure(ed))
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Storage toggle (NVMe / S3). S3 is disabled until those runs are ingested,
/// but shown so the dimension reads cleanly rather than living in a paren.
fn storage_toggle() -> Markup {
    html! {
        div.dim-toggle data-role="storage-toggle" role="group" aria-label="Storage" {
            button.dim-btn.dim-btn--active type="button" data-storage="nvme" aria-pressed="true" {
                "NVMe"
            }
            button.dim-btn type="button" data-storage="s3" disabled aria-pressed="false" {
                "S3"
            }
        }
    }
}

/// Scale-factor toggle; `chart-init.js` swaps the matching `.speedup-sf` set.
fn sf_toggle(sets: &[SfSet]) -> Markup {
    html! {
        div.dim-toggle data-role="sf-toggle" role="group" aria-label="Scale factor" {
            @for set in sets {
                button.dim-btn.dim-btn--active[set.current]
                    type="button"
                    data-sf=(set.value)
                    aria-pressed=(set.current) {
                    (set.label)
                }
            }
        }
    }
}

/// A short editorial blurb plus a link into the Raw-data view. Rendered inside
/// the collapsible body so it folds away with the charts. Empty when there's no
/// description. `slug` targets the Raw-data link; `description` is passed in so
/// fan-out sections can strip the scale-factor clause (the SF toggle owns that).
fn group_blurb(slug: &str, description: Option<&str>) -> Markup {
    let Some(description) = description else {
        return html! {};
    };
    html! {
        p.current-group-blurb {
            (description) ". "
            a.current-group-rawlink href=(format!("/group/{slug}")) {
                "Detailed charts in Raw data →"
            }
        }
    }
}

/// The group heading, rendered as a collapse toggle: clicking it expands or
/// collapses the section's charts (`chart-init.js`'s `initCurrentCollapse`).
/// Sits alongside (not wrapping) the count link and sort control, so those stay
/// independently clickable.
fn collapsible_name(name: &str) -> Markup {
    html! {
        h2.current-group-name {
            button.current-collapse-btn
                type="button"
                data-role="current-collapse"
                aria-expanded="true" {
                span.current-collapse-caret aria-hidden="true" {}
                (name)
            }
        }
    }
}

/// Sort toggle for a speedup section: "Speedup" (default, biggest win first) vs
/// the natural item order (labelled `order_noun`, e.g. "Query #" or "Dataset").
/// `chart-init.js` wires the clicks and re-sorts every chart in the section.
fn speedup_sort_control(order_noun: &str) -> Markup {
    html! {
        div.speedup-sort data-role="speedup-sort" role="group" aria-label="Sort" {
            span.speedup-sort-label { "Sort" }
            button.speedup-sort-btn.speedup-sort-btn--active
                type="button" data-sort="speedup" aria-pressed="true" { "Speedup" }
            button.speedup-sort-btn type="button" data-sort="query" aria-pressed="false" { (order_noun) }
        }
    }
}

/// One facet's comparison figure: the facet label (when present), the "A vs B"
/// format dropdowns, and the per-item diverging-bar distribution. The headline
/// stat and win count are computed client-side (they depend on the selection),
/// so their spans start empty for `chart-init.js` to fill.
fn speedup_figure(ed: &EngineData) -> Markup {
    let height = 64 + ed.queries.len() * 13;
    let json = serde_json::to_string(ed).unwrap_or_else(|_| "{}".into());
    html! {
        figure.speedup data-role="speedup-chart" {
            figcaption.speedup-head {
                @if !ed.facet.is_empty() {
                    span.speedup-engine { (ed.facet) }
                }
                div.speedup-compare {
                    (format_select("speedup-a", &ed.formats, &ed.default_a))
                    span.speedup-vs { "vs" }
                    (format_select("speedup-b", &ed.formats, &ed.default_b))
                }
                span.speedup-stat data-role="speedup-stat" {}
                span.speedup-wins data-role="speedup-wins" {}
            }
            div.speedup-chart-wrap style=(format!("height:{height}px")) {
                canvas data-role="speedup-canvas" {}
            }
            script type="application/json" data-role="speedup-data" {
                (PreEscaped(escape_json_for_script(&json)))
            }
        }
    }
}

/// A format-comparison `<select>` pre-selecting `default`. `chart-init.js` reads
/// it by `data-role` and recomputes the chart on change.
fn format_select(role: &str, formats: &[FormatOpt], default: &str) -> Markup {
    html! {
        select.speedup-select data-role=(role) aria-label="Comparison format" {
            @for f in formats {
                option value=(f.id) selected[f.id == default] { (f.label) }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// "As of <short-sha> · <date>" stamp linking to the snapshot commit, pulled
/// from the latest commit in any chart payload (the commit timeline is global,
/// so the first payload that has one is representative). Renders nothing when
/// no commit is available.
fn snapshot_stamp(generation: &ReadGeneration, groups: &[LandingGroup]) -> Markup {
    let commit = groups
        .iter()
        .flat_map(|g| g.chart_links.iter())
        .find_map(|link| {
            generation
                .chart_payload(&link.slug)
                .and_then(|p| p.commits.last().cloned())
        });
    let Some(commit) = commit else {
        return html! {};
    };
    let short = commit.sha.get(..7).unwrap_or(&commit.sha).to_string();
    let date = commit
        .timestamp
        .get(..10)
        .unwrap_or(&commit.timestamp)
        .to_string();
    html! {
        p.current-stamp {
            "As of "
            a.current-stamp-sha href=(commit.url) rel="noopener noreferrer" target="_blank" {
                code { (short) }
            }
            " · " (date)
            @if !commit.message.is_empty() {
                span.current-stamp-msg { " — " (commit.message) }
            }
        }
    }
}
