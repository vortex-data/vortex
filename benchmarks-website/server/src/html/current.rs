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
use std::collections::BTreeSet;

use maud::Markup;
use maud::PreEscaped;
use maud::html;
use serde::Serialize;

use super::anchor_for;
use super::landing::LandingGroup;
use super::landing::ScalePill;
use super::render::chart_controls;
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
                div.methodology {
                    p.methodology-text {
                        "Each chart distils one benchmark suite into a single "
                        strong { "Vortex / Parquet ratio" }
                        " at the latest develop commit — geometric mean over the suite's items (queries, datasets, access patterns). "
                        strong { "1× is parity; above 1× means Vortex wins" }
                        " (faster for time, smaller for size). Swap either side with the dropdowns; "
                        a href="/historic" { "Historic Data" }
                        " plots the same number at every commit."
                    }
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

/// Headline-split key for one line: engine when set, else facet (the op for
/// compression-time, `""` for single-chart kinds).
fn history_split_key(l: &HistoryLine) -> &str {
    if !l.engine.is_empty() {
        &l.engine
    } else {
        &l.facet
    }
}

/// Pretty label for a headline-split key (engine or op). The headline charts
/// split per engine for query suites and per op for compression-time, mirroring
/// the per-card layout — this turns the raw key into the chart's sub-title.
fn pretty_split_label(key: &str) -> &str {
    match key {
        "datafusion" => "DataFusion",
        "duckdb" => "DuckDB",
        "encode" => "Encode",
        "decode" => "Decode",
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

/// One facet's Overview headline: the geomean of the default Vortex-vs-Parquet
/// ratios (`B / A`, so > 1 = Vortex wins), with its win count and item total,
/// plus the unit's comparison verb. This is exactly the number the Current page
/// headlines for the same facet, so the Overview claim and its proof can't
/// disagree.
pub(super) struct FacetGeomean {
    /// The facet name: an engine (`datafusion`), an operation (`encode`), or
    /// `""` for a no-facet group (random access, size).
    pub(super) facet: String,
    /// Geomean of the per-item `B / A` ratios.
    pub(super) geomean: f64,
    /// Items where Vortex won (ratio ≥ 1).
    pub(super) wins: usize,
    /// Items that measured both formats.
    pub(super) total: usize,
}

/// Per-facet Overview geomeans for a group's charts — one per engine /
/// operation / the single no-facet chart. Facets with no comparable items are
/// skipped. Used by the Overview ([`super::showcase`]) to source each claim's
/// number from the same synthesis the Current page renders.
pub(super) fn facet_geomeans(
    generation: &ReadGeneration,
    chart_links: &[ChartLink],
) -> Vec<FacetGeomean> {
    let (facets, _) = build_facets(generation, chart_links);
    facets
        .iter()
        .filter_map(|ed| {
            let mut ratios = Vec::new();
            let mut wins = 0;
            for q in &ed.queries {
                let (Some(&a), Some(&b)) =
                    (q.values.get(&ed.default_a), q.values.get(&ed.default_b))
                else {
                    continue;
                };
                if a > 0.0 && b > 0.0 {
                    let ratio = b / a;
                    if ratio >= 1.0 {
                        wins += 1;
                    }
                    ratios.push(ratio);
                }
            }
            if ratios.is_empty() {
                return None;
            }
            let geomean = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
            Some(FacetGeomean {
                facet: ed.facet.clone(),
                geomean,
                wins,
                total: ratios.len(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Synthesis over time (Previous Versions headline chart)
// ---------------------------------------------------------------------------

/// One line on the headline chart: a format's geomean speedup vs Parquet under
/// one facet, at each commit (`null` where that commit has no data).
#[derive(Serialize, Clone)]
struct HistoryLine {
    /// Legend label, e.g. `"DataFusion · Vortex"` / `"Encode"` / `"Vortex"`.
    label: String,
    /// Facet (engine / op / `""`).
    facet: String,
    /// Query engine if this facet is one (`datafusion` / `duckdb`), else `""`.
    /// The client maps it to a line dash pattern (engine lexicon).
    engine: String,
    /// Format id (the thing compared to Parquet); the client maps it to a colour.
    format: String,
    /// Geomean `parquet / format` per commit (> 1 = the format beats Parquet).
    speedups: Vec<Option<f64>>,
}

/// One commit on the headline chart's x-axis.
#[derive(Serialize, Clone)]
struct HistoryCommit {
    /// Short SHA, surfaced in the tooltip title.
    sha: String,
    /// First-line message, for the tooltip.
    msg: String,
    /// ISO timestamp of the commit. Drives the date label on the x-axis tick
    /// callback (see `chart-init.js::formatAxisDate`).
    timestamp: String,
}

/// Inline payload for the Previous-Versions headline line chart: every
/// `engine:format` measured against Parquet, as a geomean computed at each
/// commit so it can be plotted as a trajectory. Parquet is the implicit 1×
/// baseline (not drawn).
#[derive(Serialize)]
struct HistoryData {
    /// Comparison verb from the unit (`"faster"` / `"smaller"`).
    metric: &'static str,
    /// x-axis, oldest commit first.
    commits: Vec<HistoryCommit>,
    /// One line per `(facet, non-Parquet format)`.
    lines: Vec<HistoryLine>,
}

/// Pretty label for a facet name.
fn facet_label(facet: &str) -> String {
    match facet {
        "datafusion" => "DataFusion".to_string(),
        "duckdb" => "DuckDB".to_string(),
        "encode" => "Encode".to_string(),
        "decode" => "Decode".to_string(),
        other => other.to_string(),
    }
}

/// Geometric mean of positive, finite values.
fn geomean_of(ratios: &[f64]) -> Option<f64> {
    let valid: Vec<f64> = ratios
        .iter()
        .copied()
        .filter(|r| *r > 0.0 && r.is_finite())
        .collect();
    if valid.is_empty() {
        return None;
    }
    Some((valid.iter().map(|r| r.ln()).sum::<f64>() / valid.len() as f64).exp())
}

/// Compute, for every `(facet, non-Parquet format)`, the per-commit geomean of
/// `parquet / format` across a group's charts — each becomes one headline line.
/// Mirrors [`facet_geomeans`] but over the full per-commit series arrays (not
/// just the latest), and over every format rather than just Vortex. Parquet is
/// the baseline; its 1× line is implicit. The newest point of the
/// Vortex line equals the Latest-Commit headline.
fn build_history(generation: &ReadGeneration, chart_links: &[ChartLink]) -> Option<HistoryData> {
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;

    // (facet, format) -> full-sha -> per-item parquet/format ratios at that commit
    let mut acc: BTreeMap<(String, String), BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new(); // full shas, oldest first, deduped
    let mut labels: BTreeMap<String, String> = BTreeMap::new(); // sha -> message
    let mut shorts: BTreeMap<String, String> = BTreeMap::new(); // sha -> short sha
    let mut times: BTreeMap<String, String> = BTreeMap::new(); // sha -> ISO timestamp
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut engine_facets: BTreeSet<String> = BTreeSet::new(); // facets that are query engines
    let mut unit = UnitKind::TimeNs;

    for link in chart_links {
        let Some(payload) = generation.chart_payload(&link.slug) else {
            continue;
        };
        unit = payload.unit_kind;
        for c in &payload.commits {
            if seen.insert(c.sha.clone()) {
                order.push(c.sha.clone());
                shorts.insert(c.sha.clone(), c.sha.chars().take(7).collect());
                labels.insert(c.sha.clone(), c.message.clone());
                times.insert(c.sha.clone(), c.timestamp.clone());
            }
        }
        // Per facet, every format's series name.
        let mut by_facet: BTreeMap<String, BTreeMap<String, &str>> = BTreeMap::new();
        for (name, tag) in &payload.series_meta {
            let Some(fmt) = tag.format.as_deref() else {
                continue;
            };
            let facet = facet_of(name, tag.engine.as_deref());
            if tag.engine.is_some() {
                engine_facets.insert(facet.clone());
            }
            by_facet
                .entry(facet)
                .or_default()
                .insert(fmt.to_string(), name.as_str());
        }
        for (facet, formats) in &by_facet {
            let Some(pname) = formats.get(PARQUET_FORMAT) else {
                continue; // no baseline in this facet
            };
            let Some(parr) = payload.series.get(*pname).and_then(|v| v.as_array()) else {
                continue;
            };
            for (fmt, sname) in formats {
                if fmt == PARQUET_FORMAT {
                    continue;
                }
                let Some(sarr) = payload.series.get(*sname).and_then(|v| v.as_array()) else {
                    continue;
                };
                for (i, c) in payload.commits.iter().enumerate() {
                    let (Some(pv), Some(sv)) = (
                        parr.get(i).and_then(|x| x.as_f64()),
                        sarr.get(i).and_then(|x| x.as_f64()),
                    ) else {
                        continue;
                    };
                    if pv > 0.0 && sv > 0.0 {
                        acc.entry((facet.clone(), fmt.clone()))
                            .or_default()
                            .entry(c.sha.clone())
                            .or_default()
                            .push(pv / sv);
                    }
                }
            }
        }
    }

    if acc.is_empty() || order.is_empty() {
        return None;
    }
    // Label rule: prefix the facet only when faceted, append the format only
    // when more than one non-Parquet format is present (so compression reads
    // "Encode"/"Decode", random access reads "Vortex", ClickBench reads
    // "DataFusion · Vortex").
    let faceted = acc.keys().any(|(f, _)| !f.is_empty());
    let multi_format = acc
        .keys()
        .map(|(_, fmt)| fmt)
        .collect::<BTreeSet<_>>()
        .len()
        > 1;
    // Order lines by format (Vortex first) then facet, so colour assignment is stable.
    let mut keys: Vec<(String, String)> = acc.keys().cloned().collect();
    keys.sort_by(|a, b| {
        format_order(&a.1)
            .cmp(&format_order(&b.1))
            .then_with(|| a.0.cmp(&b.0))
    });

    let commits = order
        .iter()
        .map(|sha| HistoryCommit {
            sha: shorts.get(sha).cloned().unwrap_or_default(),
            msg: labels.get(sha).cloned().unwrap_or_default(),
            timestamp: times.get(sha).cloned().unwrap_or_default(),
        })
        .collect();
    let lines = keys
        .into_iter()
        .map(|(facet, fmt)| {
            let mut parts: Vec<String> = Vec::new();
            if faceted && !facet.is_empty() {
                parts.push(facet_label(&facet));
            }
            if multi_format {
                parts.push(format_label(&fmt).to_string());
            }
            let label = if parts.is_empty() {
                format_label(&fmt).to_string()
            } else {
                parts.join(" · ")
            };
            let by_sha = &acc[&(facet.clone(), fmt.clone())];
            let speedups = order
                .iter()
                .map(|sha| by_sha.get(sha).and_then(|r| geomean_of(r)))
                .collect();
            let engine = if engine_facets.contains(&facet) {
                facet.clone()
            } else {
                String::new()
            };
            HistoryLine {
                label,
                facet,
                engine,
                format: fmt,
                speedups,
            }
        })
        .collect();
    Some(HistoryData {
        metric: metric_for(unit),
        commits,
        lines,
    })
}

/// The Previous-Versions headline for a group. TPC suites (which carry storage
/// and scale-factor pills) get in-place toggles that swap the headline chart;
/// everything else is a single chart. Empty when there's no comparable history.
pub(super) fn history_section(generation: &ReadGeneration, group: &LandingGroup) -> Markup {
    if group.scale_pills.is_empty() {
        return history_headline(generation, &group.chart_links);
    }
    let all = generation.groups();
    html! {
        div.history-fanout {
            div.history-controls {
                (storage_toggle_pills(&group.scale_pills))
                (sf_toggle_pills(&group.scale_pills))
            }
            div.history-sf-sets {
                @for pill in &group.scale_pills {
                    div.speedup-sf
                        data-sf=(pill.sf_value)
                        data-storage=(pill.storage_value)
                        hidden[!pill.current] {
                        @if let Some(sf_group) = all.iter().find(|g| g.slug == pill.slug) {
                            (history_headline(generation, &sf_group.charts))
                        }
                    }
                }
            }
        }
    }
}

/// Scale-factor toggle buttons for a TPC suite. One button per distinct SF in
/// `pills`, ordered by first appearance (the cluster sorts pills smallest →
/// largest). Active is the SF carried by the current pill.
pub(super) fn sf_toggle_pills(pills: &[ScalePill]) -> Markup {
    let current_sf = pills
        .iter()
        .find(|p| p.current)
        .map(|p| p.sf_value.as_str())
        .unwrap_or_default();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut row: Vec<&ScalePill> = Vec::new();
    for p in pills {
        if seen.insert(p.sf_value.as_str()) {
            row.push(p);
        }
    }
    html! {
        div.dim-toggle data-role="sf-toggle" role="group" aria-label="Scale factor" {
            @for p in &row {
                @let active = p.sf_value == current_sf;
                button.dim-btn.dim-btn--active[active]
                    type="button"
                    data-sf=(p.sf_value)
                    aria-pressed=(active) {
                    (p.sf_label)
                }
            }
        }
    }
}

/// Storage toggle buttons for a TPC suite. Renders one button per tier that
/// actually has data, in canonical order (NVMe before S3). A lone tier still
/// renders as a single (active) pill so the dimension stays visible under the
/// bare heading (e.g. TPC-DS is NVMe-only); the row only collapses with no data.
pub(super) fn storage_toggle_pills(pills: &[ScalePill]) -> Markup {
    let current_storage = pills
        .iter()
        .find(|p| p.current)
        .map(|p| p.storage_value.as_str())
        .unwrap_or_default();
    let present: BTreeSet<&str> = pills.iter().map(|p| p.storage_value.as_str()).collect();
    let row: Vec<&&str> = super::TPC_STORAGE_TIERS
        .iter()
        .filter(|t| present.contains(*t))
        .collect();
    if row.is_empty() {
        return html! {};
    }
    html! {
        div.dim-toggle data-role="storage-toggle" role="group" aria-label="Storage" {
            @for tier in &row {
                @let active = **tier == current_storage;
                @let label = super::storage_label(tier);
                button.dim-btn.dim-btn--active[active]
                    type="button"
                    data-storage=(tier)
                    aria-pressed=(active) {
                    (label)
                }
            }
        }
    }
}

/// One group's headline chart: one geomean-speedup line per facet over the
/// version timeline (1× baseline). Empty when there's no comparable history.
/// Headline synthesis chart for one group. Mirrors the per-card split rule:
/// if any line has an engine, split per engine (DataFusion | DuckDB); else if
/// any line has a non-empty facet (op for compression-time), split per facet
/// (Encode | Decode); else render one chart. Each sub-chart re-labels its
/// lines by format only (the sub-title carries the engine/op).
pub(super) fn history_headline(generation: &ReadGeneration, chart_links: &[ChartLink]) -> Markup {
    let Some(data) = build_history(generation, chart_links) else {
        return html! {};
    };
    let keys: BTreeSet<String> = data
        .lines
        .iter()
        .map(|l| history_split_key(l).to_string())
        .collect();
    if keys.len() <= 1 {
        let json = serde_json::to_string(&data).unwrap_or_default();
        return html! {
            figure.history-figure data-role="history-chart" {
                div.history-chart-wrap {
                    (chart_controls(true))
                    canvas data-role="history-canvas" {}
                }
                script type="application/json" data-role="history-data" {
                    (PreEscaped(escape_json_for_script(&json)))
                }
            }
        };
    }
    html! {
        div.history-headline-grid {
            @for key in &keys {
                @let sub_lines: Vec<HistoryLine> = data.lines.iter()
                    .filter(|l| history_split_key(l) == key.as_str())
                    .map(|l| HistoryLine {
                        label: format_label(&l.format).to_string(),
                        facet: l.facet.clone(),
                        engine: l.engine.clone(),
                        format: l.format.clone(),
                        speedups: l.speedups.clone(),
                    })
                    .collect();
                @let sub = HistoryData {
                    metric: data.metric,
                    commits: data.commits.clone(),
                    lines: sub_lines,
                };
                @let sub_json = serde_json::to_string(&sub).unwrap_or_default();
                figure.history-figure data-role="history-chart" {
                    h4.history-facet-title { (pretty_split_label(key)) }
                    div.history-chart-wrap {
                        (chart_controls(true))
                        canvas data-role="history-canvas" {}
                    }
                    script type="application/json" data-role="history-data" {
                        (PreEscaped(escape_json_for_script(&sub_json)))
                    }
                }
            }
        }
    }
}

/// Render a group as comparison charts. TPC suites (which carry scale-factor
/// pills) get storage + scale-factor toggles that swap the visible charts in
/// place; everything else is a single set of facet charts.
fn speedup_section(generation: &ReadGeneration, group: &LandingGroup) -> Markup {
    if !group.scale_pills.is_empty() {
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
    // All facets in a group share the unit, so the first facet's verb labels
    // the section's magnitude sort ("Speedup" vs "Smaller").
    let metric = facets.first().map(|f| f.metric).unwrap_or("faster");
    html! {
        section.current-group id=(anchor_for(&group.slug)) {
            header.current-group-head {
                (collapsible_name(&group.name, group.description.as_deref()))
                (speedup_sort_control(order_noun, metric))
            }
            div.speedup-grid {
                @for ed in &facets {
                    (speedup_figure(ed))
                }
            }
        }
    }
}

/// One (storage, scale-factor) panel's pre-built charts for a fan-out section.
struct SfSet {
    /// Scale-factor raw value, e.g. `10`. Drives `data-sf` and toggle matching.
    sf_value: String,
    /// Storage raw value, e.g. `nvme`. Drives `data-storage` and toggle matching.
    storage_value: String,
    /// Whether this is the (storage, SF) combination shown initially.
    current: bool,
    facets: Vec<EngineData>,
}

/// A TPC suite as one section with storage and scale-factor toggles that swap
/// the visible charts in place (no navigation). The heading drops both the
/// `(NVMe|S3)` and `(SF=N)` parentheticals — the toggles own those dimensions.
/// Each (storage, SF) panel is pre-rendered and hidden until selected
/// (`chart-init.js` resizes it on show).
fn query_fanout_section(generation: &ReadGeneration, group: &LandingGroup) -> Markup {
    let all = generation.groups();
    let heading = strip_tpc_parentheticals(&group.name);
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
            sf_value: pill.sf_value.clone(),
            storage_value: pill.storage_value.clone(),
            current: pill.current,
            facets,
        });
    }
    if sets.is_empty() {
        return html! {};
    }
    // TPC fan-out is always a query suite (time), but read the verb rather than
    // assume it, so the magnitude sort stays correct if a non-time suite ever
    // fans out.
    let metric = sets
        .first()
        .and_then(|s| s.facets.first())
        .map(|f| f.metric)
        .unwrap_or("faster");

    html! {
        section.current-group id=(anchor_for(&group.slug)) {
            header.current-group-head {
                // The toggles in the body own storage and SF, so the folded-in
                // blurb drops both the "(NVMe|S3)" segment of the description
                // and the "at SF=N (~XGB)" clause.
                (collapsible_name(&heading, group.description.as_deref().map(strip_tpc_blurb_dims)))
                (speedup_sort_control("Query #", metric))
            }
            // Toggles live in the collapsible body so they fold away with the
            // charts. Storage row first, then SF — same order as on /historic.
            div.current-group-body {
                div.history-controls {
                    (storage_toggle_pills(&group.scale_pills))
                    (sf_toggle_pills(&group.scale_pills))
                }
                div.speedup-sf-sets {
                    @for set in &sets {
                        div.speedup-sf
                            data-sf=(set.sf_value)
                            data-storage=(set.storage_value)
                            hidden[!set.current] {
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
}

/// Drop the trailing `(NVMe|S3)` and `(SF=N)` parentheticals from a TPC group
/// name. Both are now in-section toggles, so the heading reads as the bare
/// suite (`TPC-H`, `TPC-DS`).
pub(super) fn strip_tpc_parentheticals(name: &str) -> String {
    let mut s = name;
    if let Some((head, _)) = s.rsplit_once(" (SF=") {
        s = head;
    }
    for tier in [" (NVMe)", " (S3)"] {
        if let Some(stripped) = s.strip_suffix(tier) {
            return stripped.to_string();
        }
    }
    s.to_string()
}

/// Strip the storage parenthetical and the `at SF=N (~XGB)` tail from a TPC
/// description so the folded-in blurb reads cleanly under the merged heading.
fn strip_tpc_blurb_dims(d: &str) -> &str {
    let trimmed = d.split(" at SF=").next().unwrap_or(d);
    for tier in [" (NVMe)", " (S3)"] {
        if let Some(stripped) = trimmed.strip_suffix(tier) {
            return stripped;
        }
    }
    trimmed
}

/// The group heading, rendered as a collapse toggle: clicking it expands or
/// collapses the section's charts (`chart-init.js`'s `initCurrentCollapse`).
/// Sits alongside (not wrapping) the count link and sort control, so those stay
/// independently clickable. The one-line description (when present) is folded in
/// after the title, separated by a middot, so it reads as part of the heading
/// rather than a separate paragraph.
fn collapsible_name(name: &str, blurb: Option<&str>) -> Markup {
    html! {
        h2.current-group-name {
            button.current-collapse-btn
                type="button"
                data-role="current-collapse"
                aria-expanded="true" {
                span.current-collapse-caret aria-hidden="true" {}
                (name)
            }
            @if let Some(blurb) = blurb {
                span.current-group-desc { (blurb) }
            }
        }
    }
}

/// Sort toggle for a speedup section: the magnitude sort (default, biggest win
/// first) vs the natural item order (labelled `order_noun`, e.g. "Query #" or
/// "Dataset"). The magnitude label tracks the unit's comparison verb — "Speedup"
/// for time sections, "Smaller" for size sections — so a compression-size chart
/// doesn't read "Speedup". `chart-init.js` wires the clicks and re-sorts every
/// chart in the section.
fn speedup_sort_control(order_noun: &str, metric: &str) -> Markup {
    // `data-sort="speedup"` is the internal sort-mode key the JS reads (sort by
    // the B/A ratio), independent of the displayed verb; only the label changes.
    let magnitude_label = if metric == "smaller" {
        "Smaller"
    } else {
        "Speedup"
    };
    html! {
        div.speedup-sort data-role="speedup-sort" role="group" aria-label="Sort" {
            span.speedup-sort-label { "Sort" }
            button.speedup-sort-btn.speedup-sort-btn--active
                type="button" data-sort="speedup" aria-pressed="true" { (magnitude_label) }
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
                (chart_controls(false))
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
