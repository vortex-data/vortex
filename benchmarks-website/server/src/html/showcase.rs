// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Landing-page "claim → why it matters → proof" body.
//!
//! The landing leads with four Vortex-vs-Parquet claims in a 2×2 grid. Each
//! cell pairs a headline stat with a real bar chart of the most recent
//! snapshot — the latest-commit value of each series in that workload's chart
//! payload, labelled with its actual figure and captioned with the commit's
//! date and message. The full catalogue lives behind "Show me everything"
//! (`/current`); "Show me more" / "Source" deep-link to the matching section
//! there (see [`super::anchor_for`]).
//!
//! Taglines use Vortex's canonical published figures; the bars and the scans
//! geomeans ([`ScanGeomeans`]) are live, from the latest commit.

use maud::Markup;
use maud::PreEscaped;
use maud::html;
use serde_json::Value;

use super::anchor_for;
use crate::api::Group;
use crate::api::Summary;
use crate::api::UnitKind;
use crate::read_model::ReadGeneration;
use crate::slug::GroupKey;

const RANDOM_ACCESS_WHY: &str = "Point lookups (reading specific rows by position) drive feature stores, vector search, and \
     anything that serves individual records. Parquet packs rows into large row groups that must \
     be decoded almost whole to return a single value, so one row costs as much as thousands. \
     Vortex addresses rows directly: a lookup Parquet answers in ~200 ms returns in ~1.5 ms.";

const SCANS_WHY: &str = "Analytical queries like dashboards, reports, and the read side of ETL spend most of their \
     time scanning columns off disk and decoding them, so scan throughput sets how fast they \
     return. This number comes from TPC-H's scan-heavy queries, Q1 and Q6, which read almost an \
     entire fact table while doing little compute, so it measures the format, not the query planner.";

const WRITES_WHY: &str = "Data is encoded once and read many times, but the encode step gates ingestion: how quickly \
     new data becomes queryable. Parquet spends heavily in its row-group encoder; Vortex encodes \
     the same data in about a fifth of the time, so pipelines clear backlogs and data goes live \
     sooner.";

const SIZE_WHY: &str = "Storage cost and the bytes a query must move both track file size, so a faster format that \
     bloated on disk would trade one bill for another. Vortex holds within a few percent of \
     Parquet's compression ratio, so the speed above comes at no size penalty.";

/// Render the landing body from the live read generation.
pub(super) fn showcase_body(generation: &ReadGeneration) -> Markup {
    let groups = generation.groups();

    let random_access = group_by_summary(&groups, |s| matches!(s, Summary::RandomAccess { .. }));
    let writes = group_by_summary(&groups, |s| matches!(s, Summary::Compression { .. }));
    let size = group_by_summary(&groups, |s| matches!(s, Summary::CompressionSize { .. }));
    let tpch_all = tpch_groups(&groups);
    let tpch = tpch_all.first().map(|&(_, g)| g);

    html! {
        section.showcase {
            header.showcase-intro {
                p.showcase-eyebrow { "Vortex vs Apache Parquet" }
                h2.showcase-headline {
                    "A columnar format built for the read patterns Parquet wasn't."
                }
            }
            div.claims {
                (claim(generation, "100×", "faster random access", RANDOM_ACCESS_WHY,
                       random_access.and_then(first_chart_slug), None, more_href(random_access),
                       source_label_for(random_access, "Random Access"), None))
                (claim(generation, "10–20×", "faster scans", SCANS_WHY,
                       tpch.and_then(|g| chart_slug_named(g, "Q1")), Some("datafusion"),
                       more_href(tpch), Some("scan-heavy TPC-H · TPC-H · TPC-DS".to_string()),
                       (!tpch_all.is_empty()).then(|| scans_figure(generation, &tpch_all))))
                (claim(generation, "5×", "faster writes", WRITES_WHY,
                       writes.and_then(first_chart_slug), Some("encode"), more_href(writes),
                       source_label_for(writes, "Compression encode"), None))
                (claim(generation, "≈1×", "the size of Parquet", SIZE_WHY,
                       size.and_then(first_chart_slug), None, more_href(size),
                       source_label_for(size, "Compression Size"), None))
            }
            div.showcase-cta {
                a.show-everything href="/current" {
                    "Show me everything"
                    span.show-everything-arrow aria-hidden="true" { " →" }
                }
            }
        }
    }
}

/// One claim cell: the stacked stat + snapshot bars on top, the "why it
/// matters" spanning the full cell width below, then "show me more".
#[expect(clippy::too_many_arguments)]
fn claim(
    generation: &ReadGeneration,
    metric: &str,
    label: &str,
    why: &str,
    chart_slug: Option<String>,
    series_filter: Option<&str>,
    more_href: Option<String>,
    source: Option<String>,
    figure_override: Option<Markup>,
) -> Markup {
    html! {
        article.claim {
            div.claim-head {
                div.claim-stat {
                    span.claim-metric { (metric) }
                    span.claim-label { (label) }
                }
                div.claim-figure {
                    @if let Some(fig) = &figure_override {
                        (fig)
                    } @else if let Some(slug) = &chart_slug {
                        (snapshot_bars(generation, slug, series_filter))
                    }
                    @if let (Some(slug), Some(label)) = (&chart_slug, &source) {
                        @let href = more_href.clone().unwrap_or_else(|| format!("/chart/{slug}"));
                        div.claim-source {
                            span.claim-source-label { "Source" }
                            a.claim-source-val href=(href) { (label) }
                        }
                    }
                }
            }
            p.claim-why { (why) }
            @if let Some(href) = &more_href {
                a.claim-more href=(href) {
                    "Show me more"
                    span.claim-more-arrow aria-hidden="true" { " →" }
                }
            }
        }
    }
}

/// A bar chart of the most recent snapshot: the latest-commit value of each
/// series in the chart payload, longest first, each bar labelled with its real
/// figure and captioned with the commit's date + message. `series_filter`
/// keeps only series whose name contains it (falling back to all if that would
/// be empty), so the scans cell shows just datafusion and writes just encode.
fn snapshot_bars(generation: &ReadGeneration, slug: &str, series_filter: Option<&str>) -> Markup {
    let Some(payload) = generation.chart_payload(slug) else {
        return html! {};
    };
    let mut bars: Vec<(&String, f64)> = payload
        .series
        .iter()
        .filter(|(name, _)| series_filter.is_none_or(|f| name.contains(f)))
        .filter_map(|(name, vals)| latest_value(vals).map(|v| (name, v)))
        .collect();
    if bars.is_empty() {
        bars = payload
            .series
            .iter()
            .filter_map(|(name, vals)| latest_value(vals).map(|v| (name, v)))
            .collect();
    }
    if bars.is_empty() {
        return html! {};
    }
    // Parquet/baseline on top, Vortex below — consistent across all claims,
    // regardless of which is larger (Vortex can be bigger, e.g. on size).
    bars.sort_by(|a, b| {
        a.0.contains("vortex")
            .cmp(&b.0.contains("vortex"))
            .then(b.1.total_cmp(&a.1))
    });
    let items: Vec<BarItem> = bars
        .into_iter()
        .map(|(name, v)| BarItem::plain(series_label(name), v, name.contains("vortex")))
        .collect();
    bars_markup(&items, payload.unit_kind)
}

/// One bar in a [`bars_markup`] chart. `engine`/`format` are the series'
/// classification tags; when present they're emitted as `data-engine` /
/// `data-format` so the client's global filter can hide the row by engine or
/// format (the Current page uses this). The showcase's own bars carry no tags
/// (it has no filter), so they use [`BarItem::plain`].
pub(super) struct BarItem {
    /// Human label shown in the row's name column.
    pub(super) label: String,
    /// The bar's value, in the chart's base unit.
    pub(super) value: f64,
    /// Whether this is a Vortex series (gets the solid accent fill).
    pub(super) is_vortex: bool,
    /// Engine tag, e.g. `duckdb`; `None` when the series has no engine dimension.
    pub(super) engine: Option<String>,
    /// Format tag, e.g. `parquet`; `None` when the series has no format dimension.
    pub(super) format: Option<String>,
}

impl BarItem {
    /// A bar with no engine/format tags (not filterable).
    pub(super) fn plain(label: String, value: f64, is_vortex: bool) -> Self {
        Self {
            label,
            value,
            is_vortex,
            engine: None,
            format: None,
        }
    }
}

/// Render labelled bars (Vortex solid, others muted), scaled to the largest
/// value, each labelled with its formatted figure. Shared with the Current
/// page ([`super::current`]), which renders one of these per group.
pub(super) fn bars_markup(items: &[BarItem], unit: UnitKind) -> Markup {
    if items.is_empty() {
        return html! {};
    }
    let max = items
        .iter()
        .map(|b| b.value)
        .fold(f64::MIN_POSITIVE, f64::max);
    html! {
        figure.claim-chart {
            div.snapshot-bars {
                @for item in items {
                    @let pct = (item.value / max).clamp(0.03, 1.0) * 100.0;
                    div.sbar-row
                        data-engine=[item.engine.as_deref()]
                        data-format=[item.format.as_deref()] {
                        span.sbar-name { (item.label) }
                        div.sbar-track {
                            div.sbar-fill.sbar-fill--vortex[item.is_vortex]
                                style=(PreEscaped(format!("width:{pct:.1}%"))) {}
                        }
                        span.sbar-val { (format_value(item.value, unit)) }
                    }
                }
            }
        }
    }
}

/// Scans bars from the scan-heavy geomean (Q1, Q6 at the largest SF): geomean
/// parquet vs geomean vortex datafusion latency — the aggregate the "faster
/// scans" claim is about, rather than a single (possibly unrepresentative) query.
fn scan_geomean_bars(generation: &ReadGeneration, tpch: &Group) -> Markup {
    let mut parquet = Vec::new();
    let mut vortex = Vec::new();
    for q in ["Q1", "Q6"] {
        let Some(slug) = chart_slug_named(tpch, q) else {
            continue;
        };
        let Some(payload) = generation.chart_payload(&slug) else {
            continue;
        };
        if let Some(v) = payload
            .series
            .get("datafusion:parquet")
            .and_then(latest_value)
        {
            parquet.push(v);
        }
        if let Some(v) = payload
            .series
            .get("datafusion:vortex-file-compressed")
            .and_then(latest_value)
        {
            vortex.push(v);
        }
    }
    match (geomean(&parquet), geomean(&vortex)) {
        (Some(p), Some(v)) => bars_markup(
            &[
                BarItem::plain("Parquet".to_string(), p, false),
                BarItem::plain("Vortex".to_string(), v, true),
            ],
            UnitKind::TimeNs,
        ),
        _ => html! {},
    }
}

/// Geometric mean of the positive, finite values.
fn geomean(values: &[f64]) -> Option<f64> {
    let valid: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| *v > 0.0 && v.is_finite())
        .collect();
    if valid.is_empty() {
        return None;
    }
    Some((valid.iter().map(|v| v.ln()).sum::<f64>() / valid.len() as f64).exp())
}

/// Latest finite value in a series' value array (arrays run oldest-first).
pub(super) fn latest_value(values: &Value) -> Option<f64> {
    values
        .as_array()?
        .iter()
        .rev()
        .find_map(|v| v.as_f64().filter(|n| n.is_finite()))
}

/// Short, human series label. Collapses `engine:` prefixes and `:op` suffixes
/// to the format, and renames the on-disk format strings.
fn series_label(name: &str) -> String {
    if name.contains("vortex") {
        "Vortex".to_string()
    } else if name.contains("parquet") {
        "Parquet".to_string()
    } else {
        name.to_string()
    }
}

pub(super) fn format_value(v: f64, unit: UnitKind) -> String {
    match unit {
        UnitKind::TimeNs => format_time_ns(v),
        UnitKind::Bytes => format_bytes(v),
        UnitKind::ThroughputMbS => format!("{v:.0} MB/s"),
        UnitKind::Ratio | UnitKind::Count => format!("{v:.2}"),
    }
}

fn format_time_ns(ns: f64) -> String {
    let abs = ns.abs();
    if abs >= 1e9 {
        format!("{:.1} s", ns / 1e9)
    } else if abs >= 1e6 {
        format!("{:.1} ms", ns / 1e6)
    } else if abs >= 1e3 {
        format!("{:.1} µs", ns / 1e3)
    } else {
        format!("{ns:.0} ns")
    }
}

fn format_bytes(bytes: f64) -> String {
    if bytes >= 1_073_741_824.0 {
        format!("{:.1} GiB", bytes / 1_073_741_824.0)
    } else if bytes >= 1_048_576.0 {
        format!("{:.1} MiB", bytes / 1_048_576.0)
    } else if bytes >= 1024.0 {
        format!("{:.1} KiB", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B")
    }
}

fn group_by_summary(groups: &[Group], pred: impl Fn(&Summary) -> bool) -> Option<&Group> {
    groups
        .iter()
        .find(|g| g.summary.as_ref().is_some_and(|s| pred(s)))
}

/// Deep link from a claim into the Current page's matching group section.
/// [`anchor_for`] keeps the fragment in lockstep with the section id the
/// Current page emits, so the link scrolls to and `:target`-highlights it.
fn more_href(group: Option<&Group>) -> Option<String> {
    group.map(|g| format!("/current#{}", anchor_for(&g.slug)))
}

fn first_chart_slug(group: &Group) -> Option<String> {
    group.charts.first().map(|c| c.slug.clone())
}

/// Slug of a named chart in a group (e.g. `Q1`), falling back to the first chart.
fn chart_slug_named(group: &Group, name: &str) -> Option<String> {
    group
        .charts
        .iter()
        .find(|c| c.name == name)
        .or_else(|| group.charts.first())
        .map(|c| c.slug.clone())
}

/// All TPC-H query groups with their scale factors, largest first.
fn tpch_groups(groups: &[Group]) -> Vec<(f64, &Group)> {
    let mut v: Vec<(f64, &Group)> = groups
        .iter()
        .filter_map(|g| match GroupKey::from_slug(&g.slug) {
            Ok(GroupKey::QueryGroup {
                dataset,
                scale_factor: Some(sf),
                ..
            }) if dataset == "tpch" => Some((sf.parse::<f64>().unwrap_or(0.0), g)),
            _ => None,
        })
        .collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    v
}

/// Scans figure: the largest SF's scan-heavy geomean bars (Q1+Q6, datafusion
/// Vortex vs Parquet). Smaller scale factors live on the Current/Historic pages.
fn scans_figure(generation: &ReadGeneration, tpch: &[(f64, &Group)]) -> Markup {
    let Some(&(_, largest)) = tpch.first() else {
        return html! {};
    };
    scan_geomean_bars(generation, largest)
}

/// The "Source" label for a single-comparison claim, from its group's first
/// chart (the dataset) and a benchmark label. [`claim`] renders it as a link
/// to that chart.
fn source_label_for(group: Option<&Group>, label: &str) -> Option<String> {
    let group = group?;
    let dataset = group
        .charts
        .first()
        .map(|c| c.name.as_str())
        .unwrap_or_default();
    Some(format!("{label} · {dataset}"))
}
