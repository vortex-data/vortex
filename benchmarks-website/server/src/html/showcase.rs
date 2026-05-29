// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Landing-page "claim → why it matters → proof" body.
//!
//! The landing leads with four Vortex-vs-Parquet claims in a 2×2 grid. Each
//! cell pairs a headline number with a small blueprint schematic of the
//! *workload* (the access pattern, not the result) and a "why it matters"
//! paragraph, then links to the proof on the Current page.
//!
//! Every headline number is a **live benchmark-set geomean**, computed by the
//! same synthesis the Current page renders ([`super::current::facet_geomeans`]),
//! so a claim and its proof can never disagree. Nothing here is hardcoded.

use maud::Markup;
use maud::PreEscaped;
use maud::html;
use serde_json::Value;

use super::anchor_for;
use super::current::FacetGeomean;
use super::current::facet_geomeans;
use crate::api::Group;
use crate::api::Summary;
use crate::api::UnitKind;
use crate::read_model::ReadGeneration;
use crate::slug::GroupKey;

const RANDOM_ACCESS_WHY: &str = "Point lookups — reading specific rows by position — drive feature stores, vector search, and \
     anything that serves individual records. Parquet packs rows into large row groups that must \
     be decoded almost whole to return a single value, so one row costs as much as thousands. \
     Vortex addresses rows directly.";

const ANALYTICS_WHY: &str = "Dashboards, reports, and the read side of ETL are mostly column scans and aggregations. \
     ClickBench — ClickHouse's 43-query suite over real web-analytics data — is the field's \
     standard test. Vortex is a drop-in: keep your engine, swap Parquet (or the engine's own \
     native format) for a Vortex file, and the same queries return faster — on DataFusion and \
     DuckDB alike.";

const WRITES_WHY: &str = "Data is encoded once and read many times, but the encode step gates ingestion — how quickly \
     new data becomes queryable. Parquet spends heavily in its row-group encoder; Vortex encodes \
     the same data faster, so pipelines clear backlogs and data goes live sooner.";

const SIZE_WHY: &str = "Storage cost and the bytes a query must move both track file size, so a faster format that \
     bloated on disk would trade one bill for another. Vortex holds within a few percent of \
     Parquet's compression ratio — the speed above comes at no size penalty.";

/// Render the landing body from the live read generation.
pub(super) fn showcase_body(generation: &ReadGeneration) -> Markup {
    let groups = generation.groups();

    let random_access = group_by_summary(&groups, |s| matches!(s, Summary::RandomAccess { .. }));
    let compression = group_by_summary(&groups, |s| matches!(s, Summary::Compression { .. }));
    let size = group_by_summary(&groups, |s| matches!(s, Summary::CompressionSize { .. }));
    let clickbench = query_group(&groups, "clickbench");

    html! {
        section.showcase {
            header.showcase-intro {
                p.showcase-eyebrow { "Vortex vs Apache Parquet" }
                h2.showcase-headline {
                    "A columnar format built for the read patterns Parquet wasn't."
                }
            }
            div.claims {
                (random_access_claim(generation, random_access))
                (analytics_claim(generation, clickbench))
                (writes_claim(generation, compression))
                (size_claim(generation, size))
            }
            div.showcase-cta {
                a.show-everything href="/latest" {
                    "Show me everything"
                    span.show-everything-arrow aria-hidden="true" { " →" }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------

/// "Random access" — geomean of the per-pattern Vortex-vs-Parquet speedup
/// (a single no-facet chart, facet `""`).
fn random_access_claim(generation: &ReadGeneration, group: Option<&Group>) -> Markup {
    let facets = group
        .map(|g| facet_geomeans(generation, &g.charts))
        .unwrap_or_default();
    let g = pick_facet(&facets, "");
    let (hero, detail) = match &g {
        Some(f) => (
            mult(f.geomean),
            detail_line(format!("Vortex wins {}/{}", f.wins, f.total)),
        ),
        None => ("—".to_string(), html! {}),
    };
    claim(
        &hero,
        "faster random access",
        detail,
        RANDOM_ACCESS_WHY,
        workload_svg(Workload::RandomAccess),
        more_href(group),
    )
}

/// "Data analytics" — ClickBench, our home turf. One headline pooled across
/// engines, with the per-engine breakdown beneath, to make the cross-engine
/// drop-in story explicit.
fn analytics_claim(generation: &ReadGeneration, group: Option<&Group>) -> Markup {
    let facets = group
        .map(|g| facet_geomeans(generation, &g.charts))
        .unwrap_or_default();
    let combined = pooled_geomean(&facets);
    let hero = combined.map(mult).unwrap_or_else(|| "—".to_string());
    let df = pick_facet(&facets, "datafusion");
    let duck = pick_facet(&facets, "duckdb");
    let detail = html! {
        span.claim-detail {
            "ClickBench"
            @if let Some(f) = df { " · DataFusion " (mult(f.geomean)) }
            @if let Some(f) = duck { " · DuckDB " (mult(f.geomean)) }
        }
    };
    claim(
        &hero,
        "faster data analytics",
        detail,
        ANALYTICS_WHY,
        workload_svg(Workload::Analytics),
        more_href(group),
    )
}

/// "Writes" — the compression encode facet's Vortex-vs-Parquet speedup.
fn writes_claim(generation: &ReadGeneration, group: Option<&Group>) -> Markup {
    let facets = group
        .map(|g| facet_geomeans(generation, &g.charts))
        .unwrap_or_default();
    let g = pick_facet(&facets, "encode");
    let (hero, detail) = match &g {
        Some(f) => (
            mult(f.geomean),
            detail_line(format!(
                "Compression encode · Vortex wins {}/{}",
                f.wins, f.total
            )),
        ),
        None => ("—".to_string(), html! {}),
    };
    claim(
        &hero,
        "faster writes",
        detail,
        WRITES_WHY,
        workload_svg(Workload::Writes),
        more_href(group),
    )
}

/// "Size" — the compressed-size ratio. The synthesis ratio is `parquet/vortex`
/// (smaller-is-better), so Vortex's footprint relative to Parquet is its
/// reciprocal; we present that as "N× the size of Parquet".
fn size_claim(generation: &ReadGeneration, group: Option<&Group>) -> Markup {
    let facets = group
        .map(|g| facet_geomeans(generation, &g.charts))
        .unwrap_or_default();
    let g = pick_facet(&facets, "");
    let (hero, detail) = match &g {
        Some(f) if f.geomean > 0.0 => (
            mult(1.0 / f.geomean),
            detail_line(format!("geomean across {} datasets", f.total)),
        ),
        _ => ("—".to_string(), html! {}),
    };
    claim(
        &hero,
        "the size of Parquet",
        detail,
        SIZE_WHY,
        workload_svg(Workload::Size),
        more_href(group),
    )
}

/// One claim cell: the headline stat (number + label + detail + a "see the
/// proof" link) beside a workload schematic, with the "why it matters" prose
/// spanning the cell beneath.
fn claim(
    hero: &str,
    label: &str,
    detail: Markup,
    why: &str,
    figure: Markup,
    proof_href: Option<String>,
) -> Markup {
    html! {
        article.claim {
            div.claim-head {
                div.claim-stat {
                    span.claim-metric { (hero) }
                    span.claim-label { (label) }
                    (detail)
                    @if let Some(href) = &proof_href {
                        a.claim-proof href=(href) {
                            "See the proof"
                            span.claim-proof-arrow aria-hidden="true" { " →" }
                        }
                    }
                }
                div.claim-figure { (figure) }
            }
            p.claim-why { (why) }
        }
    }
}

/// A small muted detail line beneath the headline.
fn detail_line(text: String) -> Markup {
    html! { span.claim-detail { (text) } }
}

/// Format a multiplier: whole-number for big ratios (`80×`), two decimals for
/// the close ones (`1.30×`).
fn mult(v: f64) -> String {
    if v >= 10.0 {
        format!("{v:.0}×")
    } else {
        format!("{v:.2}×")
    }
}

/// The facet with the given name from a precomputed list.
fn pick_facet<'a>(facets: &'a [FacetGeomean], facet: &str) -> Option<&'a FacetGeomean> {
    facets.iter().find(|f| f.facet == facet)
}

/// Geomean pooled across every facet's items: `exp(Σ nᵢ·ln gᵢ / Σ nᵢ)`, the
/// geomean of all the underlying ratios regardless of which facet produced
/// them. Used for ClickBench's single cross-engine headline.
fn pooled_geomean(facets: &[FacetGeomean]) -> Option<f64> {
    let total: usize = facets.iter().map(|f| f.total).sum();
    if total == 0 {
        return None;
    }
    let log_sum: f64 = facets.iter().map(|f| f.total as f64 * f.geomean.ln()).sum();
    Some((log_sum / total as f64).exp())
}

// ---------------------------------------------------------------------------
// Workload schematics — small monochrome "blueprint" SVGs of the access
// pattern (not the result). Stroke/fill read CSS theme vars so they recolour
// with the page. Shared viewBox keeps the four the same size (small multiples).
// ---------------------------------------------------------------------------

enum Workload {
    RandomAccess,
    Analytics,
    Writes,
    Size,
}

fn workload_svg(kind: Workload) -> Markup {
    PreEscaped(
        match kind {
            Workload::RandomAccess => RANDOM_ACCESS_SVG,
            Workload::Analytics => ANALYTICS_SVG,
            Workload::Writes => WRITES_SVG,
            Workload::Size => SIZE_SVG,
        }
        .to_string(),
    )
}

/// A table grid with a few scattered cells lit — reading individual rows by
/// position.
const RANDOM_ACCESS_SVG: &str = r##"<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Reading scattered individual rows by position">
<g stroke="var(--line-strong)" stroke-width="1">
<rect x="12" y="10" width="96" height="64"/>
<path d="M28 10V74M44 10V74M60 10V74M76 10V74M92 10V74"/>
<path d="M12 26H108M12 42H108M12 58H108"/>
</g>
<g fill="var(--bar)">
<rect x="29" y="11" width="14" height="14"/>
<rect x="77" y="27" width="14" height="14"/>
<rect x="13" y="43" width="14" height="14"/>
<rect x="61" y="59" width="14" height="14"/>
</g>
</svg>"##;

/// Whole columns scanned into an aggregate (Σ) — analytical reads.
const ANALYTICS_SVG: &str = r##"<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Scanning whole columns into an aggregate">
<g stroke="var(--line-strong)" stroke-width="1">
<rect x="12" y="10" width="96" height="64"/>
<path d="M28 10V74M44 10V74M60 10V74M76 10V74M92 10V74"/>
<path d="M12 26H108M12 42H108M12 58H108"/>
</g>
<g fill="var(--bar)">
<rect x="29" y="11" width="14" height="62"/>
<rect x="77" y="11" width="14" height="62"/>
</g>
<g stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M108 42H117"/>
<path d="M113 38l4 4l-4 4"/>
</g>
<text x="128" y="47" fill="var(--muted)" font-family="monospace" font-size="13" text-anchor="middle">&#931;</text>
</svg>"##;

/// Loose rows encoded into a packed columnar file — the write path.
const WRITES_SVG: &str = r##"<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Encoding loose rows into a packed file">
<g stroke="var(--bar)" stroke-width="2.5" stroke-linecap="round">
<path d="M8 24H44"/>
<path d="M8 36H38"/>
<path d="M8 48H44"/>
<path d="M8 60H34"/>
</g>
<g stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M52 42H72"/>
<path d="M66 36l6 6l-6 6"/>
</g>
<rect x="84" y="12" width="48" height="60" stroke="var(--line-strong)" stroke-width="1"/>
<g fill="var(--bar)">
<rect x="86" y="14" width="44" height="12"/>
<rect x="86" y="29" width="44" height="12"/>
<rect x="86" y="44" width="44" height="12"/>
<rect x="86" y="59" width="44" height="11"/>
</g>
</svg>"##;

/// Data compressed onto disk: the raw extent (dashed) squeezed to a smaller
/// file (solid) by inward arrows.
const SIZE_SVG: &str = r##"<svg class="workload-svg" viewBox="0 0 140 84" fill="none" role="img" aria-label="Compressing data onto disk">
<rect x="14" y="22" width="112" height="40" stroke="var(--line-strong)" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="44" y="22" width="52" height="40" fill="var(--bar)"/>
<g stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M24 34l10 8l-10 8"/>
<path d="M116 34l-10 8l10 8"/>
</g>
</svg>"##;

// ---------------------------------------------------------------------------
// Shared helpers (also used by the Current page)
// ---------------------------------------------------------------------------

/// Latest finite value in a series' value array (arrays run oldest-first).
pub(super) fn latest_value(values: &Value) -> Option<f64> {
    values
        .as_array()?
        .iter()
        .rev()
        .find_map(|v| v.as_f64().filter(|n| n.is_finite()))
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
        .find(|g| g.summary.as_ref().is_some_and(&pred))
}

/// The cluster representative for a dataset's query group (e.g. `clickbench`,
/// `tpch`, `tpcds`). Matches the (storage, SF) selection `collect_landing_groups`
/// uses — NVMe-preferred, largest SF first — so showcase deep links land on the
/// exact section id `/latest` emits.
fn query_group<'a>(groups: &'a [Group], dataset: &str) -> Option<&'a Group> {
    let storage_rank = |s: &str| -> usize {
        match s {
            "nvme" => 0,
            "s3" => 1,
            _ => 2,
        }
    };
    let mut best: Option<(usize, f64, &Group)> = None;
    for g in groups {
        let Ok(GroupKey::QueryGroup {
            dataset: d,
            scale_factor,
            storage,
            ..
        }) = GroupKey::from_slug(&g.slug)
        else {
            continue;
        };
        if d != dataset {
            continue;
        }
        let sf = scale_factor
            .as_deref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let sr = storage_rank(&storage);
        if best.is_none_or(|(bsr, bsf, _)| sr < bsr || (sr == bsr && sf > bsf)) {
            best = Some((sr, sf, g));
        }
    }
    best.map(|(_, _, g)| g)
}

/// Deep link from a claim into the Current page's matching group section.
/// [`anchor_for`] keeps the fragment in lockstep with the section id the
/// Current page emits, so the link scrolls to and `:target`-highlights it.
fn more_href(group: Option<&Group>) -> Option<String> {
    group.map(|g| format!("/latest#{}", anchor_for(&g.slug)))
}
