// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Editorial group descriptions, ported from v2.
//!
//! These strings are the source of truth for the hover tooltip rendered on
//! every `<details>` group title on the landing page and on the
//! `/group/{slug}` permalink. They are deliberately editorial and
//! hand-maintained — derived from the group *name*, not from the database —
//! so that adding a new group's blurb is a one-line edit here rather than a
//! schema or ingest change.
//!
//! Source of truth in v2 (kept verbatim where applicable):
//! - `benchmarks-website/src/utils.js` — `getBenchmarkDescription`
//! - `benchmarks-website/src/config.js` — `BENCHMARK_DESCRIPTIONS`,
//!   `QUERY_SUITES.description`
//!
//! TPC-H / TPC-DS fan out by storage and scale factor; the description is
//! synthesised from the parsed name so we don't have to hand-maintain one
//! entry per `(storage, sf)` pair.

/// Look up a short editorial description for a group display name. Returns
/// `None` when the group has no canonical description (e.g. vector-search
/// groups) — callers render the title without a tooltip in that case.
pub fn group_description(name: &str) -> Option<String> {
    if let Some(d) = tpc_description(name) {
        return Some(d);
    }
    static_description(name).map(str::to_string)
}

/// Hard-coded, name-keyed descriptions for the non-fan-out groups. These
/// match v2 verbatim where v2 had a description; new strings here should
/// match the wording style v2 set.
fn static_description(name: &str) -> Option<&'static str> {
    match name {
        "Random Access" => Some(
            "Point lookups — selecting specific rows by position from an NVMe file. \
             What feature stores, vector retrieval, and per-record serving actually do.",
        ),
        "Compression Speed" => Some(
            "Encode and decode throughput (MB/s) for Vortex vs Parquet (zstd page \
             compression). Encode gates ingestion; decode gates every scan after.",
        ),
        "Compression Size" => Some(
            "Compressed file size per format across a fixed set of datasets. A faster \
             format that bloats on disk just trades one bill for another.",
        ),
        "Clickbench" => Some(
            "ClickHouse's 43-query analytical suite over real web-analytics data — the \
             field's standard test for single-table scans, filters, and aggregations.",
        ),
        "Statistical and Population Genetics" => Some(
            "Population-genetics queries over the gnomAD dataset — DuckDB-only, exercising \
             the deeply-nested array operations real genomics pipelines run on.",
        ),
        "PolarSignals Profiling" => Some(
            "Scan-layer benchmark modeled on PolarSignals/Parca: projection and filter \
             pushdown over deeply-nested profile schemas — the shape continuous-profiling \
             backends actually read.",
        ),
        _ => None,
    }
}

/// Derive a description for `TPC-H (NVMe|S3) (SF=N)` and `TPC-DS (NVMe) (SF=N)`
/// group names. The shape is fixed because [`crate::api::groups::group_name_query`]
/// emits exactly this format for tpch/tpcds. Returns `None` for any name that
/// does not start with `TPC-H ` or `TPC-DS `.
fn tpc_description(name: &str) -> Option<String> {
    let parts = if let Some(rest) = name.strip_prefix("TPC-H ") {
        Some(("TPC-H", rest))
    } else {
        name.strip_prefix("TPC-DS ").map(|rest| ("TPC-DS", rest))
    };
    let (suite, rest) = parts?;
    let storage = if rest.starts_with("(NVMe)") {
        "nvme"
    } else if rest.starts_with("(S3)") {
        "s3"
    } else {
        return None;
    };
    let sf = parse_sf(rest)?;
    Some(format_tpc(suite, storage, &sf))
}

/// Pull `SF=N` (digits only) out of strings like `(NVMe) (SF=10)`. Returns
/// `None` if no `SF=` substring or the digits don't parse.
fn parse_sf(s: &str) -> Option<String> {
    let after = s.split_once("SF=")?.1;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        None
    } else {
        Some(digits)
    }
}

/// Render a TPC blurb. Storage label comes from the parsed group name; the
/// dataset size annotation only renders for TPC-H (its scale-factor → bytes
/// mapping is canonical; TPC-DS schemas vary too much per query to annotate).
fn format_tpc(suite: &str, storage: &str, sf: &str) -> String {
    let storage_phrase = match storage {
        "nvme" => "on local NVMe",
        "s3" => "against S3",
        _ => "on local NVMe",
    };
    let bytes = match sf {
        "1" => Some("1 GB"),
        "10" => Some("10 GB"),
        "100" => Some("100 GB"),
        "1000" => Some("1 TB"),
        _ => None,
    };
    match (suite, bytes) {
        ("TPC-H", Some(b)) => format!(
            "TPC-H — 22 analytical queries against the canonical OLAP star schema — \
             at scale factor {sf} (~{b}), {storage_phrase}."
        ),
        ("TPC-H", None) => format!(
            "TPC-H — 22 analytical queries against the canonical OLAP star schema — \
             at scale factor {sf}, {storage_phrase}."
        ),
        ("TPC-DS", _) => format!(
            "TPC-DS — the broader 99-query analytical suite (larger schemas, skewed \
             distributions) — at scale factor {sf}, {storage_phrase}."
        ),
        _ => format!("{suite} benchmark queries at scale factor {sf}, {storage_phrase}."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_descriptions_present_for_known_groups() {
        for name in [
            "Random Access",
            "Compression Speed",
            "Compression Size",
            "Clickbench",
            "Statistical and Population Genetics",
            "PolarSignals Profiling",
        ] {
            let d = group_description(name).expect("description present");
            assert!(!d.is_empty(), "{name}: description is empty");
        }
    }

    #[test]
    fn tpch_descriptions_carry_scale_bytes() {
        // Each TPC-H blurb names the suite shape (22 analytical queries, OLAP) and
        // annotates the dataset size for the SF.
        let d = group_description("TPC-H (NVMe) (SF=1)").expect("present");
        assert!(d.contains("22 analytical queries"), "shape: {d}");
        assert!(d.contains("(~1 GB)"), "size: {d}");
        assert!(d.contains("on local NVMe"), "storage: {d}");

        let d = group_description("TPC-H (S3) (SF=1000)").expect("present");
        assert!(d.contains("(~1 TB)"), "size: {d}");
        assert!(d.contains("against S3"), "storage: {d}");
    }

    #[test]
    fn tpcds_descriptions_omit_scale_bytes() {
        // TPC-DS blurbs name the suite shape (99 queries, broader/skewed) but do
        // not annotate scale-bytes (mapping isn't canonical the way TPC-H's is).
        let d = group_description("TPC-DS (NVMe) (SF=1)").expect("present");
        assert!(d.contains("99-query"), "shape: {d}");
        assert!(d.contains("on local NVMe"), "storage: {d}");
        assert!(!d.contains("GB") && !d.contains("TB"), "no bytes: {d}");
    }

    #[test]
    fn unknown_groups_have_no_description() {
        assert_eq!(group_description("cohere-large-10m / partitioned"), None);
        assert_eq!(group_description("Made-up benchmark"), None);
    }

    #[test]
    fn malformed_tpc_names_fall_through() {
        // No `(NVMe)` / `(S3)` prefix → not matched.
        assert_eq!(group_description("TPC-H something else"), None);
        // SF= without digits → not matched.
        assert_eq!(group_description("TPC-H (NVMe) (SF=)"), None);
    }
}
