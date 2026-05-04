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
        "Random Access" => {
            Some("Tests performance of selecting arbitrary row indices from a file on NVMe storage")
        }
        "Compression" => Some(
            "Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet \
             files (with zstd page compression)",
        ),
        "Compression Size" => Some(
            "Compares compressed file sizes and compression ratios across different encoding \
             strategies",
        ),
        "Clickbench" => Some(
            "ClickHouse's analytical benchmark suite testing real-world query patterns on web \
             analytics data",
        ),
        "Statistical and Population Genetics" => {
            Some("A suite of Statistical and Population genetics queries using the gnomAD dataset")
        }
        "PolarSignals Profiling" => Some(
            "Profiling data benchmark modeled on PolarSignals/Parca, exercising scan-layer \
             performance with projection and filter pushdown on deeply nested schemas",
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

/// Render the v2-compatible TPC blurb. Storage label comes from the parsed
/// group name; scale-bytes annotation only renders for TPC-H (TPC-DS in v2
/// did not annotate scale bytes).
fn format_tpc(suite: &str, storage: &str, sf: &str) -> String {
    let storage_phrase = match storage {
        "nvme" => "on local NVMe storage",
        "s3" => "against S3 storage",
        _ => "on local NVMe storage",
    };
    let bytes = match sf {
        "1" => Some("1GB"),
        "10" => Some("10GB"),
        "100" => Some("100GB"),
        "1000" => Some("1TB"),
        _ => None,
    };
    match (suite, bytes) {
        ("TPC-H", Some(b)) => {
            format!("TPC-H benchmark queries {storage_phrase} at SF={sf} (~{b} of data)",)
        }
        ("TPC-H", None) => format!("TPC-H benchmark queries {storage_phrase} at SF={sf}"),
        ("TPC-DS", _) => format!("TPC-DS benchmark queries {storage_phrase} at SF={sf}"),
        _ => format!("{suite} benchmark queries {storage_phrase} at SF={sf}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_descriptions_match_v2() {
        assert_eq!(
            group_description("Random Access").as_deref(),
            Some(
                "Tests performance of selecting arbitrary row indices from a file on NVMe storage"
            ),
        );
        assert_eq!(
            group_description("Compression").as_deref(),
            Some(
                "Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet \
                 files (with zstd page compression)",
            ),
        );
        assert_eq!(
            group_description("Compression Size").as_deref(),
            Some(
                "Compares compressed file sizes and compression ratios across different encoding \
                 strategies",
            ),
        );
        assert_eq!(
            group_description("Clickbench").as_deref(),
            Some(
                "ClickHouse's analytical benchmark suite testing real-world query patterns on \
                 web analytics data",
            ),
        );
        assert_eq!(
            group_description("Statistical and Population Genetics").as_deref(),
            Some("A suite of Statistical and Population genetics queries using the gnomAD dataset",),
        );
        assert_eq!(
            group_description("PolarSignals Profiling").as_deref(),
            Some(
                "Profiling data benchmark modeled on PolarSignals/Parca, exercising scan-layer \
                 performance with projection and filter pushdown on deeply nested schemas",
            ),
        );
    }

    #[test]
    fn tpch_descriptions_carry_scale_bytes() {
        assert_eq!(
            group_description("TPC-H (NVMe) (SF=1)").as_deref(),
            Some("TPC-H benchmark queries on local NVMe storage at SF=1 (~1GB of data)"),
        );
        assert_eq!(
            group_description("TPC-H (S3) (SF=10)").as_deref(),
            Some("TPC-H benchmark queries against S3 storage at SF=10 (~10GB of data)"),
        );
        assert_eq!(
            group_description("TPC-H (NVMe) (SF=100)").as_deref(),
            Some("TPC-H benchmark queries on local NVMe storage at SF=100 (~100GB of data)"),
        );
        assert_eq!(
            group_description("TPC-H (S3) (SF=1000)").as_deref(),
            Some("TPC-H benchmark queries against S3 storage at SF=1000 (~1TB of data)"),
        );
    }

    #[test]
    fn tpcds_descriptions_omit_scale_bytes() {
        assert_eq!(
            group_description("TPC-DS (NVMe) (SF=1)").as_deref(),
            Some("TPC-DS benchmark queries on local NVMe storage at SF=1"),
        );
        assert_eq!(
            group_description("TPC-DS (NVMe) (SF=10)").as_deref(),
            Some("TPC-DS benchmark queries on local NVMe storage at SF=10"),
        );
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
