// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Structural diff between a migrated v3 DuckDB and the live v2
//! `/api/metadata` endpoint.
//!
//! Compares group / chart structure only; values aren't compared
//! because v2 converts ns → ms and bytes → MiB on read while v3
//! stores raw and the chart query divides. Group/chart structural
//! equivalence is enough to spot classifier regressions before
//! cutover.
//!
//! The diff distinguishes documented intentional asymmetries (e.g.
//! ratio charts that v3 derives at read time, the legacy
//! `RANDOM ACCESS` placeholder) from regression candidates so a clean
//! run shows only known-good differences and a regression jumps out
//! immediately. See [`INTENTIONAL_ONLY_IN_V2`] and
//! [`INTENTIONAL_ONLY_IN_V3`] for the live list.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use serde::Deserialize;

use crate::classifier::QUERY_SUITES;

/// One row of [`VerifyReport::group_chart_diffs`]: per-group lists of
/// chart names that are missing on either side, split into intentional
/// asymmetries and regression candidates.
#[derive(Debug, Default, Clone)]
pub struct GroupChartDiff {
    pub group: String,
    pub v2_count: usize,
    pub v3_count: usize,
    pub missing_in_v3_intentional: Vec<String>,
    pub missing_in_v3_regression: Vec<String>,
    pub missing_in_v2_intentional: Vec<String>,
    pub missing_in_v2_regression: Vec<String>,
}

impl GroupChartDiff {
    /// True if every chart-name asymmetry between v2 and v3 for this
    /// group is documented as intentional. False means at least one
    /// regression candidate is on the list.
    pub fn is_clean(&self) -> bool {
        self.missing_in_v3_regression.is_empty() && self.missing_in_v2_regression.is_empty()
    }
}

/// Result of one `verify` run.
#[derive(Debug, Default)]
pub struct VerifyReport {
    pub matched_groups: Vec<String>,
    /// Groups that exist in v3 but not v2, where the asymmetry is NOT
    /// on the documented allowlist — counts as a regression.
    pub only_in_v3: Vec<String>,
    /// Groups that exist in v2 but not v3, where the asymmetry is NOT
    /// on the documented allowlist — counts as a regression.
    pub only_in_v2: Vec<String>,
    /// Groups whose v2/v3 asymmetry is on the documented allowlist
    /// (e.g. `Fineweb` in v3, an empty `TPC-H (NVMe) (SF=1000)` fan-out
    /// in v2). Surfaced for the human reader; not a regression.
    pub only_in_v3_intentional: Vec<String>,
    pub only_in_v2_intentional: Vec<String>,
    pub group_chart_diffs: Vec<GroupChartDiff>,
}

impl VerifyReport {
    /// True if every v2 group is represented in v3 *and* every per-
    /// chart-name asymmetry is documented as intentional. The CLI's
    /// exit code reflects this.
    pub fn is_clean(&self) -> bool {
        self.only_in_v2.is_empty()
            && self.only_in_v3.is_empty()
            && self.group_chart_diffs.iter().all(|d| d.is_clean())
    }

    /// Backwards-compatible: were all v2 groups covered by v3?
    pub fn v2_groups_covered(&self) -> bool {
        self.only_in_v2.is_empty()
    }
}

impl std::fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Groups in both v2 and v3:")?;
        for g in &self.matched_groups {
            writeln!(f, "  + {g}")?;
        }
        if !self.only_in_v2.is_empty() {
            writeln!(f, "Groups only in v2 (regression candidates):")?;
            for g in &self.only_in_v2 {
                writeln!(f, "  ✗ {g}")?;
            }
        }
        if !self.only_in_v2_intentional.is_empty() {
            writeln!(f, "Groups only in v2 (documented intentional skip):")?;
            for g in &self.only_in_v2_intentional {
                writeln!(f, "  · {g}")?;
            }
        }
        if !self.only_in_v3.is_empty() {
            writeln!(f, "Groups only in v3 (regression candidates):")?;
            for g in &self.only_in_v3 {
                writeln!(f, "  ✗ {g}")?;
            }
        }
        if !self.only_in_v3_intentional.is_empty() {
            writeln!(f, "Groups only in v3 (documented intentional addition):")?;
            for g in &self.only_in_v3_intentional {
                writeln!(f, "  · {g}")?;
            }
        }
        if !self.group_chart_diffs.is_empty() {
            writeln!(f, "Chart name diffs (per group):")?;
            for d in &self.group_chart_diffs {
                writeln!(
                    f,
                    "  {} : v2={} v3={} (delta={})",
                    d.group,
                    d.v2_count,
                    d.v3_count,
                    d.v3_count as i64 - d.v2_count as i64,
                )?;
                for c in &d.missing_in_v3_regression {
                    writeln!(f, "      ✗ only in v2 (regression candidate): {c}")?;
                }
                for c in &d.missing_in_v2_regression {
                    writeln!(f, "      ✗ only in v3 (regression candidate): {c}")?;
                }
                for c in &d.missing_in_v3_intentional {
                    writeln!(f, "      · only in v2 (documented intentional skip): {c}")?;
                }
                for c in &d.missing_in_v2_intentional {
                    writeln!(
                        f,
                        "      · only in v3 (documented intentional addition): {c}"
                    )?;
                }
            }
        }
        if self.is_clean() {
            writeln!(f, "verify: clean (every asymmetry is documented).")?;
        } else {
            writeln!(f, "verify: regression candidates present (see ✗ above).")?;
        }
        Ok(())
    }
}

/// v2's `/api/metadata` reply — only the fields we need.
#[derive(Debug, Deserialize)]
struct V2Metadata {
    groups: BTreeMap<String, V2GroupMeta>,
}

#[derive(Debug, Deserialize)]
struct V2GroupMeta {
    #[serde(default)]
    charts: Vec<V2ChartMeta>,
}

#[derive(Debug, Deserialize)]
struct V2ChartMeta {
    #[serde(default)]
    name: String,
}

/// Open the migrated DuckDB at `duckdb_path`, fetch `<v2_server>/api/metadata`,
/// and produce a structural diff.
pub fn run(v2_server: &str, duckdb_path: &Path) -> Result<VerifyReport> {
    let v3 = collect_v3_groups(duckdb_path)?;
    let v2 = fetch_v2_metadata(v2_server)?;
    Ok(diff(&v2, &v3))
}

/// Charts present in v2 metadata but intentionally absent from v3.
///
/// Each entry is a `(group, normalized_chart_name)` pair. The
/// normalization matches [`normalize_chart`] so the lookup uses the
/// same key shape the diff produces. Update this list whenever the
/// classifier deliberately drops a v2 chart pattern (e.g. a derived
/// ratio, a placeholder, a deprecated format) so a future regression
/// shows up as a fresh ✗ instead of getting lost in the noise.
const INTENTIONAL_ONLY_IN_V2: &[(&str, &str)] = &[
    // Compression ratios are derived at read time from compression_sizes
    // (joined to itself); the migrator routes them to Skip::DerivedRatio.
    ("Compression", "VORTEX:LANCE RATIO COMPRESS TIME"),
    ("Compression", "VORTEX:LANCE RATIO DECOMPRESS TIME"),
    ("Compression", "VORTEX:PARQUET ZSTD RATIO COMPRESS TIME"),
    ("Compression", "VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME"),
    // Compression-size ratios — same story.
    ("Compression Size", "VORTEX:LANCE SIZE"),
    ("Compression Size", "VORTEX:PARQUET ZSTD SIZE"),
    ("Compression Size", "VORTEX:RAW SIZE"),
    // The legacy 2-part `random-access/<format>-tokio-local-disk` records
    // render in v2 under a "RANDOM ACCESS" placeholder chart. The
    // migrator recovers their *values* under `dataset="taxi"` (see
    // `bin_random_access`) instead of carrying the placeholder name
    // forward, so v3 has a "TAXI" chart and v2 has "RANDOM ACCESS".
    // Both sides are documented intentional asymmetries.
    ("Random Access", "RANDOM ACCESS"),
];

/// Charts emitted by the migrator that v2 intentionally doesn't render.
///
/// Pair shape matches [`INTENTIONAL_ONLY_IN_V2`].
const INTENTIONAL_ONLY_IN_V3: &[(&str, &str)] = &[
    // `vortex-compact` size rows come in via `migrate_file_sizes` (the
    // file-sizes-*.json.gz path). v2 never rendered the format because
    // its `getGroup` didn't recognize the `vortex-compact` suite.
    ("Compression Size", "VORTEX COMPACT SIZE"),
    // 2-part legacy random-access records (per the "RANDOM ACCESS"
    // entry in INTENTIONAL_ONLY_IN_V2 above) are recovered in v3 as
    // dataset="taxi". v2 never had a chart by that name in Random
    // Access — its taxi dataset always rode the `taxi/correlated`
    // and `taxi/uniform` 4-part patterns.
    ("Random Access", "TAXI"),
];

/// Groups intentionally surfaced by v3 but skipped by v2's metadata.
///
/// `fineweb` is on `V3_QUERY_SUITES` because the live CI workflow still
/// emits fineweb measurements; v2's `getGroup` marks the suite
/// `skip: true` so the v2 server never builds metadata for it.
const INTENTIONAL_ONLY_IN_V3_GROUPS: &[&str] = &["Fineweb"];

/// Groups intentionally listed by v2 metadata that v3 doesn't materialize.
///
/// v2's `FAN_OUT_GROUPS` registers TPC-H and TPC-DS group names for
/// every `(storage, scale_factor)` pair the UI knows about, even when
/// no records exist (the chart list comes back empty). The migrator
/// only writes a group when matching rows exist, so empty fan-outs
/// don't appear in v3 — which is the intended behavior.
const INTENTIONAL_ONLY_IN_V2_GROUPS: &[&str] = &[
    "TPC-DS (NVMe) (SF=10)",
    "TPC-H (NVMe) (SF=1000)",
    "TPC-H (S3) (SF=1000)",
];

fn is_intentional(table: &[(&str, &str)], group: &str, chart: &str) -> bool {
    table
        .iter()
        .any(|(g, c)| *g == group && normalize_chart(c) == chart)
}

fn collect_v3_groups(duckdb_path: &Path) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let conn = Connection::open(duckdb_path)
        .with_context(|| format!("opening DuckDB at {}", duckdb_path.display()))?;
    let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    // query_measurements: chart per (dataset, query_idx); group per
    // (dataset, dataset_variant, scale_factor, storage). We want v2
    // group display names so the verifier can compare apples to
    // apples, so we re-format them here using the same suite table.
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, dataset_variant, scale_factor, storage, query_idx
          FROM query_measurements
         GROUP BY dataset, dataset_variant, scale_factor, storage, query_idx
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i32>(4)?,
        ))
    })?;
    for row in rows {
        let (dataset, _variant, sf, storage, query_idx) = row?;
        let group_name = display_query_group(&dataset, sf.as_deref(), &storage);
        let chart_name = chart_name_query(&dataset, query_idx);
        groups
            .entry(group_name)
            .or_default()
            .insert(normalize_chart(&chart_name));
    }

    // compression_times: group "Compression", charts per dataset.
    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, format, op
          FROM compression_times
         GROUP BY dataset, format, op
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    for row in rows {
        let (dataset, format, op) = row?;
        let chart = chart_name_compression_time(&format, &op, &dataset);
        groups
            .entry("Compression".to_string())
            .or_default()
            .insert(normalize_chart(&chart));
    }

    let mut stmt = conn.prepare(
        r#"
        SELECT dataset, format
          FROM compression_sizes
         GROUP BY dataset, format
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (_dataset, format) = row?;
        let chart = chart_name_compression_size(&format);
        groups
            .entry("Compression Size".to_string())
            .or_default()
            .insert(normalize_chart(&chart));
    }

    let mut stmt = conn.prepare(
        r#"
        SELECT DISTINCT dataset
          FROM random_access_times
        "#,
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        let dataset = row?;
        groups
            .entry("Random Access".to_string())
            .or_default()
            .insert(normalize_chart(&dataset));
    }

    Ok(groups)
}

fn fetch_v2_metadata(server: &str) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let url = format!("{}/api/metadata", server.trim_end_matches('/'));
    let body = reqwest::blocking::get(&url)
        .with_context(|| format!("GET {url}"))?
        .error_for_status()
        .with_context(|| format!("non-2xx from {url}"))?
        .json::<V2Metadata>()
        .with_context(|| format!("parsing {url} as v2 /api/metadata"))?;
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, group) in body.groups {
        let charts = group
            .charts
            .into_iter()
            .map(|c| normalize_chart(&c.name))
            .collect();
        out.insert(name, charts);
    }
    Ok(out)
}

fn diff(
    v2: &BTreeMap<String, BTreeSet<String>>,
    v3: &BTreeMap<String, BTreeSet<String>>,
) -> VerifyReport {
    let mut report = VerifyReport::default();
    let v2_keys: BTreeSet<&String> = v2.keys().collect();
    let v3_keys: BTreeSet<&String> = v3.keys().collect();

    // Group-level membership. An entry that's "only in v2" but with
    // zero charts (e.g. a pre-registered FAN_OUT_GROUPS placeholder)
    // and that's on the documented allowlist isn't a regression.
    for g in v2_keys.intersection(&v3_keys) {
        report.matched_groups.push((**g).clone());
        let v2_charts = &v2[*g];
        let v3_charts = &v3[*g];
        let only_v3 = v3_charts.difference(v2_charts).cloned().collect::<Vec<_>>();
        let only_v2 = v2_charts.difference(v3_charts).cloned().collect::<Vec<_>>();
        if only_v3.is_empty() && only_v2.is_empty() {
            continue;
        }
        let mut row = GroupChartDiff {
            group: (**g).clone(),
            v2_count: v2_charts.len(),
            v3_count: v3_charts.len(),
            ..Default::default()
        };
        for c in only_v2 {
            if is_intentional(INTENTIONAL_ONLY_IN_V2, g, &c) {
                row.missing_in_v3_intentional.push(c);
            } else {
                row.missing_in_v3_regression.push(c);
            }
        }
        for c in only_v3 {
            if is_intentional(INTENTIONAL_ONLY_IN_V3, g, &c) {
                row.missing_in_v2_intentional.push(c);
            } else {
                row.missing_in_v2_regression.push(c);
            }
        }
        row.missing_in_v3_intentional.sort();
        row.missing_in_v3_regression.sort();
        row.missing_in_v2_intentional.sort();
        row.missing_in_v2_regression.sort();
        report.group_chart_diffs.push(row);
    }
    for g in v3_keys.difference(&v2_keys) {
        // Group exists only in v3. If documented (e.g. fineweb), shunt
        // it to the intentional list; otherwise it's a regression
        // candidate.
        if INTENTIONAL_ONLY_IN_V3_GROUPS.contains(&g.as_str()) {
            report.only_in_v3_intentional.push((**g).clone());
        } else {
            report.only_in_v3.push((**g).clone());
        }
    }
    for g in v2_keys.difference(&v3_keys) {
        // Group exists only in v2. Documented empty fan-outs (the
        // hard-coded TPC-H/TPC-DS slots in v2's `FAN_OUT_GROUPS`) don't
        // count as regressions; surface them as intentional.
        let charts = &v2[*g];
        let documented_empty =
            INTENTIONAL_ONLY_IN_V2_GROUPS.contains(&g.as_str()) && charts.is_empty();
        if documented_empty {
            report.only_in_v2_intentional.push((**g).clone());
        } else {
            report.only_in_v2.push((**g).clone());
        }
    }
    report.matched_groups.sort();
    report.only_in_v3.sort();
    report.only_in_v3_intentional.sort();
    report.only_in_v2.sort();
    report.only_in_v2_intentional.sort();
    report
        .group_chart_diffs
        .sort_by(|a, b| a.group.cmp(&b.group));
    report
}

fn display_query_group(dataset: &str, scale_factor: Option<&str>, storage: &str) -> String {
    let suite = QUERY_SUITES
        .iter()
        .find(|s| s.prefix.eq_ignore_ascii_case(dataset))
        .copied();
    match suite {
        Some(suite) if suite.fan_out => {
            let storage_disp = match storage {
                "s3" | "S3" => "S3",
                _ => "NVMe",
            };
            let sf = scale_factor.unwrap_or("1");
            format!("{} ({}) (SF={})", suite.display_name, storage_disp, sf)
        }
        Some(suite) => suite.display_name.to_string(),
        None => format!("{dataset} ({storage})"),
    }
}

fn chart_name_query(dataset: &str, query_idx: i32) -> String {
    let suite = QUERY_SUITES
        .iter()
        .find(|s| s.prefix.eq_ignore_ascii_case(dataset))
        .copied();
    match suite {
        Some(suite) => format!("{} Q{}", suite.query_prefix, query_idx),
        None => format!("{} Q{}", dataset.to_uppercase(), query_idx),
    }
}

fn chart_name_compression_time(format: &str, op: &str, _dataset: &str) -> String {
    // Re-derive the v2 chart name (the metric, not the dataset) so we
    // can compare. v2's chart axis is the metric; series is the
    // dataset. v3 inverts that. For structural comparison, we project
    // back to v2's per-chart key.
    match (format, op) {
        ("vortex-file-compressed", "encode") => "COMPRESS TIME".into(),
        ("vortex-file-compressed", "decode") => "DECOMPRESS TIME".into(),
        ("parquet", "encode") => "PARQUET RS ZSTD COMPRESS TIME".into(),
        ("parquet", "decode") => "PARQUET RS ZSTD DECOMPRESS TIME".into(),
        ("lance", "encode") => "LANCE COMPRESS TIME".into(),
        ("lance", "decode") => "LANCE DECOMPRESS TIME".into(),
        _ => format!("{} {} TIME", format.to_uppercase(), op.to_uppercase()),
    }
}

fn chart_name_compression_size(format: &str) -> String {
    match format {
        "vortex-file-compressed" => "VORTEX SIZE".into(),
        "parquet" => "PARQUET SIZE".into(),
        "lance" => "LANCE SIZE".into(),
        _ => format!("{} SIZE", format.to_uppercase()),
    }
}

/// Strip casing and `_-` differences between v2 and v3 chart names.
/// v2 displays uppercase; v3 stores raw values. Comparing in this
/// canonical form is enough for structural verification.
fn normalize_chart(s: &str) -> String {
    s.trim()
        .to_uppercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(charts: &[&str]) -> BTreeSet<String> {
        charts.iter().map(|s| normalize_chart(s)).collect()
    }

    #[test]
    fn normalize_chart_canonicalizes() {
        assert_eq!(normalize_chart("taxi/take"), "TAXI/TAKE");
        assert_eq!(normalize_chart("TAXI/TAKE"), "TAXI/TAKE");
        assert_eq!(normalize_chart("tpc-h q1"), "TPC H Q1");
        assert_eq!(normalize_chart("tpc h q1"), "TPC H Q1");
    }

    #[test]
    fn display_query_group_handles_fan_out() {
        assert_eq!(
            display_query_group("tpch", Some("10"), "s3"),
            "TPC-H (S3) (SF=10)"
        );
        assert_eq!(
            display_query_group("tpch", Some("100"), "nvme"),
            "TPC-H (NVMe) (SF=100)"
        );
        assert_eq!(
            display_query_group("clickbench", None, "nvme"),
            "Clickbench"
        );
    }

    #[test]
    fn diff_clean_when_only_documented_asymmetries() {
        // v2 has "RANDOM ACCESS" placeholder; v3 has "TAXI" recovered.
        // Both are on the intentional allowlist.
        let mut v2 = BTreeMap::new();
        v2.insert(
            "Random Access".to_string(),
            group(&["TAXI/CORRELATED", "TAXI/UNIFORM", "RANDOM ACCESS"]),
        );
        let mut v3 = BTreeMap::new();
        v3.insert(
            "Random Access".to_string(),
            group(&["TAXI/CORRELATED", "TAXI/UNIFORM", "TAXI"]),
        );
        let report = diff(&v2, &v3);
        assert!(report.is_clean(), "expected clean, got: {report}");
        let row = &report.group_chart_diffs[0];
        assert_eq!(row.group, "Random Access");
        assert_eq!(row.missing_in_v3_intentional, vec!["RANDOM ACCESS"]);
        assert_eq!(row.missing_in_v2_intentional, vec!["TAXI"]);
        assert!(row.missing_in_v3_regression.is_empty());
        assert!(row.missing_in_v2_regression.is_empty());
    }

    #[test]
    fn diff_flags_undocumented_only_in_v2_chart() {
        let mut v2 = BTreeMap::new();
        v2.insert("Random Access".to_string(), group(&["NEW CHART NAME"]));
        let mut v3 = BTreeMap::new();
        v3.insert("Random Access".to_string(), group(&[]));
        let report = diff(&v2, &v3);
        assert!(!report.is_clean(), "expected regression, got clean");
        let row = &report.group_chart_diffs[0];
        assert_eq!(row.missing_in_v3_regression, vec!["NEW CHART NAME"]);
    }

    #[test]
    fn diff_flags_undocumented_only_in_v3_chart() {
        let mut v2 = BTreeMap::new();
        v2.insert("Random Access".to_string(), group(&[]));
        let mut v3 = BTreeMap::new();
        v3.insert("Random Access".to_string(), group(&["MYSTERY CHART"]));
        let report = diff(&v2, &v3);
        assert!(!report.is_clean(), "expected regression, got clean");
        let row = &report.group_chart_diffs[0];
        assert_eq!(row.missing_in_v2_regression, vec!["MYSTERY CHART"]);
    }

    #[test]
    fn diff_documented_empty_fan_out_group_not_a_regression() {
        // v2 metadata always lists `TPC-H (NVMe) (SF=1000)` with zero
        // charts (the FAN_OUT_GROUPS hard-coding); v3 doesn't
        // materialize empty groups. The verifier should accept this
        // and route the asymmetry to the intentional list.
        let mut v2 = BTreeMap::new();
        v2.insert("TPC-H (NVMe) (SF=1000)".to_string(), group(&[]));
        v2.insert("Clickbench".to_string(), group(&["CLICKBENCH Q0"]));
        let mut v3 = BTreeMap::new();
        v3.insert("Clickbench".to_string(), group(&["CLICKBENCH Q0"]));
        let report = diff(&v2, &v3);
        assert!(
            report.is_clean(),
            "documented empty fan-out should not be a regression: {report}"
        );
        assert!(report.only_in_v2.is_empty());
        assert_eq!(
            report.only_in_v2_intentional,
            vec!["TPC-H (NVMe) (SF=1000)"]
        );
    }

    #[test]
    fn diff_undocumented_only_in_v2_group_is_a_regression() {
        let mut v2 = BTreeMap::new();
        v2.insert("Brand New Group".to_string(), group(&["X"]));
        let v3 = BTreeMap::new();
        let report = diff(&v2, &v3);
        assert!(!report.is_clean());
        assert_eq!(report.only_in_v2, vec!["Brand New Group"]);
    }

    #[test]
    fn diff_documented_only_in_v3_group_not_a_regression() {
        // `Fineweb` is on the v3 query-suite allowlist (CI still emits
        // fineweb data); v2's `getGroup` skips fineweb so its metadata
        // never lists the group. The verifier should accept this and
        // surface the asymmetry as intentional.
        let v2 = BTreeMap::new();
        let mut v3 = BTreeMap::new();
        v3.insert("Fineweb".to_string(), group(&["FINEWEB Q0"]));
        let report = diff(&v2, &v3);
        assert!(report.is_clean(), "documented v3-only group: {report}");
        assert!(report.only_in_v3.is_empty());
        assert_eq!(report.only_in_v3_intentional, vec!["Fineweb"]);
    }

    #[test]
    fn diff_undocumented_only_in_v3_group_is_a_regression() {
        let v2 = BTreeMap::new();
        let mut v3 = BTreeMap::new();
        v3.insert("Mystery Group".to_string(), group(&["X"]));
        let report = diff(&v2, &v3);
        assert!(!report.is_clean());
        assert_eq!(report.only_in_v3, vec!["Mystery Group"]);
    }
}
