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

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use duckdb::Connection;
use serde::Deserialize;

use crate::classifier::QUERY_SUITES;

/// Result of one `verify` run.
#[derive(Debug, Default)]
pub struct VerifyReport {
    pub matched_groups: Vec<String>,
    pub only_in_v3: Vec<String>,
    pub only_in_v2: Vec<String>,
    pub chart_diffs: Vec<ChartDiff>,
}

/// One group's chart-count divergence between v2 and v3, captured when the
/// group is structurally present on both sides but the counts differ.
#[derive(Debug, Clone)]
pub struct ChartDiff {
    pub group: String,
    pub v2_count: usize,
    pub v3_count: usize,
}

impl VerifyReport {
    /// True if every v2 group is represented in v3. The CLI's exit
    /// code reflects this.
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
                writeln!(f, "  - {g}")?;
            }
        }
        if !self.only_in_v3.is_empty() {
            writeln!(f, "Groups only in v3:")?;
            for g in &self.only_in_v3 {
                writeln!(f, "  + {g}")?;
            }
        }
        if !self.chart_diffs.is_empty() {
            writeln!(f, "Chart count diffs:")?;
            for d in &self.chart_diffs {
                writeln!(
                    f,
                    "  {} : v2={} v3={} (delta={})",
                    d.group,
                    d.v2_count,
                    d.v3_count,
                    d.v3_count as i64 - d.v2_count as i64,
                )?;
            }
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
    for g in v2_keys.intersection(&v3_keys) {
        report.matched_groups.push((**g).clone());
        let v2_charts = &v2[*g];
        let v3_charts = &v3[*g];
        if v2_charts.len() != v3_charts.len() {
            report.chart_diffs.push(ChartDiff {
                group: (**g).clone(),
                v2_count: v2_charts.len(),
                v3_count: v3_charts.len(),
            });
        }
    }
    for g in v3_keys.difference(&v2_keys) {
        report.only_in_v3.push((**g).clone());
    }
    for g in v2_keys.difference(&v3_keys) {
        report.only_in_v2.push((**g).clone());
    }
    report.matched_groups.sort();
    report.only_in_v3.sort();
    report.only_in_v2.sort();
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
}
