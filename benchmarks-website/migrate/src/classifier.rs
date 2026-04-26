// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bug-for-bug port of v2's `getGroup`, `formatQuery`, and
//! `normalizeChartName` from `benchmarks-website/server.js`, plus the
//! mapping from v2 group + name pattern to a v3 fact-table bin.
//!
//! The v2 classifier was the source of truth for what historical
//! records mean. It groups records by name prefix into one of:
//! "Random Access", "Compression", "Compression Size", or one of the
//! SQL query suites (with optional fan-out by storage and scale
//! factor for TPC-H/TPC-DS). This module reproduces that logic and
//! then hops to a v3 fact-table bin, since v3 stores dim values as
//! columns instead of name fragments.
//!
//! Engine and format strings stored in v3 columns are pulled from the
//! raw, pre-rename v2 record name. v2's `ENGINE_RENAMES` was a v2
//! read-time UI concern (e.g. `vortex-file-compressed` rendered as
//! `vortex` and `parquet-tokio-local-disk` rendered as `parquet-nvme`).
//! v3 stores canonical `Format::name()` strings to match what the v3
//! live emitter writes, so historical and live records share series.

use crate::v2::V2Record;
use crate::v2::dataset_scale_factor;

/// Static port of v2's `QUERY_SUITES`.
pub const QUERY_SUITES: &[QuerySuite] = &[
    QuerySuite {
        prefix: "clickbench",
        display_name: "Clickbench",
        query_prefix: "CLICKBENCH",
        dataset_key: None,
        fan_out: false,
        skip: false,
    },
    QuerySuite {
        prefix: "statpopgen",
        display_name: "Statistical and Population Genetics",
        query_prefix: "STATPOPGEN",
        dataset_key: None,
        fan_out: false,
        skip: false,
    },
    QuerySuite {
        prefix: "polarsignals",
        display_name: "PolarSignals Profiling",
        query_prefix: "POLARSIGNALS",
        dataset_key: None,
        fan_out: false,
        skip: false,
    },
    QuerySuite {
        prefix: "tpch",
        display_name: "TPC-H",
        query_prefix: "TPC-H",
        dataset_key: Some("tpch"),
        fan_out: true,
        skip: false,
    },
    QuerySuite {
        prefix: "tpcds",
        display_name: "TPC-DS",
        query_prefix: "TPC-DS",
        dataset_key: Some("tpcds"),
        fan_out: true,
        skip: false,
    },
    QuerySuite {
        prefix: "fineweb",
        display_name: "Fineweb",
        query_prefix: "FINEWEB",
        dataset_key: None,
        fan_out: false,
        skip: true,
    },
];

/// Static port of v2's `ENGINE_RENAMES`. Applied to the "series" half
/// of a benchmark name (the part after the first `/`) before splitting
/// on `:` into engine/format. Order doesn't matter — keys are unique.
const ENGINE_RENAMES: &[(&str, &str)] = &[
    ("datafusion:vortex-file-compressed", "datafusion:vortex"),
    ("datafusion:parquet", "datafusion:parquet"),
    ("datafusion:arrow", "datafusion:in-memory-arrow"),
    ("datafusion:lance", "datafusion:lance"),
    ("datafusion:vortex-compact", "datafusion:vortex-compact"),
    ("duckdb:vortex-file-compressed", "duckdb:vortex"),
    ("duckdb:parquet", "duckdb:parquet"),
    ("duckdb:duckdb", "duckdb:duckdb"),
    ("duckdb:vortex-compact", "duckdb:vortex-compact"),
    ("vortex-tokio-local-disk", "vortex-nvme"),
    ("vortex-compact-tokio-local-disk", "vortex-compact-nvme"),
    ("lance-tokio-local-disk", "lance-nvme"),
    ("parquet-tokio-local-disk", "parquet-nvme"),
    ("lance", "lance"),
];

/// One entry of `QUERY_SUITES`.
#[derive(Debug, Clone, Copy)]
pub struct QuerySuite {
    pub prefix: &'static str,
    pub display_name: &'static str,
    pub query_prefix: &'static str,
    pub dataset_key: Option<&'static str>,
    pub fan_out: bool,
    pub skip: bool,
}

/// Group a v2 record falls into. Mirrors `getGroup` in `server.js`,
/// including the fan-out group naming for TPC-H/TPC-DS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V2Group {
    RandomAccess,
    Compression,
    CompressionSize,
    Query {
        suite_index: usize,
        /// `Some` for fan-out suites only.
        storage: Option<String>,
        /// `Some` for fan-out suites only.
        scale_factor: Option<String>,
    },
}

impl V2Group {
    /// Display name as v2 served it from `/api/metadata`.
    pub fn display_name(&self) -> String {
        match self {
            V2Group::RandomAccess => "Random Access".into(),
            V2Group::Compression => "Compression".into(),
            V2Group::CompressionSize => "Compression Size".into(),
            V2Group::Query {
                suite_index,
                storage,
                scale_factor,
            } => {
                let suite = &QUERY_SUITES[*suite_index];
                if let (Some(storage), Some(sf)) = (storage, scale_factor) {
                    format!("{} ({}) (SF={})", suite.display_name, storage, sf)
                } else {
                    suite.display_name.to_string()
                }
            }
        }
    }
}

/// Apply v2's `ENGINE_RENAMES`. Reproduces the JS `rename`:
/// `RENAMES[s.toLowerCase()] || RENAMES[s] || s`.
pub fn rename_engine(s: &str) -> String {
    let lower = s.to_lowercase();
    for (k, v) in ENGINE_RENAMES {
        if *k == lower {
            return (*v).to_string();
        }
    }
    for (k, v) in ENGINE_RENAMES {
        if *k == s {
            return (*v).to_string();
        }
    }
    s.to_string()
}

/// Faithful port of v2's `formatQuery`: maps `clickbench_q07` →
/// `"CLICKBENCH Q7"`. Returns the original (uppercased,
/// `-` and `_` replaced with spaces) when no suite matches.
pub fn format_query(q: &str) -> String {
    let lower = q.to_lowercase();
    for suite in QUERY_SUITES {
        if suite.skip {
            continue;
        }
        let prefix = suite.prefix;
        if let Some(rest) = lower.strip_prefix(prefix)
            && let Some(idx) = parse_query_index(rest)
        {
            return format!("{} Q{}", suite.query_prefix, idx);
        }
    }
    let mut out = q.to_uppercase();
    out = out.replace(['_', '-'], " ");
    out
}

/// Parse the `_q07` / ` q7` / `q42` tail used by `format_query`.
/// Returns the integer query index if the tail matches the v2 regex
/// `^[_ ]?q(\d+)`.
fn parse_query_index(rest: &str) -> Option<u32> {
    let after_sep = rest
        .strip_prefix('_')
        .or_else(|| rest.strip_prefix(' '))
        .unwrap_or(rest);
    let after_q = after_sep
        .strip_prefix('q')
        .or_else(|| after_sep.strip_prefix('Q'))?;
    let digits: String = after_q.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Faithful port of v2's `normalizeChartName`.
pub fn normalize_chart_name(group: &V2Group, chart_name: &str) -> String {
    if matches!(group, V2Group::CompressionSize) && chart_name == "VORTEX FILE COMPRESSED SIZE" {
        return "VORTEX SIZE".into();
    }
    chart_name.to_string()
}

/// Port of v2's `getGroup`. Returns `None` for skipped suites
/// (e.g. `fineweb`) or names that match nothing.
pub fn get_group(record: &V2Record) -> Option<V2Group> {
    let lower = record.name.to_lowercase();

    if lower.starts_with("random-access/") || lower.starts_with("random access/") {
        return Some(V2Group::RandomAccess);
    }

    if lower.starts_with("vortex size/")
        || lower.starts_with("vortex-file-compressed size/")
        || lower.starts_with("parquet size/")
        || lower.starts_with("lance size/")
        || lower.contains(":raw size/")
        || lower.contains(":parquet-zstd size/")
        || lower.contains(":lance size/")
    {
        return Some(V2Group::CompressionSize);
    }

    if lower.starts_with("compress time/")
        || lower.starts_with("decompress time/")
        || lower.starts_with("parquet_rs-zstd compress")
        || lower.starts_with("parquet_rs-zstd decompress")
        || lower.starts_with("lance compress")
        || lower.starts_with("lance decompress")
        || lower.starts_with("vortex:lance ratio")
        || lower.starts_with("vortex:parquet-zstd ratio")
        || lower.starts_with("vortex:raw ratio")
    {
        return Some(V2Group::Compression);
    }

    for (i, suite) in QUERY_SUITES.iter().enumerate() {
        let prefix_q = format!("{}_q", suite.prefix);
        let prefix_slash = format!("{}/", suite.prefix);
        if !lower.starts_with(&prefix_q) && !lower.starts_with(&prefix_slash) {
            continue;
        }
        if suite.skip {
            return None;
        }
        if !suite.fan_out {
            return Some(V2Group::Query {
                suite_index: i,
                storage: None,
                scale_factor: None,
            });
        }
        let storage = match record.storage.as_deref().map(str::to_uppercase).as_deref() {
            Some("S3") => "S3",
            _ => "NVMe",
        };
        let dataset_key = suite.dataset_key.unwrap_or(suite.prefix);
        let raw_sf = record
            .dataset
            .as_ref()
            .and_then(|d| dataset_scale_factor(d, dataset_key));
        let sf = raw_sf
            .as_deref()
            .and_then(|s| s.parse::<f64>().ok())
            .map(|f| f.round() as i64)
            .unwrap_or(1);
        return Some(V2Group::Query {
            suite_index: i,
            storage: Some(storage.into()),
            scale_factor: Some(sf.to_string()),
        });
    }

    None
}

/// Group + chart + series breakdown for a v2 record, using the same
/// rules `server.js` applies in `refresh()`. Equivalent to v2's
/// `(group, chartName, seriesName)` triple after rename / skip rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V2Classification {
    pub group: V2Group,
    pub chart: String,
    pub series: String,
}

/// Apply the same chart / series naming v2's `refresh()` does, plus
/// the throughput / `PARQUET-UNC` skip rules.
pub fn classify_v2(record: &V2Record) -> Option<V2Classification> {
    if record.name.contains(" throughput") {
        return None;
    }
    let group = get_group(record)?;
    let parts: Vec<&str> = record.name.split('/').collect();
    let (chart, series) = match (&group, parts.len()) {
        (V2Group::RandomAccess, 4) => {
            let chart = format!("{}/{}", parts[1], parts[2])
                .to_uppercase()
                .replace(['_', '-'], " ");
            let series = rename_engine(if parts[3].is_empty() {
                "default"
            } else {
                parts[3]
            });
            (chart, series)
        }
        (V2Group::RandomAccess, 2) => (
            "RANDOM ACCESS".to_string(),
            rename_engine(if parts[1].is_empty() {
                "default"
            } else {
                parts[1]
            }),
        ),
        (V2Group::RandomAccess, _) => return None,
        _ => {
            let series_raw = if parts.len() >= 2 && !parts[1].is_empty() {
                parts[1]
            } else {
                "default"
            };
            let series = rename_engine(series_raw);
            let chart = format_query(parts[0]);
            (chart, series)
        }
    };
    let chart = normalize_chart_name(&group, &chart);
    if chart.contains("PARQUET-UNC") {
        return None;
    }
    Some(V2Classification {
        group,
        chart,
        series,
    })
}

/// Mapping target: which v3 fact table a v2 record lands in, plus the
/// dim values that table needs.
#[derive(Debug, Clone, PartialEq)]
pub enum V3Bin {
    Query {
        dataset: String,
        dataset_variant: Option<String>,
        scale_factor: Option<String>,
        query_idx: i32,
        storage: String,
        engine: String,
        format: String,
    },
    CompressionTime {
        dataset: String,
        dataset_variant: Option<String>,
        format: String,
        op: String,
    },
    CompressionSize {
        dataset: String,
        dataset_variant: Option<String>,
        format: String,
    },
    RandomAccess {
        dataset: String,
        format: String,
    },
}

/// Top-level entry point. Combines `classify_v2` with the v3 fact-table
/// mapping. Returns `None` for records that:
///
/// - Don't match any v2 group (uncategorized prefix).
/// - Are explicitly skipped by v2 (throughput, PARQUET-UNC, fineweb).
/// - Are computed-at-read-time ratios that v3 derives from
///   `compression_sizes` (`vortex:parquet-zstd ratio …`,
///   `vortex:lance ratio …`, `vortex:raw ratio …`,
///   `vortex:* size/…`).
pub fn classify(record: &V2Record) -> Option<V3Bin> {
    let cls = classify_v2(record)?;
    match &cls.group {
        V2Group::RandomAccess => bin_random_access(&cls, record),
        V2Group::Compression => bin_compression_time(&cls, record),
        V2Group::CompressionSize => bin_compression_size(&cls, record),
        V2Group::Query { .. } => bin_query(&cls, record),
    }
}

fn bin_random_access(cls: &V2Classification, record: &V2Record) -> Option<V3Bin> {
    // v2 chart name shape: "RANDOM ACCESS" or "DATASET/PATTERN" (uppercase).
    // We store it as the v3 dataset value verbatim, lowercased so
    // `/api/groups` returns canonical lowercase names.
    let dataset = cls.chart.to_lowercase();
    if dataset.is_empty() {
        return None;
    }
    // Pull format from the raw, pre-rename v2 name so v3 stores the
    // canonical `Format::name()` string (matching what the v3 live
    // emitter writes). Raw shape is
    // `random-access/<dataset>/<pattern>/<format>-tokio-local-disk`
    // (4-part) or `random-access/<format>-tokio-local-disk` (2-part
    // legacy). After stripping the `-tokio-local-disk` suffix, map the
    // v2 random-access ext label (`vortex`, from `Format::ext()`) to
    // the canonical name (`vortex-file-compressed`, from
    // `Format::name()`). `parquet`, `lance`, and `vortex-compact`
    // already match between ext and name.
    let parts: Vec<&str> = record.name.split('/').collect();
    let raw = match parts.len() {
        4 => parts[3],
        2 => parts[1],
        _ => return None,
    };
    if raw.is_empty() || raw == "default" {
        return None;
    }
    let stripped = raw.strip_suffix("-tokio-local-disk").unwrap_or(raw);
    let format = match stripped {
        "vortex" => "vortex-file-compressed".to_string(),
        other => other.to_lowercase(),
    };
    Some(V3Bin::RandomAccess { dataset, format })
}

fn bin_compression_time(cls: &V2Classification, _record: &V2Record) -> Option<V3Bin> {
    // v2 compression chart names look like (after format_query):
    //   "COMPRESS TIME"                                       [vortex/encode]
    //   "DECOMPRESS TIME"                                     [vortex/decode]
    //   "PARQUET RS ZSTD COMPRESS TIME"                       [parquet/encode]
    //   "PARQUET RS ZSTD DECOMPRESS TIME"                     [parquet/decode]
    //   "LANCE COMPRESS TIME"                                 [lance/encode]
    //   "LANCE DECOMPRESS TIME"                               [lance/decode]
    //   "VORTEX:LANCE RATIO COMPRESS TIME"                    [drop]
    //   "VORTEX:PARQUET-ZSTD RATIO COMPRESS TIME"             [drop]
    //   "VORTEX:RAW RATIO COMPRESS TIME"                      [drop]
    let lc = cls.chart.to_lowercase();
    if lc.contains("ratio") || lc.contains(':') {
        // Ratios are computed at read time from compression_sizes.
        return None;
    }
    let (format, op) = if lc.starts_with("compress time") {
        ("vortex-file-compressed", "encode")
    } else if lc.starts_with("decompress time") {
        ("vortex-file-compressed", "decode")
    } else if lc.starts_with("parquet rs zstd compress time") {
        ("parquet", "encode")
    } else if lc.starts_with("parquet rs zstd decompress time") {
        ("parquet", "decode")
    } else if lc.starts_with("lance compress time") {
        ("lance", "encode")
    } else if lc.starts_with("lance decompress time") {
        ("lance", "decode")
    } else {
        return None;
    };
    let dataset = cls.series.to_lowercase();
    if dataset.is_empty() || dataset == "default" {
        return None;
    }
    Some(V3Bin::CompressionTime {
        dataset,
        dataset_variant: None,
        format: format.to_string(),
        op: op.to_string(),
    })
}

fn bin_compression_size(cls: &V2Classification, _record: &V2Record) -> Option<V3Bin> {
    let lc = cls.chart.to_lowercase();
    // Ratios like "VORTEX:PARQUET ZSTD SIZE" / "VORTEX:LANCE SIZE" /
    // "VORTEX:RAW SIZE" are derived from compression_sizes at read
    // time, not stored.
    if lc.contains(':') {
        return None;
    }
    let format = if lc.starts_with("vortex size") {
        "vortex-file-compressed"
    } else if lc.starts_with("parquet size") {
        "parquet"
    } else if lc.starts_with("lance size") {
        "lance"
    } else {
        return None;
    };
    let dataset = cls.series.to_lowercase();
    if dataset.is_empty() || dataset == "default" {
        return None;
    }
    Some(V3Bin::CompressionSize {
        dataset,
        dataset_variant: None,
        format: format.to_string(),
    })
}

fn bin_query(cls: &V2Classification, record: &V2Record) -> Option<V3Bin> {
    let V2Group::Query {
        suite_index,
        storage,
        scale_factor,
    } = &cls.group
    else {
        return None;
    };
    let suite = &QUERY_SUITES[*suite_index];

    // Pull the query index from the *raw* name's first part instead of
    // the formatted chart, so we don't have to round-trip "Q07".
    let raw_first = record.name.split('/').next().unwrap_or("");
    let query_idx = parse_query_index_from_first(raw_first)?;

    // Pull engine:format from the raw, pre-rename second segment so v3
    // stores canonical `Format::name()` strings (e.g.
    // `vortex-file-compressed`) that match what the v3 live emitter
    // writes. `cls.series` has been through v2's `ENGINE_RENAMES` for
    // UI display and is not appropriate for v3 columns.
    let raw_series = record.name.split('/').nth(1)?;
    let (engine, format) = split_engine_format(raw_series)?;

    let storage_v3 = match storage.as_deref() {
        Some("S3") => "s3".to_string(),
        Some("NVMe") => "nvme".to_string(),
        _ => "nvme".to_string(),
    };

    // ClickBench's "flavor" lives in dataset_variant per benchmark-mapping.md
    // - we don't have it from a v2 name string, so we leave it None.
    Some(V3Bin::Query {
        dataset: suite.prefix.to_string(),
        dataset_variant: None,
        scale_factor: scale_factor.clone(),
        query_idx,
        storage: storage_v3,
        engine,
        format,
    })
}

/// Pull the integer query index out of the leading name part, which is
/// always `<prefix>_q<NN>` or `<prefix> q<NN>` for SQL query records.
fn parse_query_index_from_first(first: &str) -> Option<i32> {
    let lower = first.to_lowercase();
    for suite in QUERY_SUITES {
        if let Some(rest) = lower.strip_prefix(suite.prefix)
            && let Some(idx) = parse_query_index(rest)
        {
            return Some(idx as i32);
        }
    }
    None
}

/// Split a renamed series like `datafusion:parquet` into
/// `(engine, format)`. Returns `None` for series with no `:` since
/// v3 requires both columns.
fn split_engine_format(series: &str) -> Option<(String, String)> {
    let mut split = series.splitn(2, ':');
    let engine = split.next()?.trim().to_string();
    let format = split.next()?.trim().to_string();
    if engine.is_empty() || format.is_empty() {
        return None;
    }
    Some((engine, format))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(name: &str) -> V2Record {
        V2Record {
            name: name.to_string(),
            commit_id: Some("deadbeef".into()),
            unit: None,
            value: None,
            storage: None,
            dataset: None,
            all_runtimes: None,
            env_triple: None,
        }
    }

    #[test]
    fn format_query_round_trips() {
        assert_eq!(format_query("clickbench_q07"), "CLICKBENCH Q7");
        assert_eq!(format_query("tpch_q01"), "TPC-H Q1");
        assert_eq!(format_query("tpcds_q42"), "TPC-DS Q42");
        assert_eq!(format_query("statpopgen_q3"), "STATPOPGEN Q3");
        assert_eq!(format_query("foo bar"), "FOO BAR");
    }

    #[test]
    fn rename_engine_canonicalizes_disk_names() {
        assert_eq!(rename_engine("vortex-tokio-local-disk"), "vortex-nvme");
        assert_eq!(
            rename_engine("datafusion:vortex-file-compressed"),
            "datafusion:vortex"
        );
        assert_eq!(rename_engine("unknown-engine"), "unknown-engine");
    }

    #[test]
    fn parse_query_index_handles_separators() {
        assert_eq!(parse_query_index("_q07"), Some(7));
        assert_eq!(parse_query_index(" q7"), Some(7));
        assert_eq!(parse_query_index("q42"), Some(42));
        assert_eq!(parse_query_index("xq7"), None);
    }

    #[test]
    fn random_access_bins_dataset_pattern() {
        let bin = classify(&record("random-access/taxi/take/parquet")).unwrap();
        assert_eq!(
            bin,
            V3Bin::RandomAccess {
                dataset: "taxi/take".into(),
                format: "parquet".into(),
            }
        );
    }
}
