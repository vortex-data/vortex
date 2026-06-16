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
        prefix: "gharchive",
        display_name: "GhArchive",
        query_prefix: "GHARCHIVE",
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
        skip: false,
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

/// One entry of [`QUERY_SUITES`].
#[derive(Debug, Clone, Copy)]
pub struct QuerySuite {
    /// Lowercase suite prefix used to match v2 record names (e.g. `tpch`).
    pub prefix: &'static str,
    /// Human-readable suite name as v2 served it from `/api/metadata`.
    pub display_name: &'static str,
    /// Uppercase prefix v2's `formatQuery` produced (e.g. `TPC-H`).
    pub query_prefix: &'static str,
    /// Override for the dataset key v2 records use inside their `dataset`
    /// object. Falls back to `prefix` when `None`.
    pub dataset_key: Option<&'static str>,
    /// True if the suite's group name fans out by `(storage, scale_factor)`
    /// (e.g. `TPC-H (NVMe) (SF=1)`); false collapses to a single group.
    pub fan_out: bool,
    /// True if v2 deliberately ignored this suite (no live group is rendered).
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
        || lower.starts_with("parquet-zstd size/")
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
        // Typo'd v2 emitter wrote `parquet-zst` (no `d`) for some
        // ratio records; match both spellings so they classify as
        // derived ratios instead of falling through to Unknown.
        || lower.starts_with("vortex:parquet-zst ratio")
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
    /// Group the v2 server would place this record in.
    pub group: V2Group,
    /// Chart name v2 displayed for this record (uppercase, separators
    /// normalized).
    pub chart: String,
    /// Series name after v2's `ENGINE_RENAMES` was applied.
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
        V2Group::RandomAccess => bin_random_access(record),
        V2Group::Compression => bin_compression_time(&cls, record),
        V2Group::CompressionSize => bin_compression_size(&cls, record),
        V2Group::Query { .. } => bin_query(&cls, record),
    }
}

/// Reason the classifier dropped a record. Intentional skips (v2
/// patterns v3 deliberately doesn't store) are NOT errors; they don't
/// count against the uncategorized gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skip {
    /// `vortex:* ratio …` and `vortex:* size` — derived in v3 from
    /// `compression_sizes` joined to itself.
    DerivedRatio,
    /// `throughput` records — v2 derived these from latencies.
    Throughput,
    /// A v2 query suite marked `skip: true` in QUERY_SUITES.
    SkippedSuite,
    /// random-access record with an unsupported part count.
    UnsupportedShape,
    /// Record had no `value` field.
    NoValue,
    /// Dim outside the v3 emitter's allowlist (e.g. `parquet-zstd`,
    /// historical-only suites no longer in CI).
    Deprecated,
    /// v2 memory measurements (`*_memory/*` records). Carry top-level
    /// `peak_physical_memory` / `peak_virtual_memory` /
    /// `physical_memory_delta` / `virtual_memory_delta` fields that
    /// `V2Record` doesn't deserialize. Not migrated for alpha; merging
    /// into the corresponding QueryMeasurement row is future work.
    HistoricalMemory,
}

/// Engines the v3 emitter produces today. Mirrors
/// `vortex-bench/src/lib.rs::Engine`. Anything else is historical and gets
/// bucketed as `Skip::Deprecated`.
const V3_ENGINES: &[&str] = &["datafusion", "duckdb", "vortex", "arrow"];

/// Formats the v3 emitter produces today (`Format::name()` values from
/// `vortex-bench/src/lib.rs`).
const V3_FORMATS: &[&str] = &[
    "vortex-file-compressed",
    "vortex-compact",
    "parquet",
    "lance",
    "csv",
    "arrow",
    "duckdb",
];

/// Query suites the v3 CI runs today. Suites outside this list still
/// classify (so historical analyses stay coherent) but get bucketed
/// as `Skip::Deprecated` so they don't render as orphan charts in v3.
///
/// `fineweb` is included because `.github/workflows/sql-benchmarks.yml`
/// still has `fineweb` and `fineweb-s3` matrix entries. `gharchive`
/// stays excluded — it's defined in `vortex-bench` but no current
/// workflow runs it.
const V3_QUERY_SUITES: &[&str] = &[
    "clickbench",
    "tpch",
    "tpcds",
    "statpopgen",
    "polarsignals",
    "fineweb",
];

/// Returns true if every dim that v3 stores as a column is on the
/// emitter's current allowlist. Dim values outside the allowlist mean
/// historical-only formats / engines that the v3 UI has nothing to
/// render against.
fn is_v3_dim(bin: &V3Bin) -> bool {
    match bin {
        V3Bin::Query { engine, format, .. } => {
            V3_ENGINES.contains(&engine.as_str()) && V3_FORMATS.contains(&format.as_str())
        }
        V3Bin::CompressionTime { format, .. }
        | V3Bin::CompressionSize { format, .. }
        | V3Bin::RandomAccess { format, .. } => V3_FORMATS.contains(&format.as_str()),
    }
}

/// Outcome of running the classifier on a v2 record. Distinguishes
/// "we know we don't want this" (`Skip`) from "we don't recognize this"
/// (`Unknown`); the migrator's 5% gate fires only on the latter.
#[derive(Debug, Clone)]
pub enum Outcome {
    Bin(V3Bin),
    Skip(Skip),
    Unknown,
}

/// Like [`classify`], but reports *why* a record was dropped. Intended
/// for the migrator so the 5% uncategorized gate doesn't trip on
/// records v2 deliberately doesn't render (ratios, throughput,
/// skipped suites).
pub fn classify_outcome(record: &V2Record) -> Outcome {
    if record.name.contains(" throughput") {
        return Outcome::Skip(Skip::Throughput);
    }
    // v2 memory records: e.g. "clickbench_q07_memory/datafusion:parquet".
    // Match the `_memory/` infix BEFORE the engine/format split, so they
    // route to a known Skip variant instead of slipping through to
    // Outcome::Unknown and tripping the 5% gate.
    let lower = record.name.to_lowercase();
    if let Some((head, _)) = lower.split_once('/')
        && head.ends_with("_memory")
    {
        return Outcome::Skip(Skip::HistoricalMemory);
    }
    let Some(group) = get_group(record) else {
        return Outcome::Unknown;
    };
    if let V2Group::Query { suite_index, .. } = &group
        && QUERY_SUITES[*suite_index].skip
    {
        return Outcome::Skip(Skip::SkippedSuite);
    }
    let Some(cls) = classify_v2(record) else {
        // get_group succeeded but classify_v2 didn't — shape mismatch.
        return Outcome::Skip(Skip::UnsupportedShape);
    };
    let derived = match &cls.group {
        V2Group::Compression => {
            let lc = cls.chart.to_lowercase();
            lc.contains("ratio") || lc.contains(':')
        }
        V2Group::CompressionSize => cls.chart.to_lowercase().contains(':'),
        _ => false,
    };
    if derived {
        return Outcome::Skip(Skip::DerivedRatio);
    }
    let bin = match &cls.group {
        V2Group::RandomAccess => match bin_random_access(record) {
            Some(b) => Some(b),
            // `bin_random_access` only returns None for malformed
            // shapes (empty dataset/pattern segment, empty/`default`
            // format). Route them to Skip so the `Outcome::Unknown`
            // arm below — and the 5% uncategorized gate in
            // `migrate::run` — don't trip on them.
            None => return Outcome::Skip(Skip::UnsupportedShape),
        },
        V2Group::Compression => bin_compression_time(&cls, record),
        V2Group::CompressionSize => bin_compression_size(&cls, record),
        V2Group::Query { .. } => bin_query(&cls, record),
    };
    let Some(bin) = bin else {
        return Outcome::Unknown;
    };
    if !is_v3_dim(&bin) {
        return Outcome::Skip(Skip::Deprecated);
    }
    if let V2Group::Query { suite_index, .. } = &group
        && !V3_QUERY_SUITES.contains(&QUERY_SUITES[*suite_index].prefix)
    {
        return Outcome::Skip(Skip::Deprecated);
    }
    Outcome::Bin(bin)
}

fn bin_random_access(record: &V2Record) -> Option<V3Bin> {
    // Pull dataset and format from the raw, pre-rename v2 name so v3
    // stores meaningful values. Two raw shapes are supported:
    //
    //   - 4-part `random-access/<dataset>/<pattern>/<format>-tokio-local-disk`
    //   - 2-part legacy `random-access/<format>-tokio-local-disk`
    //
    // The 2-part shape is what `random-access-bench`'s `measurement_name`
    // emits when called without an `AccessPattern`, and per its source
    // comment that path is only taken for the legacy taxi run
    // (`if dataset.name() == "taxi"` in `benchmarks/random-access-bench/
    // src/main.rs`). The live v3 emitter `random_access_record` writes
    // `dataset="taxi"` for those same measurements, so the historical
    // 2-part records are taxi too — assigning `dataset="taxi"` here
    // recovers the time series instead of letting it disappear under
    // v2's "RANDOM ACCESS" placeholder. Deriving from the raw name
    // (rather than `cls.chart`) keeps this independent of v2's
    // `normalizeChartName`.
    //
    // After stripping the `-tokio-local-disk` suffix, map the v2
    // random-access ext label (`vortex`, from `Format::ext()`) to the
    // canonical name (`vortex-file-compressed`, from `Format::name()`).
    // `parquet` and `lance` match between ext and name. The `vortex`
    // ext is shared by both `OnDiskVortex` (name
    // `vortex-file-compressed`) and `VortexCompact` (name
    // `vortex-compact`), but v2's random-access bench only emitted
    // `OnDiskVortex`, so mapping to `vortex-file-compressed` is
    // correct for all historical data.
    //
    // Records whose `<format>` segment ends in `-footer` (the bench's
    // reopen-mode variant, e.g. `parquet-tokio-local-disk-footer`)
    // intentionally do not strip clean to a v3-allowlisted format; the
    // outer `is_v3_dim` filter then routes them to `Skip::Deprecated`.
    // The live v3 emitter doesn't distinguish reopen vs cached either
    // (`random_access_record` uses `format.name()` for both), so
    // dropping `-footer` here keeps migration consistent with what
    // v3 ingests live.
    let parts: Vec<&str> = record.name.split('/').collect();
    let (dataset, raw_format) = match parts.as_slice() {
        [_, ds, pat, format] => {
            if ds.is_empty() || pat.is_empty() {
                return None;
            }
            (format!("{ds}/{pat}").to_lowercase(), *format)
        }
        [_, format] => ("taxi".to_string(), *format),
        _ => return None,
    };
    if raw_format.is_empty() || raw_format == "default" {
        return None;
    }
    let stripped = raw_format
        .strip_suffix("-tokio-local-disk")
        .unwrap_or(raw_format);
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

fn bin_compression_size(cls: &V2Classification, record: &V2Record) -> Option<V3Bin> {
    let lc = cls.chart.to_lowercase();
    // Ratios like "VORTEX:PARQUET ZSTD SIZE" / "VORTEX:LANCE SIZE" /
    // "VORTEX:RAW SIZE" are derived from compression_sizes at read
    // time, not stored.
    if lc.contains(':') {
        return None;
    }
    // `parquet-zstd size` shares a leading "parquet" with `parquet size`,
    // so check the more specific prefix first. `format_query` upper-cases
    // and replaces `-`/`_` with spaces, so the chart we match against is
    // `"PARQUET ZSTD SIZE"` (no hyphen) — same convention as the existing
    // `"parquet rs zstd compress time"` branches above.
    let format = if lc.starts_with("vortex size") {
        "vortex-file-compressed"
    } else if lc.starts_with("parquet zstd size") {
        "parquet-zstd"
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
    // Mirror the file-sizes ingest path's dataset_variant derivation
    // (see `migrate::migrate_file_sizes`): pull the SF out of the v2
    // record's `dataset` object when present and run it through
    // `canonical_scale_factor` so `"1"`, `"1.0"`, `"10"` and `"10.0"`
    // collapse to one canonical form. Without this both code paths
    // produce the same `mid` only by accident, so SF=10 file-sizes
    // rows wouldn't merge with the matching data.json.gz
    // "vortex size/tpch" rows when one side wrote `"10"` and the
    // other wrote `"10.0"`.
    let dataset_variant = crate::v2::canonical_scale_factor(
        record
            .dataset
            .as_ref()
            .and_then(|d| crate::v2::dataset_scale_factor(d, dataset.as_str()))
            .as_deref(),
    );
    Some(V3Bin::CompressionSize {
        dataset,
        dataset_variant,
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
    //
    // Older v2 records emitted display-case engines (e.g. `DataFusion`,
    // `DuckDB`); newer ones emit lowercase. Lowercase here so dedup
    // collapses both spellings into a single canonical row.
    let raw_series = record.name.split('/').nth(1)?;
    let (engine, format) = split_engine_format(raw_series)?;
    let engine = engine.to_lowercase();
    let format = format.to_lowercase();

    let storage_v3 = match storage.as_deref() {
        Some("S3") => "s3".to_string(),
        Some("NVMe") => "nvme".to_string(),
        _ => "nvme".to_string(),
    };

    // ClickBench's "flavor" lives in `dataset_variant`, but v2 record names
    // never encoded it — leave it `None` so historical and live rows merge
    // (the live emitter does the same; see `vortex-bench/src/v3.rs`'s
    // `benchmark_dataset_dims` for the matching shape).
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
    use anyhow::Context as _;

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
    fn random_access_bins_dataset_pattern() -> anyhow::Result<()> {
        let bin = classify(&record("random-access/taxi/take/parquet"))
            .context("classify returned None for a known-good 4-part random-access name")?;
        assert_eq!(
            bin,
            V3Bin::RandomAccess {
                dataset: "taxi/take".into(),
                format: "parquet".into(),
            }
        );
        Ok(())
    }
}
