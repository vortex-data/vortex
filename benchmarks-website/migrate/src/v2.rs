// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wire shapes of the v2 benchmark dataset on S3.
//!
//! These types capture only the fields the migrator reads. v2 records
//! are serialized by `vortex-bench` (see `vortex-bench/src/measurements.rs`)
//! and by older non-Rust scripts; the union of fields is loose, so we
//! deserialize permissively (`serde(default)`, untyped `serde_json::Value`
//! for the polymorphic `dataset` field).

use std::collections::BTreeMap;

use serde::Deserialize;

/// One JSONL line of `data.json.gz`.
///
/// The shape is the union of every emitter's output. Most fields are
/// optional because different benches emit different subsets.
#[derive(Debug, Clone, Deserialize)]
pub struct V2Record {
    /// Slash-separated benchmark identifier (e.g. `tpch_q01/datafusion:vortex-file-compressed`).
    /// The classifier parses this string to recover dim values.
    pub name: String,
    /// 40-hex commit SHA. Present on every well-formed v2 record.
    #[serde(default)]
    pub commit_id: Option<String>,
    /// v2 unit string (`ns`, `bytes`, `ratio`, ...). Not used for routing —
    /// the classifier picks the v3 fact table from the `name` prefix instead.
    #[serde(default)]
    pub unit: Option<String>,
    /// Polymorphic value — emitters wrote both numbers and stringified
    /// numbers. Use [`value_as_f64`] to normalize.
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    /// Storage backend the run targeted (`S3` or `NVMe`, mixed case in v2).
    #[serde(default)]
    pub storage: Option<String>,
    /// Polymorphic dataset block — sometimes a string, sometimes an object
    /// keyed by suite name with a `scale_factor` inside (use
    /// [`dataset_scale_factor`]).
    #[serde(default)]
    pub dataset: Option<serde_json::Value>,
    /// Per-iteration runtimes; same numeric polymorphism as `value`.
    #[serde(default)]
    pub all_runtimes: Option<Vec<serde_json::Value>>,
    /// Host environment triple block.
    #[serde(default)]
    pub env_triple: Option<V2EnvTriple>,
}

/// `dataset` in v2 records is sometimes a string, sometimes an object
/// keyed by suite name (`{ "tpch": { "scale_factor": "10" } }`).
/// This helper looks up the scale factor for a given suite without
/// assuming a particular shape.
pub fn dataset_scale_factor(dataset: &serde_json::Value, key: &str) -> Option<String> {
    let obj = dataset.as_object()?;
    let entry = obj.get(key)?;
    let sf = entry.get("scale_factor")?;
    match sf {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Canonicalize a v2 scale-factor string for use in `dataset_variant`.
///
/// v2 emitters wrote scale factors as either `"1"`, `"1.0"`, `"10"`, or
/// `"10.0"` for the same logical SF, so the data.json.gz path
/// (`bin_compression_size`) and the file-sizes-*.json.gz path
/// (`migrate_file_sizes`) would otherwise produce different
/// `dataset_variant` strings and never collapse onto the same
/// `measurement_id`. Parse to f64 and format with no trailing zeros so
/// every shape collapses to one canonical form (`"1"`, `"10"`, `"0.1"`).
/// SF=1 is the implicit default and folds to `None`.
pub fn canonical_scale_factor(raw: Option<&str>) -> Option<String> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    let value: f64 = s.parse().ok()?;
    if value == 1.0 {
        return None;
    }
    Some(format!("{value}"))
}

/// Best-effort numeric coercion for the polymorphic `value` field.
pub fn value_as_f64(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Best-effort coercion of a runtime entry to nanoseconds.
pub fn runtime_as_i64(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                n.as_f64().map(|f| f as i64)
            }
        }
        serde_json::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Triple block as emitted by `vortex-bench`'s `--gh-json` path. v2
/// stored it as an object; we serialize it back out as `arch-os-env`.
#[derive(Debug, Clone, Deserialize)]
pub struct V2EnvTriple {
    /// Host CPU architecture (e.g. `x86_64`).
    #[serde(default)]
    pub architecture: Option<String>,
    /// Operating system name (e.g. `linux`).
    #[serde(default)]
    pub operating_system: Option<String>,
    /// Host environment label (e.g. `gnu`).
    #[serde(default)]
    pub environment: Option<String>,
}

impl V2EnvTriple {
    /// Format as the `arch-os-env` triple used by v3's `env_triple` column.
    pub fn to_triple(&self) -> Option<String> {
        let arch = self.architecture.as_deref()?;
        let os = self.operating_system.as_deref()?;
        let env = self.environment.as_deref()?;
        Some(format!("{arch}-{os}-{env}"))
    }
}

/// One JSONL line of `commits.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct V2Commit {
    /// 40-hex commit SHA (the v2 schema named this `id`, not `commit_sha`).
    pub id: String,
    /// RFC 3339 commit timestamp; required for the v3 row but tolerated as
    /// missing in the source dump.
    #[serde(default)]
    pub timestamp: Option<String>,
    /// Full commit message.
    #[serde(default)]
    pub message: Option<String>,
    /// Author block.
    #[serde(default)]
    pub author: Option<V2Person>,
    /// Committer block.
    #[serde(default)]
    pub committer: Option<V2Person>,
    /// Git tree SHA.
    #[serde(default)]
    pub tree_id: Option<String>,
    /// GitHub commit URL.
    #[serde(default)]
    pub url: Option<String>,
}

/// Author or committer block on a v2 commit record.
#[derive(Debug, Clone, Deserialize)]
pub struct V2Person {
    /// Display name.
    #[serde(default)]
    pub name: Option<String>,
    /// Email address.
    #[serde(default)]
    pub email: Option<String>,
}

/// One JSONL line of `file-sizes-*.json.gz` produced by
/// `scripts/capture-file-sizes.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct V2FileSize {
    /// 40-hex commit SHA.
    pub commit_id: String,
    /// Compression dataset name (`benchmark` is the v2 field name).
    pub benchmark: String,
    /// TPC SF as a string when relevant.
    #[serde(default)]
    pub scale_factor: Option<String>,
    /// Format the file was produced in.
    pub format: String,
    /// Path of the underlying file (e.g. `lineitem.parquet`); informational.
    pub file: String,
    /// Size in bytes; summed across files in the same `(commit, dataset, format)`.
    pub size_bytes: i64,
}

/// Build a sha-keyed map of commits.
pub fn index_commits(commits: Vec<V2Commit>) -> BTreeMap<String, V2Commit> {
    commits.into_iter().map(|c| (c.id.clone(), c)).collect()
}
