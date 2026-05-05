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
    pub name: String,
    #[serde(default)]
    pub commit_id: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub storage: Option<String>,
    #[serde(default)]
    pub dataset: Option<serde_json::Value>,
    #[serde(default)]
    pub all_runtimes: Option<Vec<serde_json::Value>>,
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
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub operating_system: Option<String>,
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
    pub id: String,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub author: Option<V2Person>,
    #[serde(default)]
    pub committer: Option<V2Person>,
    #[serde(default)]
    pub tree_id: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

/// Author or committer block on a v2 commit record.
#[derive(Debug, Clone, Deserialize)]
pub struct V2Person {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
}

/// One JSONL line of `file-sizes-*.json.gz` produced by
/// `scripts/capture-file-sizes.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct V2FileSize {
    pub commit_id: String,
    pub benchmark: String,
    #[serde(default)]
    pub scale_factor: Option<String>,
    pub format: String,
    pub file: String,
    pub size_bytes: i64,
}

/// Build a sha-keyed map of commits.
pub fn index_commits(commits: Vec<V2Commit>) -> BTreeMap<String, V2Commit> {
    commits.into_iter().map(|c| (c.id.clone(), c)).collect()
}
