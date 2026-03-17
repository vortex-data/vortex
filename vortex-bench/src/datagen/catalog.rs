// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Catalog type: the top-level index of all datasets in a repository.
//!
//! The catalog is stored at the root of the repository (both S3 and local mirror)
//! as `catalog.json`. It is intentionally minimal — just dataset names, paths, and
//! manifest hashes — so that listing is cheap.

use serde::Deserialize;
use serde::Serialize;

/// Top-level catalog listing all datasets in the repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalog {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// All datasets in the repository.
    pub datasets: Vec<DatasetEntry>,
}

/// A single entry in the catalog, pointing to a dataset directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Logical name of the dataset (e.g. "tpch-sf100").
    pub name: String,
    /// Path to the dataset directory, including random suffix
    /// (e.g. "tpch-sf100-m9d2k4/").
    pub path: String,
    /// SHA-256 hash of the dataset's `manifest.json`.
    /// Used to detect concurrent modifications and for cache validation.
    pub manifest_hash: String,
}

impl Catalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            version: 1,
            datasets: Vec::new(),
        }
    }

    /// Find a dataset entry by name.
    pub fn find(&self, name: &str) -> Option<&DatasetEntry> {
        self.datasets.iter().find(|d| d.name == name)
    }

    /// Add or replace a dataset entry. Returns the old entry if it existed.
    pub fn upsert(&mut self, entry: DatasetEntry) -> Option<DatasetEntry> {
        if let Some(pos) = self.datasets.iter().position(|d| d.name == entry.name) {
            let old = std::mem::replace(&mut self.datasets[pos], entry);
            Some(old)
        } else {
            self.datasets.push(entry);
            None
        }
    }

    /// Remove a dataset entry by name. Returns the removed entry.
    pub fn remove(&mut self, name: &str) -> Option<DatasetEntry> {
        if let Some(pos) = self.datasets.iter().position(|d| d.name == name) {
            Some(self.datasets.remove(pos))
        } else {
            None
        }
    }

    /// Serialize to JSON bytes (catalog is always JSON, not YAML).
    pub fn to_json(&self) -> anyhow::Result<Vec<u8>> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Deserialize from JSON bytes.
    pub fn from_json(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}
