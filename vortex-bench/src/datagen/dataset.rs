// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dataset descriptor: human-authored metadata for a dataset.
//!
//! This is stored as `dataset.yaml` and contains information that cannot be
//! derived from the data files: name, description, author, provenance, tags.
//! Extra fields are preserved through round-trips (via `serde_yaml::Value`).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;

/// Human-authored dataset descriptor, stored as `dataset.yaml`.
///
/// Extra fields beyond the known ones are preserved in `extra` so that users
/// can add arbitrary metadata without losing it on round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDescriptor {
    /// Dataset name. Must be lowercase alphanumeric with hyphens.
    pub name: String,
    /// Human-readable description of the dataset.
    pub description: String,
    /// Author in "Name <email>" format.
    pub author: String,
    /// Searchable tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Where the data came from.
    #[serde(default)]
    pub source: Option<Source>,
    /// Any extra fields the user wants to include.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_yaml::Value>,
}

/// Provenance information for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// One of: generator, external, production, derived.
    pub kind: String,
    /// Human-readable description of the source.
    pub description: String,
    /// Exact command used to produce the data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Name of the parent dataset if this is derived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Original download URL if external.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Valid source kinds.
const VALID_SOURCE_KINDS: &[&str] = &["generator", "external", "production", "derived"];

impl DatasetDescriptor {
    /// Create a template descriptor for `init`.
    pub fn template(name: &str, author: &str) -> Self {
        Self {
            name: name.to_string(),
            description: String::new(),
            author: author.to_string(),
            tags: Vec::new(),
            source: Some(Source {
                kind: String::new(),
                description: String::new(),
                command: None,
                parent: None,
                url: None,
            }),
            extra: BTreeMap::new(),
        }
    }

    /// Serialize to YAML string with a header comment.
    pub fn to_yaml(&self) -> Result<String> {
        let yaml = serde_yaml::to_string(self)?;
        Ok(format!(
            "# Dataset descriptor — fill in the fields below.\n\
             # You may add any extra fields you like; they will be preserved.\n\
             #\n\
             # source.kind must be one of: generator, external, production, derived\n\
             # source.parent is required when kind is \"derived\"\n\
             {yaml}"
        ))
    }

    /// Serialize to YAML bytes (without header comment, for S3 storage).
    pub fn to_yaml_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_yaml::to_string(self)?.into_bytes())
    }

    /// Deserialize from YAML bytes.
    pub fn from_yaml(bytes: &[u8]) -> Result<Self> {
        Ok(serde_yaml::from_slice(bytes)?)
    }

    /// Read from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_yaml(&bytes)
    }

    /// Write to a file path with header comment.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let content = self.to_yaml()?;
        std::fs::write(path, content).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Validate the descriptor, returning a list of problems.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.name.is_empty() {
            problems.push("name is empty".to_string());
        } else if !self
            .name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            problems.push(format!(
                "name '{}' must be lowercase alphanumeric with hyphens",
                self.name
            ));
        }

        if self.description.is_empty() {
            problems.push("description is empty".to_string());
        }

        if self.author.is_empty() {
            problems.push("author is empty".to_string());
        }

        if let Some(source) = &self.source {
            if source.kind.is_empty() {
                problems.push("source.kind is empty".to_string());
            } else if !VALID_SOURCE_KINDS.contains(&source.kind.as_str()) {
                problems.push(format!(
                    "source.kind '{}' must be one of: {}",
                    source.kind,
                    VALID_SOURCE_KINDS.join(", ")
                ));
            }

            if source.description.is_empty() {
                problems.push("source.description is empty".to_string());
            }

            if source.kind == "derived" && source.parent.is_none() {
                problems
                    .push("source.parent is required when source.kind is 'derived'".to_string());
            }
        }

        problems
    }
}
