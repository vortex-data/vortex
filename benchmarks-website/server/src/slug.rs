// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opaque slugs for `/api/chart/:slug`.
//!
//! Per `02-contracts.md`, the web-ui treats slugs as opaque strings: it
//! receives them from `/api/groups` and feeds them back unchanged to
//! `/api/chart/:slug`. The server is free to choose any format.
//!
//! Slugs here are `<prefix>.<base64url-of-json>` where `<prefix>` names the
//! source fact table and the JSON encodes the chart key. Round-tripping the
//! slug back gives a strongly-typed [`ChartKey`].

use anyhow::Context as _;
use anyhow::Result;
use anyhow::anyhow;
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Deserialize;
use serde::Serialize;

const PREFIX_QUERY: &str = "qm";
const PREFIX_COMPRESSION_TIME: &str = "ct";
const PREFIX_COMPRESSION_SIZE: &str = "cs";
const PREFIX_RANDOM_ACCESS: &str = "rat";
const PREFIX_VECTOR_SEARCH: &str = "vsr";

const PREFIX_QUERY_GROUP: &str = "qmg";
const PREFIX_COMPRESSION_TIME_GROUP: &str = "ctg";
const PREFIX_COMPRESSION_SIZE_GROUP: &str = "csg";
const PREFIX_RANDOM_ACCESS_GROUP: &str = "rag";
const PREFIX_VECTOR_SEARCH_GROUP: &str = "vsg";

/// The strongly-typed chart key parsed from a slug.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "k")]
pub enum ChartKey {
    /// `query_measurements` chart: `(dataset, query_idx)` per `01-schema.md`.
    /// Group context (`dataset_variant`, `scale_factor`, `storage`) is carried
    /// alongside so the slug fully specifies the chart.
    QueryMeasurement {
        dataset: String,
        dataset_variant: Option<String>,
        scale_factor: Option<String>,
        storage: String,
        query_idx: i32,
    },
    /// `compression_times` chart: `(dataset, dataset_variant)`.
    CompressionTime {
        dataset: String,
        dataset_variant: Option<String>,
    },
    /// `compression_sizes` chart: `(dataset, dataset_variant)`.
    CompressionSize {
        dataset: String,
        dataset_variant: Option<String>,
    },
    /// `random_access_times` chart: `dataset`.
    RandomAccess { dataset: String },
    /// `vector_search_runs` chart: `(dataset, layout, threshold)`.
    VectorSearch {
        dataset: String,
        layout: String,
        threshold: f64,
    },
}

impl ChartKey {
    fn prefix(&self) -> &'static str {
        match self {
            Self::QueryMeasurement { .. } => PREFIX_QUERY,
            Self::CompressionTime { .. } => PREFIX_COMPRESSION_TIME,
            Self::CompressionSize { .. } => PREFIX_COMPRESSION_SIZE,
            Self::RandomAccess { .. } => PREFIX_RANDOM_ACCESS,
            Self::VectorSearch { .. } => PREFIX_VECTOR_SEARCH,
        }
    }

    /// Render the slug for this chart key.
    pub fn to_slug(&self) -> String {
        let json = serde_json::to_vec(self).expect("ChartKey is always JSON-serializable");
        format!("{}.{}", self.prefix(), URL_SAFE_NO_PAD.encode(json))
    }

    /// Parse a slug previously produced by [`Self::to_slug`].
    pub fn from_slug(slug: &str) -> Result<Self> {
        let (_, encoded) = slug
            .split_once('.')
            .ok_or_else(|| anyhow!("slug missing '.' separator"))?;
        let json = URL_SAFE_NO_PAD
            .decode(encoded.as_bytes())
            .context("slug payload was not valid base64url")?;
        let key: Self = serde_json::from_slice(&json).context("slug payload was not valid JSON")?;
        Ok(key)
    }
}

/// Slug for a *group* of charts. Mirrors [`ChartKey`] but at the group
/// granularity: a group is a set of charts that share every dimension except
/// one (e.g. all 22 TPC-H queries at sf=1 on nvme).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "k")]
pub enum GroupKey {
    /// All `query_measurements` charts at one `(dataset, dataset_variant,
    /// scale_factor, storage)`. Charts inside vary by `query_idx`.
    QueryGroup {
        dataset: String,
        dataset_variant: Option<String>,
        scale_factor: Option<String>,
        storage: String,
    },
    /// Every compression-time chart (one per `(dataset, dataset_variant)`).
    CompressionTimeGroup,
    /// Every compression-size chart.
    CompressionSizeGroup,
    /// Every random-access chart.
    RandomAccessGroup,
    /// All vector-search charts at one `(dataset, layout)`. Charts inside
    /// vary by `threshold`.
    VectorSearchGroup { dataset: String, layout: String },
}

impl GroupKey {
    fn prefix(&self) -> &'static str {
        match self {
            Self::QueryGroup { .. } => PREFIX_QUERY_GROUP,
            Self::CompressionTimeGroup => PREFIX_COMPRESSION_TIME_GROUP,
            Self::CompressionSizeGroup => PREFIX_COMPRESSION_SIZE_GROUP,
            Self::RandomAccessGroup => PREFIX_RANDOM_ACCESS_GROUP,
            Self::VectorSearchGroup { .. } => PREFIX_VECTOR_SEARCH_GROUP,
        }
    }

    /// Render the slug for this group key.
    pub fn to_slug(&self) -> String {
        let json = serde_json::to_vec(self).expect("GroupKey is always JSON-serializable");
        format!("{}.{}", self.prefix(), URL_SAFE_NO_PAD.encode(json))
    }

    /// Parse a slug previously produced by [`Self::to_slug`].
    pub fn from_slug(slug: &str) -> Result<Self> {
        let (_, encoded) = slug
            .split_once('.')
            .ok_or_else(|| anyhow!("slug missing '.' separator"))?;
        let json = URL_SAFE_NO_PAD
            .decode(encoded.as_bytes())
            .context("slug payload was not valid base64url")?;
        let key: Self = serde_json::from_slice(&json).context("slug payload was not valid JSON")?;
        Ok(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(key: ChartKey) {
        let slug = key.to_slug();
        let parsed = ChartKey::from_slug(&slug).expect("parses back");
        assert_eq!(parsed, key);
    }

    #[test]
    fn query_measurement_roundtrips() {
        roundtrip(ChartKey::QueryMeasurement {
            dataset: "tpch".into(),
            dataset_variant: None,
            scale_factor: Some("1".into()),
            storage: "nvme".into(),
            query_idx: 7,
        });
    }

    #[test]
    fn vector_search_roundtrips() {
        roundtrip(ChartKey::VectorSearch {
            dataset: "cohere-large-10m".into(),
            layout: "partitioned".into(),
            threshold: 0.75,
        });
    }

    #[test]
    fn random_access_roundtrips() {
        roundtrip(ChartKey::RandomAccess {
            dataset: "taxi".into(),
        });
    }

    #[test]
    fn malformed_slug_rejected() {
        assert!(ChartKey::from_slug("not-a-slug").is_err());
        assert!(ChartKey::from_slug("qm.****").is_err());
    }

    fn roundtrip_group(key: GroupKey) {
        let slug = key.to_slug();
        let parsed = GroupKey::from_slug(&slug).expect("parses back");
        assert_eq!(parsed, key);
    }

    #[test]
    fn group_keys_roundtrip() {
        roundtrip_group(GroupKey::QueryGroup {
            dataset: "tpch".into(),
            dataset_variant: None,
            scale_factor: Some("1".into()),
            storage: "nvme".into(),
        });
        roundtrip_group(GroupKey::CompressionTimeGroup);
        roundtrip_group(GroupKey::CompressionSizeGroup);
        roundtrip_group(GroupKey::RandomAccessGroup);
        roundtrip_group(GroupKey::VectorSearchGroup {
            dataset: "cohere".into(),
            layout: "partitioned".into(),
        });
    }

    #[test]
    fn group_slug_prefix_distinct_from_chart() {
        let chart = ChartKey::CompressionTime {
            dataset: "tpch".into(),
            dataset_variant: None,
        }
        .to_slug();
        let group = GroupKey::CompressionTimeGroup.to_slug();
        let chart_prefix = chart.split_once('.').unwrap().0;
        let group_prefix = group.split_once('.').unwrap().0;
        assert_ne!(chart_prefix, group_prefix);
    }

    #[test]
    fn malformed_group_slug_rejected() {
        assert!(GroupKey::from_slug("not-a-slug").is_err());
        assert!(GroupKey::from_slug("qmg.****").is_err());
    }
}
