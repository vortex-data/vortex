// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tencent Cloud GooseFS, served over OpenDAL's `services::GooseFs`.
//!
//! GooseFS is a distributed caching file system accessed via native gRPC. Unlike COS/OSS it has
//! no bucket or HTTP endpoint — instead the client connects to a GooseFS master at
//! `host:port`. An HA cluster is expressed as a comma-separated list of master addresses.

use std::sync::Arc;

use ::opendal::services;
use object_store::ObjectStore;
use object_store_opendal::OpendalStore;
use url::Url;
use vortex_utils::aliases::hash_map::HashMap;

use crate::opendal::OpenDALStoreError;
use crate::opendal::build_operator;
use crate::opendal::property_or_env;
use crate::opendal::warn_on_unknown_properties;

/// The URL scheme served by Tencent Cloud GooseFS.
pub const GOOSEFS_SCHEME: &str = "goosefs";

/// Property keys recognized for `goosefs://` URLs. Anything else is warned about and dropped.
const KNOWN_PROPERTIES: &[&str] = &[
    "master_addr",
    "root",
    "block_size",
    "chunk_size",
    "write_type",
    "auth_type",
    "auth_username",
];

/// Strongly-typed configuration for building an OpenDAL store against Tencent Cloud GooseFS.
///
/// The fields mirror the keyword arguments of the `GoosefsStore` Python class. Building from a
/// [`GoosefsConfig`] avoids the URL-round-trip that the [`crate::opendal::make_opendal_store`]
/// entry point uses, and is the preferred way to construct a GooseFS store.
#[derive(Debug, Clone, Default)]
pub struct GoosefsConfig {
    /// GooseFS master address(es).
    ///
    /// Single master: `"10.0.0.1:9200"`
    /// HA (comma-separated): `"10.0.0.1:9200,10.0.0.2:9200,10.0.0.3:9200"`
    ///
    /// May be overridden by `properties["master_addr"]` when adapting a URL. Falls back to the
    /// `GOOSEFS_MASTER_ADDR` environment variable.
    pub master_addr: String,
    /// Optional root prefix applied to all operations.
    pub root: Option<String>,
    /// Block size in bytes for new files (default: 64 MiB).
    pub block_size: Option<u64>,
    /// Chunk size in bytes for streaming RPCs (default: 1 MiB).
    pub chunk_size: Option<u64>,
    /// Default write type: `"must_cache"`, `"cache_through"`, `"through"`, `"async_through"`.
    pub write_type: Option<String>,
    /// Authentication type: `"nosasl"` or `"simple"` (default: `"simple"`).
    pub auth_type: Option<String>,
    /// Authentication username (default: current OS user).
    pub auth_username: Option<String>,
}

/// Build an [`object_store::ObjectStore`] for Tencent Cloud GooseFS directly from a
/// [`GoosefsConfig`].
///
/// This is the preferred entry point for callers that have a strongly-typed configuration object
/// (such as the `GoosefsStore` pyclass in `vortex-python`). It does not synthesize a URL and so is
/// not fragile against reordering of `master_addr` vs URL authority precedence.
pub fn make_goosefs_store(
    config: GoosefsConfig,
) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError> {
    if config.master_addr.is_empty() {
        return Err(OpenDALStoreError::MissingConfig("master_addr"));
    }

    let mut builder = services::GooseFs::default().master_addr(&config.master_addr);

    if let Some(root) = config.root.as_deref() {
        builder = builder.root(root);
    }
    if let Some(block_size) = config.block_size {
        builder = builder.block_size(block_size);
    }
    if let Some(chunk_size) = config.chunk_size {
        builder = builder.chunk_size(chunk_size);
    }
    if let Some(write_type) = config.write_type.as_deref() {
        builder = builder.write_type(write_type);
    }
    if let Some(auth_type) = config.auth_type.as_deref() {
        builder = builder.auth_type(auth_type);
    }
    if let Some(auth_username) = config.auth_username.as_deref() {
        builder = builder.auth_username(auth_username);
    }

    let operator = build_operator(builder)?;
    Ok(Arc::new(OpendalStore::new(operator)))
}

/// Translate a (`goosefs://` URL, properties) pair into a strongly-typed [`GoosefsConfig`].
///
/// `master_addr` is taken from `properties["master_addr"]` first and falls back to the URL
/// authority (`host:port`); if neither is present it falls back to the `GOOSEFS_MASTER_ADDR`
/// environment variable. The remaining fields are taken from `properties` directly.
///
/// `env_lookup` is the source of truth for environment-variable fallbacks. The production entry
/// point passes the real environment; tests pass a closure that returns from a fixed map, so they
/// do not race against the process environment.
pub(crate) fn url_and_properties_to_config<F>(
    url: &Url,
    properties: &HashMap<String, String>,
    env_lookup: F,
) -> Result<GoosefsConfig, OpenDALStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    warn_on_unknown_properties(properties, KNOWN_PROPERTIES);

    // master_addr: properties → URL authority → GOOSEFS_MASTER_ADDR env
    let master_addr = properties
        .get("master_addr")
        .cloned()
        .or_else(|| {
            let host = url.host_str();
            let port = url.port();
            match (host, port) {
                (Some(h), Some(p)) => Some(format!("{h}:{p}")),
                (Some(h), None) => Some(h.to_string()),
                _ => None,
            }
        })
        .or_else(|| {
            property_or_env(
                properties,
                "master_addr",
                "GOOSEFS_MASTER_ADDR",
                &env_lookup,
            )
        })
        .ok_or(OpenDALStoreError::MissingConfig("master_addr"))?;

    Ok(GoosefsConfig {
        master_addr,
        root: properties.get("root").cloned(),
        block_size: properties.get("block_size").and_then(|s| s.parse().ok()),
        chunk_size: properties.get("chunk_size").and_then(|s| s.parse().ok()),
        write_type: properties.get("write_type").cloned(),
        auth_type: properties.get("auth_type").cloned(),
        auth_username: properties.get("auth_username").cloned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `master_addr` is required. With a fixed env-lookup that returns `None` and a URL without
    /// authority, the call must fail with `MissingConfig("master_addr")` regardless of what the
    /// process environment happens to contain.
    #[test]
    fn goosefs_requires_master_addr() {
        let url = Url::parse("goosefs:///path/to/file").unwrap();
        let props = HashMap::new();
        let result = url_and_properties_to_config(&url, &props, |_| None).unwrap_err();
        assert!(matches!(
            result,
            OpenDALStoreError::MissingConfig("master_addr")
        ));
    }

    /// The URL authority (`host:port`) should be used as `master_addr` when properties do not
    /// contain it.
    #[test]
    fn goosefs_uses_url_authority_as_master_addr() {
        let url = Url::parse("goosefs://10.0.0.1:9200/path/to/file").unwrap();
        let props = HashMap::new();
        let config = url_and_properties_to_config(&url, &props, |_| None).expect("config");
        assert_eq!(config.master_addr, "10.0.0.1:9200");
    }

    /// When `properties` does not contain `master_addr` and the URL has no authority, the
    /// env-lookup should be consulted.
    #[test]
    fn goosefs_falls_back_to_env() {
        let url = Url::parse("goosefs:///path/to/file").unwrap();
        let env = |key: &str| match key {
            "GOOSEFS_MASTER_ADDR" => Some("10.0.0.1:9200".to_string()),
            _ => None,
        };
        let props = HashMap::new();
        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        assert_eq!(config.master_addr, "10.0.0.1:9200");
    }

    /// An explicit property must win over the URL authority and the environment fallback.
    #[test]
    fn goosefs_property_overrides_url_and_env() {
        let url = Url::parse("goosefs://url-host:9200/path").unwrap();
        let env = |key: &str| match key {
            "GOOSEFS_MASTER_ADDR" => Some("from-env:9200".to_string()),
            _ => None,
        };
        let mut props = HashMap::new();
        props.insert("master_addr".to_string(), "from-prop:9200".to_string());

        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        assert_eq!(config.master_addr, "from-prop:9200");
    }

    /// The URL authority must win over the environment fallback when no explicit property is set.
    #[test]
    fn goosefs_url_authority_overrides_env() {
        let url = Url::parse("goosefs://url-host:9200/path").unwrap();
        let env = |key: &str| match key {
            "GOOSEFS_MASTER_ADDR" => Some("from-env:9200".to_string()),
            _ => None,
        };
        let props = HashMap::new();
        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        assert_eq!(config.master_addr, "url-host:9200");
    }

    /// With a fixed env-lookup that returns an explicit master address, the strongly-typed
    /// `make_goosefs_store` should build a store successfully.
    #[test]
    fn goosefs_builds_with_explicit_config() {
        let config = GoosefsConfig {
            master_addr: "127.0.0.1:9200".to_string(),
            root: Some("/data".to_string()),
            ..GoosefsConfig::default()
        };
        let store = make_goosefs_store(config).expect("store should build");
        // Sanity: the returned store is a non-null `Arc<dyn ObjectStore>`.
        assert!(Arc::strong_count(&store) >= 1);
    }

    /// The strongly-typed `make_goosefs_store` entry point must reject an empty `master_addr`
    /// with a `MissingConfig` error before consulting any environment or builder.
    #[test]
    fn goosefs_config_rejects_empty_master_addr() {
        assert!(matches!(
            make_goosefs_store(GoosefsConfig {
                master_addr: String::new(),
                ..GoosefsConfig::default()
            }),
            Err(OpenDALStoreError::MissingConfig("master_addr"))
        ));
    }

    /// HA mode: a comma-separated list of master addresses should be accepted and build
    /// successfully.
    #[test]
    fn goosefs_builds_ha_config() {
        let config = GoosefsConfig {
            master_addr: "10.0.0.1:9200,10.0.0.2:9200,10.0.0.3:9200".to_string(),
            root: Some("/data".to_string()),
            ..GoosefsConfig::default()
        };
        let store = make_goosefs_store(config).expect("HA store should build");
        assert!(Arc::strong_count(&store) >= 1);
    }
}
