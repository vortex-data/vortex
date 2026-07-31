// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tencent Cloud COS, served over OpenDAL's `services::Cos`.

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

/// The URL scheme served by Tencent Cloud COS.
pub const COS_SCHEME: &str = "cos";

/// Property keys recognized for `cos://` URLs. Anything else is warned about and dropped.
const KNOWN_PROPERTIES: &[&str] = &[
    "bucket",
    "endpoint",
    "secret_id",
    "secret_key",
    "root",
    "disable_config_load",
];

/// Strongly-typed configuration for building an OpenDAL store against Tencent Cloud COS.
///
/// The fields mirror the keyword arguments of the `CosStore` Python class. Building from a
/// `CosConfig` avoids the URL-round-trip that the [`crate::opendal::make_opendal_store`] entry point uses,
/// and is the preferred way to construct a COS store.
#[derive(Debug, Clone, Default)]
pub struct CosConfig {
    /// COS bucket name. May be overridden by `properties["bucket"]` when adapting a URL.
    pub bucket: String,
    /// COS endpoint, e.g. `https://cos.ap-guangzhou.myqcloud.com`.
    pub endpoint: String,
    /// Tencent Cloud secret id (mapped to `TENCENTCLOUD_SECRET_ID`).
    pub secret_id: Option<String>,
    /// Tencent Cloud secret key (mapped to `TENCENTCLOUD_SECRET_KEY`).
    pub secret_key: Option<String>,
    /// Optional root prefix applied to all operations.
    pub root: Option<String>,
    /// When `true`, disable OpenDAL's automatic config loading (so only the explicit
    /// configuration is used).
    pub disable_config_load: bool,
}

/// Build an [`object_store::ObjectStore`] for Tencent Cloud COS directly from a [`CosConfig`].
///
/// This is the preferred entry point for callers that have a strongly-typed configuration object
/// (such as the `CosStore` pyclass in `vortex-python`). It does not synthesize a URL and so is not
/// fragile against reordering of `bucket` vs URL host precedence.
pub fn make_cos_store(config: CosConfig) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError> {
    if config.bucket.is_empty() {
        return Err(OpenDALStoreError::MissingConfig("bucket"));
    }
    if config.endpoint.is_empty() {
        return Err(OpenDALStoreError::MissingConfig("endpoint"));
    }

    let mut builder = services::Cos::default()
        .bucket(&config.bucket)
        .endpoint(&config.endpoint);

    if let Some(root) = config.root.as_deref() {
        builder = builder.root(root);
    }
    if let Some(secret_id) = config.secret_id.as_deref() {
        builder = builder.secret_id(secret_id);
    }
    if let Some(secret_key) = config.secret_key.as_deref() {
        builder = builder.secret_key(secret_key);
    }
    if config.disable_config_load {
        builder = builder.disable_config_load();
    }

    let operator = build_operator(builder)?;
    Ok(Arc::new(OpendalStore::new(operator)))
}

/// Translate a (`cos://` URL, properties) pair into a strongly-typed [`CosConfig`].
///
/// Bucket is taken from `properties["bucket"]` first and falls back to the URL host; endpoint is
/// taken from `properties["endpoint"]` first and falls back to `COS_ENDPOINT`; credentials fall
/// back to the variables OpenDAL's COS builder reads.
///
/// `env_lookup` is the source of truth for environment-variable fallbacks. The production entry
/// point passes the real environment; tests pass a closure that returns from a fixed map, so they
/// do not race against the process environment.
pub(crate) fn url_and_properties_to_config<F>(
    url: &Url,
    properties: &HashMap<String, String>,
    env_lookup: F,
) -> Result<CosConfig, OpenDALStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    warn_on_unknown_properties(properties, KNOWN_PROPERTIES);

    let bucket = properties
        .get("bucket")
        .cloned()
        .or_else(|| url.host_str().map(str::to_string))
        .ok_or(OpenDALStoreError::MissingConfig("bucket"))?;

    let endpoint = property_or_env(properties, "endpoint", "COS_ENDPOINT", &env_lookup)
        .ok_or(OpenDALStoreError::MissingConfig("endpoint"))?;

    Ok(CosConfig {
        bucket,
        endpoint,
        secret_id: property_or_env(
            properties,
            "secret_id",
            "TENCENTCLOUD_SECRET_ID",
            &env_lookup,
        ),
        secret_key: property_or_env(
            properties,
            "secret_key",
            "TENCENTCLOUD_SECRET_KEY",
            &env_lookup,
        ),
        root: properties.get("root").cloned(),
        disable_config_load: properties.get("disable_config_load").map(String::as_str)
            == Some("true"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The endpoint is required. With a fixed env-lookup that returns `None`, the call must
    /// fail with `MissingConfig("endpoint")` regardless of what the process environment
    /// happens to contain. This makes the test deterministic under `cargo test`'s default
    /// multi-threaded runner (and avoids the `unsafe std::env::set_var` that became unsound in
    /// Rust 2024).
    #[test]
    fn cos_requires_endpoint() {
        let url = Url::parse("cos://my-bucket/path").unwrap();
        let props = HashMap::new();
        let result = url_and_properties_to_config(&url, &props, |_| None).unwrap_err();
        assert!(matches!(
            result,
            OpenDALStoreError::MissingConfig("endpoint")
        ));
    }

    /// When `properties` does not contain `endpoint`, the env-lookup should be consulted.
    #[test]
    fn cos_falls_back_to_cos_endpoint_env() {
        let url = Url::parse("cos://my-bucket/path").unwrap();
        let env = |key: &str| match key {
            "COS_ENDPOINT" => Some("https://example.com".to_string()),
            _ => None,
        };
        let props = HashMap::new();
        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        assert_eq!(config.endpoint, "https://example.com");
    }

    /// An explicit property must win over the environment fallback.
    #[test]
    fn cos_property_overrides_env() {
        let url = Url::parse("cos://url-bucket/path").unwrap();
        let env = |key: &str| match key {
            "COS_ENDPOINT" => Some("https://from-env.example.com".to_string()),
            _ => None,
        };
        let mut props = HashMap::new();
        props.insert("bucket".to_string(), "prop-bucket".to_string());
        props.insert(
            "endpoint".to_string(),
            "https://from-prop.example.com".to_string(),
        );

        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        assert_eq!(config.bucket, "prop-bucket");
        assert_eq!(config.endpoint, "https://from-prop.example.com");
    }

    /// With a fixed env-lookup that returns explicit credentials, the strongly-typed
    /// `make_cos_store` should build a store successfully.
    #[test]
    fn cos_builds_with_explicit_config() {
        let url = Url::parse("cos://my-bucket/path/to/dataset.vortex").unwrap();
        let env = |key: &str| match key {
            "COS_ENDPOINT" => Some("https://example.com".to_string()),
            "TENCENTCLOUD_SECRET_ID" => Some("AKID".to_string()),
            "TENCENTCLOUD_SECRET_KEY" => Some("secret".to_string()),
            _ => None,
        };
        let mut props = HashMap::new();
        props.insert("disable_config_load".to_string(), "true".to_string());

        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        assert_eq!(config.secret_id.as_deref(), Some("AKID"));
        let store = make_cos_store(config).expect("store should build");
        // Sanity: the returned store is a non-null `Arc<dyn ObjectStore>`.
        assert!(Arc::strong_count(&store) >= 1);
    }

    /// The strongly-typed `make_cos_store` entry point must reject an empty bucket or endpoint
    /// with a `MissingConfig` error before consulting any environment or builder.
    #[test]
    fn cos_config_rejects_empty_fields() {
        assert!(matches!(
            make_cos_store(CosConfig {
                bucket: String::new(),
                ..CosConfig::default()
            }),
            Err(OpenDALStoreError::MissingConfig("bucket"))
        ));
        assert!(matches!(
            make_cos_store(CosConfig {
                bucket: "b".to_string(),
                endpoint: String::new(),
                ..CosConfig::default()
            }),
            Err(OpenDALStoreError::MissingConfig("endpoint"))
        ));
    }
}
