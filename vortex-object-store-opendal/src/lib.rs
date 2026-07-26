// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OpenDAL-backed [`object_store::ObjectStore`] implementations for cloud providers that are not
//! natively supported by the `object_store` crate, such as Tencent Cloud COS.
//!
//! OpenDAL exposes each service as an `Operator`. We adapt an `Operator` into an
//! `object_store::ObjectStore` via the `object_store_opendal::OpendalStore` bridge, which is built
//! against the same `object_store 0.13.x` version the rest of Vortex uses. This lets Vortex consume
//! COS through its existing `ObjectStoreFileSystem` abstraction without any changes to
//! `vortex-io`.
//!
//! # Limitations
//!
//! The [`OpendalStore`] bridge owns its own HTTP request client. Configuration that the JNI/Python
//! layers normally pass through [`object_store::ClientOptions`] — connect/request timeouts,
//! retries, proxy settings, `allow_http` — has no effect on COS URLs handled here. Properties
//! that are not recognized by this crate are dropped without a warning; callers that need strict
//! validation must pre-filter their property maps.

use std::sync::Arc;

use object_store::ObjectStore;
use object_store_opendal::OpendalStore;
use opendal::Operator;
use opendal::services;
use tracing::warn;
use url::Url;
use vortex_utils::aliases::hash_map::HashMap;

/// Error type for building an OpenDAL-backed object store.
#[derive(Debug)]
pub enum OpenDALStoreError {
    /// The URL scheme is not one this crate handles (e.g. `s3`, `gs`, ...).
    UnsupportedScheme(String),
    /// A required configuration value (bucket and/or endpoint) was missing.
    MissingConfig(&'static str),
    /// The OpenDAL builder rejected the provided configuration.
    Build(opendal::Error),
}

impl std::fmt::Display for OpenDALStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenDALStoreError::UnsupportedScheme(s) => {
                write!(f, "unsupported OpenDAL scheme: {s}")
            }
            OpenDALStoreError::MissingConfig(k) => {
                write!(f, "missing required OpenDAL store configuration: {k}")
            }
            OpenDALStoreError::Build(e) => write!(f, "failed to build OpenDAL store: {e}"),
        }
    }
}

impl std::error::Error for OpenDALStoreError {}

impl From<OpenDALStoreError> for object_store::Error {
    fn from(e: OpenDALStoreError) -> Self {
        object_store::Error::Generic {
            store: "OpenDAL",
            source: Box::new(e),
        }
    }
}

/// Schemes handled by this crate.
pub const COS_SCHEME: &str = "cos";

/// Strongly-typed configuration for building a [`OpendalStore`] against Tencent Cloud COS.
///
/// The fields mirror the keyword arguments of the `CosStore` Python class. Building from a
/// `CosConfig` avoids the URL-round-trip that the legacy `make_opendal_store(url, properties)`
/// entry point used, and is the preferred way to construct a COS store.
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

/// Build an [`object_store::ObjectStore`] for a `cos://` URL.
///
/// `properties` are per-request configuration overrides (matching the `HashMap<String, String>`
/// passed through the JNI/Python layers). Missing values fall back to environment variables that
/// OpenDAL's builders read automatically (e.g. `TENCENTCLOUD_SECRET_ID`).
///
/// Returns [`OpenDALStoreError::UnsupportedScheme`] if `url` does not use `cos://`.
pub fn make_opendal_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError> {
    if url.scheme() != COS_SCHEME {
        return Err(OpenDALStoreError::UnsupportedScheme(
            url.scheme().to_string(),
        ));
    }

    // Translate the (URL, properties) pair into a strongly-typed `CosConfig` and delegate to
    // `make_cos_store`. Bucket is taken from `properties["bucket"]` first (matching the historical
    // precedence) and falls back to the URL host; endpoint is taken from `properties["endpoint"]`
    // first and falls back to `COS_ENDPOINT`; credentials fall back to the variables that
    // OpenDAL's COS builder reads.
    let config = url_and_properties_to_cos_config(url, properties, env_var_lookup)?;
    make_cos_store(config)
}

/// Default environment-variable lookup: reads `key` from the process environment.
///
/// The lookup is factored out so tests can pass a fixed map instead of mutating the global
/// environment, which is unsound when `cargo test` runs tests on multiple threads within one
/// process (the `unsafe std::env::set_var` block became `unsafe` in Rust 2024 for exactly this
/// reason).
fn env_var_lookup(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Translate the (URL, properties) pair into a strongly-typed [`CosConfig`].
///
/// `env_lookup` is the source of truth for environment-variable fallbacks. The production
/// entry points pass [`env_var_lookup`]; tests pass a closure that returns from a fixed map, so
/// they do not race against the process environment.
fn url_and_properties_to_cos_config<F>(
    url: &Url,
    properties: &HashMap<String, String>,
    env_lookup: F,
) -> Result<CosConfig, OpenDALStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    warn_on_unknown_properties(properties);

    let bucket = properties
        .get("bucket")
        .cloned()
        .or_else(|| url.host_str().map(str::to_string))
        .ok_or(OpenDALStoreError::MissingConfig("bucket"))?;

    let endpoint = properties
        .get("endpoint")
        .cloned()
        .or_else(|| env_lookup("COS_ENDPOINT"))
        .ok_or(OpenDALStoreError::MissingConfig("endpoint"))?;

    Ok(CosConfig {
        bucket,
        endpoint,
        secret_id: properties
            .get("secret_id")
            .cloned()
            .or_else(|| env_lookup("TENCENTCLOUD_SECRET_ID")),
        secret_key: properties
            .get("secret_key")
            .cloned()
            .or_else(|| env_lookup("TENCENTCLOUD_SECRET_KEY")),
        root: properties.get("root").cloned(),
        disable_config_load: properties.get("disable_config_load").map(String::as_str)
            == Some("true"),
    })
}

fn warn_on_unknown_properties(properties: &HashMap<String, String>) {
    const KNOWN: &[&str] = &[
        "bucket",
        "endpoint",
        "secret_id",
        "secret_key",
        "root",
        "disable_config_load",
    ];
    for key in properties.keys() {
        if !KNOWN.contains(&key.as_str()) {
            warn!("ignoring unknown OpenDAL store property: {key}");
        }
    }
}

fn build_operator<B>(builder: B) -> Result<Operator, OpenDALStoreError>
where
    B: opendal::Builder,
{
    let operator: Operator = Operator::new(builder)
        .map_err(OpenDALStoreError::Build)?
        .finish();
    Ok(operator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_scheme() {
        let url = Url::parse("s3://bucket/path").unwrap();
        let props = HashMap::new();
        assert!(matches!(
            make_opendal_store(&url, &props),
            Err(OpenDALStoreError::UnsupportedScheme(_))
        ));
    }

    /// The endpoint is required. With a fixed env-lookup that returns `None`, the call must
    /// fail with `MissingConfig("endpoint")` regardless of what the process environment
    /// happens to contain. This makes the test deterministic under `cargo test`'s default
    /// multi-threaded runner (and avoids the `unsafe std::env::set_var` that became unsound in
    /// Rust 2024).
    #[test]
    fn cos_requires_endpoint() {
        let url = Url::parse("cos://my-bucket/path").unwrap();
        let props = HashMap::new();
        let result = url_and_properties_to_cos_config(&url, &props, |_| None).unwrap_err();
        assert!(matches!(
            result,
            OpenDALStoreError::MissingConfig("endpoint")
        ));
    }

    /// When `properties` does not contain `endpoint`, the env-lookup should be consulted.
    /// The fixed lookup returns `"https://example.com"`, so the build must succeed.
    #[test]
    fn cos_falls_back_to_cos_endpoint_env() {
        let url = Url::parse("cos://my-bucket/path").unwrap();
        let env = |key: &str| match key {
            "COS_ENDPOINT" => Some("https://example.com".to_string()),
            _ => None,
        };
        let props = HashMap::new();
        let config = url_and_properties_to_cos_config(&url, &props, env).expect("config");
        assert_eq!(config.endpoint, "https://example.com");
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

        let config = url_and_properties_to_cos_config(&url, &props, env).expect("config");
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
