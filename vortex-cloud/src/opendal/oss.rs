// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Alibaba Cloud OSS, served over OpenDAL's `services::Oss`.

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

/// The URL scheme served by Alibaba Cloud OSS.
pub const OSS_SCHEME: &str = "oss";

/// Property keys recognized for `oss://` URLs. Anything else is warned about and dropped.
const KNOWN_PROPERTIES: &[&str] = &[
    "bucket",
    "endpoint",
    "access_key_id",
    "access_key_secret",
    "security_token",
    "root",
    "skip_signature",
];

/// Strongly-typed configuration for building an OpenDAL store against Alibaba Cloud OSS.
///
/// This mirrors [`crate::opendal::CosConfig`], but OSS names its credentials `access_key_id` /
/// `access_key_secret` and has no equivalent of COS's `disable_config_load`; the closest knob is
/// [`OssConfig::skip_signature`], which sends unsigned requests for public buckets.
#[derive(Debug, Clone, Default)]
pub struct OssConfig {
    /// OSS bucket name. May be overridden by `properties["bucket"]` when adapting a URL.
    pub bucket: String,
    /// OSS endpoint, e.g. `https://oss-cn-hangzhou.aliyuncs.com`.
    pub endpoint: String,
    /// Alibaba Cloud access key id (mapped to `ALIBABA_CLOUD_ACCESS_KEY_ID`).
    pub access_key_id: Option<String>,
    /// Alibaba Cloud access key secret (mapped to `ALIBABA_CLOUD_ACCESS_KEY_SECRET`).
    pub access_key_secret: Option<String>,
    /// STS security token for temporary credentials (mapped to `ALIBABA_CLOUD_SECURITY_TOKEN`).
    pub security_token: Option<String>,
    /// Optional root prefix applied to all operations.
    pub root: Option<String>,
    /// When `true`, send unsigned requests. Use this for public, read-only buckets.
    pub skip_signature: bool,
}

/// Build an [`object_store::ObjectStore`] for Alibaba Cloud OSS directly from an [`OssConfig`].
///
/// This is the preferred entry point for callers that have a strongly-typed configuration object.
/// It does not synthesize a URL and so is not fragile against reordering of `bucket` vs URL host
/// precedence.
pub fn make_oss_store(config: OssConfig) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError> {
    if config.bucket.is_empty() {
        return Err(OpenDALStoreError::MissingConfig("bucket"));
    }
    if config.endpoint.is_empty() {
        return Err(OpenDALStoreError::MissingConfig("endpoint"));
    }

    let mut builder = services::Oss::default()
        .bucket(&config.bucket)
        .endpoint(&config.endpoint);

    if let Some(root) = config.root.as_deref() {
        builder = builder.root(root);
    }
    if let Some(access_key_id) = config.access_key_id.as_deref() {
        builder = builder.access_key_id(access_key_id);
    }
    if let Some(access_key_secret) = config.access_key_secret.as_deref() {
        builder = builder.access_key_secret(access_key_secret);
    }
    if let Some(security_token) = config.security_token.as_deref() {
        builder = builder.security_token(security_token);
    }
    if config.skip_signature {
        builder = builder.skip_signature();
    }

    let operator = build_operator(builder)?;
    Ok(Arc::new(OpendalStore::new(operator)))
}

/// Translate an (`oss://` URL, properties) pair into a strongly-typed [`OssConfig`].
///
/// Bucket is taken from `properties["bucket"]` first and falls back to the URL host; endpoint is
/// taken from `properties["endpoint"]` first and falls back to `OSS_ENDPOINT`; credentials fall
/// back to the `ALIBABA_CLOUD_*` variables OpenDAL's OSS builder reads.
///
/// `env_lookup` is the source of truth for environment-variable fallbacks. The production entry
/// point passes the real environment; tests pass a closure that returns from a fixed map, so they
/// do not race against the process environment.
pub(crate) fn url_and_properties_to_config<F>(
    url: &Url,
    properties: &HashMap<String, String>,
    env_lookup: F,
) -> Result<OssConfig, OpenDALStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    warn_on_unknown_properties(properties, KNOWN_PROPERTIES);

    let bucket = properties
        .get("bucket")
        .cloned()
        .or_else(|| url.host_str().map(str::to_string))
        .ok_or(OpenDALStoreError::MissingConfig("bucket"))?;

    let endpoint = property_or_env(properties, "endpoint", "OSS_ENDPOINT", &env_lookup)
        .ok_or(OpenDALStoreError::MissingConfig("endpoint"))?;

    Ok(OssConfig {
        bucket,
        endpoint,
        access_key_id: property_or_env(
            properties,
            "access_key_id",
            "ALIBABA_CLOUD_ACCESS_KEY_ID",
            &env_lookup,
        ),
        access_key_secret: property_or_env(
            properties,
            "access_key_secret",
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
            &env_lookup,
        ),
        security_token: property_or_env(
            properties,
            "security_token",
            "ALIBABA_CLOUD_SECURITY_TOKEN",
            &env_lookup,
        ),
        root: properties.get("root").cloned(),
        skip_signature: properties.get("skip_signature").map(String::as_str) == Some("true"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The endpoint is required, and the failure must be deterministic regardless of what the
    /// process environment happens to contain — hence the fixed `|_| None` lookup.
    #[test]
    fn oss_requires_endpoint() {
        let url = Url::parse("oss://my-bucket/path").unwrap();
        let props = HashMap::new();
        let result = url_and_properties_to_config(&url, &props, |_| None).unwrap_err();
        assert!(matches!(
            result,
            OpenDALStoreError::MissingConfig("endpoint")
        ));
    }

    /// OpenDAL's OSS service reads `ALIBABA_CLOUD_*` (with the underscore), not `ALIBABACLOUD_*`.
    /// Pinning the exact names here keeps the fallbacks honest.
    #[test]
    fn oss_falls_back_to_env() {
        let url = Url::parse("oss://my-bucket/path").unwrap();
        let env = |key: &str| match key {
            "OSS_ENDPOINT" => Some("https://oss-cn-hangzhou.aliyuncs.com".to_string()),
            "ALIBABA_CLOUD_ACCESS_KEY_ID" => Some("LTAI".to_string()),
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET" => Some("secret".to_string()),
            "ALIBABA_CLOUD_SECURITY_TOKEN" => Some("token".to_string()),
            _ => None,
        };
        let props = HashMap::new();
        let config = url_and_properties_to_config(&url, &props, env).expect("config");

        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.endpoint, "https://oss-cn-hangzhou.aliyuncs.com");
        assert_eq!(config.access_key_id.as_deref(), Some("LTAI"));
        assert_eq!(config.access_key_secret.as_deref(), Some("secret"));
        assert_eq!(config.security_token.as_deref(), Some("token"));
    }

    /// An explicit property must win over the environment fallback.
    #[test]
    fn oss_property_overrides_env() {
        let url = Url::parse("oss://url-bucket/path").unwrap();
        let env = |key: &str| match key {
            "OSS_ENDPOINT" => Some("https://from-env.example.com".to_string()),
            "ALIBABA_CLOUD_ACCESS_KEY_ID" => Some("from-env".to_string()),
            _ => None,
        };
        let mut props = HashMap::new();
        props.insert("bucket".to_string(), "prop-bucket".to_string());
        props.insert(
            "endpoint".to_string(),
            "https://from-prop.example.com".to_string(),
        );
        props.insert("access_key_id".to_string(), "from-prop".to_string());

        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        assert_eq!(config.bucket, "prop-bucket");
        assert_eq!(config.endpoint, "https://from-prop.example.com");
        assert_eq!(config.access_key_id.as_deref(), Some("from-prop"));
    }

    /// A full config must reach the OpenDAL builder and produce a store.
    #[test]
    fn oss_builds_with_explicit_config() {
        let url = Url::parse("oss://my-bucket/path/to/dataset.vortex").unwrap();
        let env = |key: &str| match key {
            "OSS_ENDPOINT" => Some("https://oss-cn-hangzhou.aliyuncs.com".to_string()),
            "ALIBABA_CLOUD_ACCESS_KEY_ID" => Some("LTAI".to_string()),
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET" => Some("secret".to_string()),
            _ => None,
        };
        let props = HashMap::new();
        let config = url_and_properties_to_config(&url, &props, env).expect("config");
        let store = make_oss_store(config).expect("store should build");
        assert!(Arc::strong_count(&store) >= 1);
    }

    /// Public buckets need no credentials when `skip_signature` is set.
    #[test]
    fn oss_builds_anonymous() {
        let store = make_oss_store(OssConfig {
            bucket: "public-bucket".to_string(),
            endpoint: "https://oss-cn-hangzhou.aliyuncs.com".to_string(),
            skip_signature: true,
            ..OssConfig::default()
        })
        .expect("anonymous store should build");
        assert!(Arc::strong_count(&store) >= 1);
    }

    /// `make_oss_store` must reject an empty bucket or endpoint before touching the builder.
    #[test]
    fn oss_config_rejects_empty_fields() {
        assert!(matches!(
            make_oss_store(OssConfig {
                bucket: String::new(),
                ..OssConfig::default()
            }),
            Err(OpenDALStoreError::MissingConfig("bucket"))
        ));
        assert!(matches!(
            make_oss_store(OssConfig {
                bucket: "b".to_string(),
                endpoint: String::new(),
                ..OssConfig::default()
            }),
            Err(OpenDALStoreError::MissingConfig("endpoint"))
        ));
    }
}
