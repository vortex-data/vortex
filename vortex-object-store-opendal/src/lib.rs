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

use std::sync::Arc;

use object_store::ObjectStore;
use object_store_opendal::OpendalStore;
use opendal::Operator;
use opendal::services;
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

/// Build an [`object_store::ObjectStore`] for a COS URL.
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
    match url.scheme() {
        COS_SCHEME => make_cos_store(url, properties),
        other => Err(OpenDALStoreError::UnsupportedScheme(other.to_string())),
    }
}

fn make_cos_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError> {
    let bucket = properties
        .get("bucket")
        .cloned()
        .or_else(|| url.host_str().map(str::to_string))
        .ok_or(OpenDALStoreError::MissingConfig("bucket"))?;
    let endpoint = properties
        .get("endpoint")
        .cloned()
        .ok_or(OpenDALStoreError::MissingConfig("endpoint"))?;

    let mut builder = services::Cos::default().bucket(&bucket).endpoint(&endpoint);

    if let Some(root) = properties.get("root") {
        builder = builder.root(root);
    }
    if let Some(secret_id) = properties.get("secret_id") {
        builder = builder.secret_id(secret_id);
    }
    if let Some(secret_key) = properties.get("secret_key") {
        builder = builder.secret_key(secret_key);
    }
    if properties.get("disable_config_load").map(String::as_str) == Some("true") {
        builder = builder.disable_config_load();
    }

    let operator = build_operator(builder)?;
    Ok(Arc::new(OpendalStore::new(operator)))
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

    #[test]
    fn cos_requires_endpoint() {
        let url = Url::parse("cos://my-bucket/path").unwrap();
        let props = HashMap::new();
        assert!(matches!(
            make_cos_store(&url, &props),
            Err(OpenDALStoreError::MissingConfig("endpoint"))
        ));
    }

    #[test]
    fn cos_derives_bucket_from_host() {
        let url = Url::parse("cos://my-bucket/path").unwrap();
        let mut props = HashMap::new();
        props.insert(
            "endpoint".to_string(),
            "https://cos.ap-guangzhou.myqcloud.com".to_string(),
        );
        // Missing secret_id/secret_key -> build fails before we can assert bucket derivation,
        // but we should get past the MissingConfig checks (i.e. not a MissingConfig error).
        assert!(!matches!(
            make_cos_store(&url, &props),
            Err(OpenDALStoreError::MissingConfig(_))
        ));
    }
}
