// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OpenDAL-backed [`object_store::ObjectStore`] implementations for cloud providers that are not
//! natively supported by the `object_store` crate: Tencent Cloud COS and Alibaba Cloud OSS.
//!
//! OpenDAL exposes each service as an `Operator`. We adapt an `Operator` into an
//! `object_store::ObjectStore` via the `object_store_opendal::OpendalStore` bridge, which is built
//! against the same `object_store 0.13.x` version the rest of Vortex uses. This lets Vortex consume
//! these services through its existing `ObjectStoreFileSystem` abstraction.
//!
//! Callers that dispatch on a URL scheme should ask [`supports_scheme`] rather than comparing
//! against [`COS_SCHEME`] / [`OSS_SCHEME`] themselves, so that enabling another service does not
//! require touching every call site.
//!
//! # Cargo features
//!
//! * `cos` — Tencent Cloud COS, the `cos://` scheme.
//! * `oss` — Alibaba Cloud OSS, the `oss://` scheme.
//!
//! With a single service enabled the module still compiles: [`supports_scheme`] returns `false`
//! for every scheme it does not serve and [`make_opendal_store`] reports
//! [`OpenDALStoreError::UnsupportedScheme`].
//!
//! # Limitations
//!
//! The `OpendalStore` bridge owns its own HTTP request client. Configuration that the JNI/Python
//! layers normally pass through [`object_store::ClientOptions`] — connect/request timeouts,
//! retries, proxy settings, `allow_http` — has no effect on URLs handled here. Properties that a
//! service does not recognize are logged at `warn` and dropped; callers that need strict
//! validation must pre-filter their property maps.

#[cfg(feature = "cos")]
mod cos;
#[cfg(feature = "oss")]
mod oss;

use std::sync::Arc;

#[cfg(any(feature = "cos", feature = "oss"))]
use ::opendal::Operator;
use object_store::ObjectStore;
#[cfg(any(feature = "cos", feature = "oss"))]
use tracing::warn;
use url::Url;
use vortex_utils::aliases::hash_map::HashMap;

#[cfg(feature = "cos")]
pub use crate::opendal::cos::COS_SCHEME;
#[cfg(feature = "cos")]
pub use crate::opendal::cos::CosConfig;
#[cfg(feature = "cos")]
pub use crate::opendal::cos::make_cos_store;
#[cfg(feature = "oss")]
pub use crate::opendal::oss::OSS_SCHEME;
#[cfg(feature = "oss")]
pub use crate::opendal::oss::OssConfig;
#[cfg(feature = "oss")]
pub use crate::opendal::oss::make_oss_store;

/// Error type for building an OpenDAL-backed object store.
#[derive(Debug)]
pub enum OpenDALStoreError {
    /// The URL scheme is not one this crate handles (e.g. `s3`, `gs`, ...), or its Cargo feature
    /// is not enabled.
    UnsupportedScheme(String),
    /// A required configuration value (bucket and/or endpoint) was missing.
    MissingConfig(&'static str),
    /// The OpenDAL builder rejected the provided configuration.
    Build(::opendal::Error),
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

/// The URL schemes this crate can build stores for, given the enabled Cargo features.
pub const SUPPORTED_SCHEMES: &[&str] = &[
    #[cfg(feature = "cos")]
    COS_SCHEME,
    #[cfg(feature = "oss")]
    OSS_SCHEME,
];

/// Returns `true` if `scheme` is served by an OpenDAL-backed store in this build.
///
/// This is the dispatch predicate every caller should use: it tracks the enabled Cargo features,
/// so a consumer that adds a service feature picks it up without changing its own scheme matching.
///
/// ```
/// # use vortex_cloud::opendal::supports_scheme;
/// assert!(!supports_scheme("s3"));
/// ```
pub fn supports_scheme(scheme: &str) -> bool {
    SUPPORTED_SCHEMES.contains(&scheme)
}

/// Build an [`object_store::ObjectStore`] for an OpenDAL-backed URL (`cos://`, `oss://`).
///
/// `properties` are per-request configuration overrides (matching the `HashMap<String, String>`
/// passed through the JNI/Python layers). Missing values fall back to the environment variables
/// the corresponding service reads (e.g. `TENCENTCLOUD_SECRET_ID`, `ALIBABA_CLOUD_ACCESS_KEY_ID`).
///
/// Returns [`OpenDALStoreError::UnsupportedScheme`] if `url` uses a scheme this build does not
/// serve; test it up-front with [`supports_scheme`].
pub fn make_opendal_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError> {
    make_opendal_store_with_env(url, properties, env_var_lookup)
}

/// Build an OpenDAL-backed store, resolving environment fallbacks through `env_lookup` instead of
/// the process environment.
///
/// Callers that already own a configuration source — such as a registry that resolves variables
/// case-insensitively — should use this so that store construction does not silently depend on
/// global state. [`make_opendal_store`] is the same call against the real environment.
pub fn make_opendal_store_with_env<F>(
    url: &Url,
    properties: &HashMap<String, String>,
    env_lookup: F,
) -> Result<Arc<dyn ObjectStore>, OpenDALStoreError>
where
    F: Fn(&str) -> Option<String>,
{
    match url.scheme() {
        #[cfg(feature = "cos")]
        COS_SCHEME => make_cos_store(cos::url_and_properties_to_config(
            url, properties, env_lookup,
        )?),
        #[cfg(feature = "oss")]
        OSS_SCHEME => make_oss_store(oss::url_and_properties_to_config(
            url, properties, env_lookup,
        )?),
        other => {
            // Consumes the arguments so they count as used even in a build with no service
            // features enabled, where every scheme lands here.
            drop((properties, env_lookup));
            Err(OpenDALStoreError::UnsupportedScheme(other.to_string()))
        }
    }
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

/// Take `key` from `properties`, falling back to `env_lookup(env_var)`.
#[cfg(any(feature = "cos", feature = "oss"))]
pub(crate) fn property_or_env<F>(
    properties: &HashMap<String, String>,
    key: &str,
    env_var: &str,
    env_lookup: &F,
) -> Option<String>
where
    F: Fn(&str) -> Option<String>,
{
    properties.get(key).cloned().or_else(|| env_lookup(env_var))
}

/// Log a warning for every property key the service does not recognize.
#[cfg(any(feature = "cos", feature = "oss"))]
pub(crate) fn warn_on_unknown_properties(properties: &HashMap<String, String>, known: &[&str]) {
    for key in properties.keys() {
        if !known.contains(&key.as_str()) {
            warn!("ignoring unknown OpenDAL store property: {key}");
        }
    }
}

/// Finish an OpenDAL builder into an [`Operator`], mapping builder errors into our error type.
#[cfg(any(feature = "cos", feature = "oss"))]
pub(crate) fn build_operator<B>(builder: B) -> Result<Operator, OpenDALStoreError>
where
    B: ::opendal::Builder,
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
    fn supports_scheme_tracks_enabled_features() {
        assert!(!supports_scheme("s3"));
        assert_eq!(supports_scheme("cos"), cfg!(feature = "cos"));
        assert_eq!(supports_scheme("oss"), cfg!(feature = "oss"));
    }

    /// Every scheme advertised by [`SUPPORTED_SCHEMES`] must actually reach a builder rather than
    /// falling through to the `UnsupportedScheme` arm of [`make_opendal_store`]. Without a
    /// configured endpoint the build fails, but it must fail with `MissingConfig`, not
    /// `UnsupportedScheme` — that difference is what pins the dispatch table to the feature set.
    #[test]
    fn every_supported_scheme_dispatches() {
        for scheme in SUPPORTED_SCHEMES {
            let url = Url::parse(&format!("{scheme}://bucket/path")).unwrap();
            let props = HashMap::new();
            assert!(
                !matches!(
                    make_opendal_store(&url, &props),
                    Err(OpenDALStoreError::UnsupportedScheme(_))
                ),
                "{scheme} is advertised but not dispatched"
            );
        }
    }
}
