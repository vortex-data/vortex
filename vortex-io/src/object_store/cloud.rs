// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! URL-based resolution and construction of cloud [`ObjectStore`]s.
//!
//! Two entry points are provided:
//!
//! - [`FileLocation::resolve`] — the canonical resolver. It maps a URL or path string to either a
//!   local filesystem path or a registered/lazily-created [`ObjectStore`], using standard
//!   `object_store` URL parsing with case-insensitive environment variables. This is the
//!   recommended entry point for nearly all callers.
//! - [`make_object_store`] — an opt-in, opinionated per-scheme builder for callers that need
//!   MinIO/LocalStack-friendly defaults (path-style S3 addressing, generic endpoint,
//!   `allow_http`) and arbitrary config-key passthrough.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use object_store::ClientOptions;
use object_store::ObjectStore;
use object_store::ObjectStoreScheme;
use object_store::aws::AmazonS3Builder;
use object_store::aws::AmazonS3ConfigKey;
use object_store::azure::AzureConfigKey;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::gcp::GoogleConfigKey;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::registry::ObjectStoreRegistry;
use parking_lot::Mutex;
use url::Url;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::object_store::registry::Registry;

/// The process-global registry used by [`FileLocation::resolve`] to cache lazily-constructed
/// stores.
static DEFAULT_REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// Where the bytes of a file live: on the local filesystem, or in an object store.
///
/// Produced by [`FileLocation::resolve`]. Local paths and `file://` URLs resolve to
/// [`FileLocation::Local`]; any other scheme resolves to [`FileLocation::Remote`].
#[derive(Debug)]
pub enum FileLocation {
    /// A local filesystem path.
    Local(PathBuf),
    /// An object store and the object's path within it.
    Remote {
        /// The object store to read from.
        store: Arc<dyn ObjectStore>,
        /// The object's path within `store`.
        path: Path,
    },
}

impl FileLocation {
    /// Resolve a URL or path string to a [`FileLocation`].
    ///
    /// - `file://` URLs and inputs that do not parse as a URL resolve to [`FileLocation::Local`].
    /// - All other schemes (`s3://`, `gs://`, `az://`, `http(s)://`, ...) resolve through the
    ///   process-global registry, which lazily constructs and caches the store from the URL and
    ///   case-insensitive environment variables, returning [`FileLocation::Remote`].
    ///
    /// This is the canonical entry point. Callers needing MinIO/LocalStack-style defaults or
    /// explicit per-property credentials can use [`make_object_store`] instead.
    pub fn resolve(url_or_path: impl AsRef<str>) -> VortexResult<Self> {
        let url_or_path = url_or_path.as_ref();
        match Url::parse(url_or_path) {
            Ok(url) if url.scheme() == "file" => {
                let path = url
                    .to_file_path()
                    .map_err(|_| vortex_err!("invalid file URL: {url_or_path}"))?;
                Ok(FileLocation::Local(path))
            }
            Ok(url) => {
                let (store, path) = DEFAULT_REGISTRY.resolve(&url)?;
                Ok(FileLocation::Remote { store, path })
            }
            // Not a URL: treat the input as a local filesystem path.
            Err(_) => Ok(FileLocation::Local(PathBuf::from(url_or_path))),
        }
    }

    /// Returns `true` if this is a local filesystem path.
    pub fn is_local(&self) -> bool {
        matches!(self, FileLocation::Local(_))
    }

    /// Returns `true` if this is a remote object store location.
    pub fn is_remote(&self) -> bool {
        matches!(self, FileLocation::Remote { .. })
    }

    /// Require a remote object store, returning the store and object path.
    ///
    /// Returns an error if this is a [`FileLocation::Local`] path.
    pub fn into_remote(self) -> VortexResult<(Arc<dyn ObjectStore>, Path)> {
        match self {
            FileLocation::Remote { store, path } => Ok((store, path)),
            FileLocation::Local(path) => {
                vortex_bail!(
                    "expected a remote object store, got local path: {}",
                    path.display()
                )
            }
        }
    }
}

/// Build an [`ObjectStore`] for a URL with opinionated, self-hosting-friendly defaults.
///
/// Unlike [`FileLocation::resolve`], this constructs the store with explicit per-scheme builders:
///
/// - S3 uses a generic endpoint and path-style addressing with `allow_http`, so it works
///   against MinIO/LocalStack as well as AWS.
/// - Azure uses an extended client timeout to avoid premature timeouts.
/// - Credentials are read from `properties` when present, falling back to the environment.
/// - Any remaining `properties` entries are passed through as scheme-specific config keys.
///
/// Resolved stores are cached process-wide, keyed by URL authority and properties.
///
/// Prefer [`FileLocation::resolve`] unless you specifically need these opinionated defaults or
/// per-property configuration passthrough.
// The cognitive-complexity lint only triggers under some feature unifications (it depends on how
// the `tracing` macros expand), so we use `allow` rather than `expect` to avoid an unfulfilled
// expectation in minimal-feature builds.
#[allow(clippy::cognitive_complexity)]
pub fn make_object_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> VortexResult<Arc<dyn ObjectStore>> {
    static OBJECT_STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let start = std::time::Instant::now();

    let (scheme, _) = ObjectStoreScheme::parse(url)
        .map_err(|error| VortexError::from(object_store::Error::from(error)))?;

    let cache_key = url_cache_key(url, properties);

    {
        if let Some(cached) = OBJECT_STORES.lock().get(&cache_key) {
            return Ok(Arc::clone(cached));
        }
        // guard dropped at close of scope
    }

    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            tracing::trace!("using LocalFileSystem object store");
            Arc::new(LocalFileSystem::default())
        }
        ObjectStoreScheme::AmazonS3 => {
            tracing::trace!("using AmazonS3 object store");
            let mut builder = AmazonS3Builder::new()
                .with_url(url.to_string())
                // Use a generic S3 endpoint to avoid DNS resolution issues with
                // region-specific endpoints.
                .with_endpoint("https://s3.amazonaws.com")
                // Use path-style URLs
                .with_virtual_hosted_style_request(false)
                // Allow overriding to HTTP endpoints, e.g. LocalStack, MinIO.
                .with_allow_http(true);

            // Load credentials from the environment if not provided in properties.
            if !properties.contains_key("access_key_id")
                && let Ok(access_key) = std::env::var("AWS_ACCESS_KEY_ID")
            {
                builder = builder.with_access_key_id(access_key);
            }
            if !properties.contains_key("secret_access_key")
                && let Ok(secret_key) = std::env::var("AWS_SECRET_ACCESS_KEY")
            {
                builder = builder.with_secret_access_key(secret_key);
            }
            if !properties.contains_key("region")
                && let Ok(region) = std::env::var("AWS_DEFAULT_REGION")
            {
                builder = builder.with_region(region);
            }

            for (key, val) in properties {
                if let Ok(config_key) = AmazonS3ConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Amazon S3 config key: {key}");
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::MicrosoftAzure => {
            tracing::trace!("using MicrosoftAzure object store");

            // Azure can time out after 30 seconds; bump the client timeout to avoid that.
            let client_opts = ClientOptions::new().with_timeout(Duration::from_secs(120));
            let mut builder = MicrosoftAzureBuilder::new()
                .with_url(url.to_string())
                .with_client_options(client_opts);
            for (key, val) in properties {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Azure config key: {key}");
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            tracing::trace!("using GoogleCloudStorage object store");

            let mut builder = GoogleCloudStorageBuilder::new().with_url(url.to_string());
            for (key, val) in properties {
                if let Ok(config_key) = GoogleConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Google Cloud Storage config key: {key}");
                }
            }

            Arc::new(builder.build()?)
        }
        store => {
            vortex_bail!("Unsupported store scheme: {store:?}");
        }
    };

    OBJECT_STORES.lock().insert(cache_key, Arc::clone(&store));

    let duration = start.elapsed();
    tracing::debug!("make_object_store latency = {duration:?}");

    Ok(store)
}

fn url_cache_key(url: &Url, properties: &HashMap<String, String>) -> String {
    let mut sorted_props: Vec<_> = properties.iter().collect();
    sorted_props.sort_by_key(|(k, _)| *k);

    let props_str: String = sorted_props
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}://{};{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort],
        props_str,
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use object_store::path::Path;

    use super::FileLocation;

    impl FileLocation {
        fn unwrap_local(self) -> PathBuf {
            match self {
                FileLocation::Local(path) => path,
                FileLocation::Remote { .. } => panic!("expected Local, got Remote"),
            }
        }
    }

    #[test]
    fn test_resolve() -> vortex_error::VortexResult<()> {
        assert_eq!(
            FileLocation::resolve("/my/absolute/path")?.unwrap_local(),
            PathBuf::from("/my/absolute/path")
        );

        assert_eq!(
            FileLocation::resolve("file:///my/absolute/path")?.unwrap_local(),
            PathBuf::from("/my/absolute/path")
        );

        let (_store, path) =
            FileLocation::resolve("s3://my-bucket/first/second/third/")?.into_remote()?;
        assert_eq!(path, Path::from("first/second/third"));

        Ok(())
    }
}
