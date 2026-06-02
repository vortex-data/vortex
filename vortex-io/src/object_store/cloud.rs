// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! URL-based resolution and construction of cloud [`ObjectStore`]s.
//!
//! Two entry points are provided:
//!
//! - [`resolve_url`] — the canonical resolver. It maps a URL or path string to either a
//!   local filesystem path or a registered/lazily-created [`ObjectStore`], using standard
//!   `object_store` URL parsing with case-insensitive environment variables.
//! - [`make_object_store`] — an opinionated per-scheme builder for callers that need
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

/// The process-global registry used by [`resolve_url`] when no explicit store is supplied.
static DEFAULT_REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// The outcome of resolving a URL or path string.
///
/// A local path resolves to [`ResolvedStore::Path`]; any other scheme resolves to an
/// [`ObjectStore`] plus the object's [`Path`] within that store.
#[derive(Debug)]
pub enum ResolvedStore {
    /// An object store and the path to the object within it.
    ObjectStore(Arc<dyn ObjectStore>, Path),
    /// A local filesystem path.
    Path(PathBuf),
}

/// Resolve a URL or path to either a local filesystem path or an object store.
///
/// If an explicit `store` is provided it is used directly, with `url_or_path` interpreted as a
/// path within that store. Otherwise:
///
/// - `file://` URLs and inputs that do not parse as a URL are treated as local paths.
/// - All other schemes (`s3://`, `gs://`, `az://`, `http(s)://`, …) are resolved through the
///   process-global registry, which lazily constructs and caches the store from the URL and
///   case-insensitive environment variables.
///
/// # Example
///
/// ```no_run
/// use vortex_io::object_store::cloud::{resolve_url, ResolvedStore};
///
/// # fn main() -> vortex_error::VortexResult<()> {
/// match resolve_url("s3://bucket/key/file.vortex", None)? {
///     ResolvedStore::ObjectStore(store, path) => { /* read from `store` at `path` */ }
///     ResolvedStore::Path(path) => { /* read local file at `path` */ }
/// }
/// # Ok(())
/// # }
/// ```
pub fn resolve_url(
    url_or_path: &str,
    store: Option<Arc<dyn ObjectStore>>,
) -> VortexResult<ResolvedStore> {
    match store {
        // If an explicit store is provided, use it.
        Some(store) => Ok(ResolvedStore::ObjectStore(store, Path::from(url_or_path))),
        None => match Url::parse(url_or_path) {
            Ok(url) if url.scheme() == "file" => {
                let path = url
                    .to_file_path()
                    .map_err(|_| vortex_err!("invalid file URL: {url_or_path}"))?;
                Ok(ResolvedStore::Path(path))
            }
            Ok(url) => {
                let (store, path) = DEFAULT_REGISTRY.resolve(&url)?;
                Ok(ResolvedStore::ObjectStore(store, path))
            }
            // Not a URL: treat the input as a local filesystem path.
            Err(_) => Ok(ResolvedStore::Path(PathBuf::from(url_or_path))),
        },
    }
}

/// Build an [`ObjectStore`] for a URL with opinionated, self-hosting-friendly defaults.
///
/// Unlike [`resolve_url`], this constructs the store with explicit per-scheme builders:
///
/// - S3 uses a generic endpoint and path-style addressing with `allow_http`, so it works
///   against MinIO/LocalStack as well as AWS.
/// - Azure uses an extended client timeout to avoid premature timeouts.
/// - Credentials are read from `properties` when present, falling back to the environment.
/// - Any remaining `properties` entries are passed through as scheme-specific config keys.
///
/// Resolved stores are cached process-wide, keyed by URL authority and properties.
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
        // Guard dropped at close of scope.
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
                // Use path-style URLs.
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
    use std::sync::Arc;

    use object_store::local::LocalFileSystem;
    use object_store::path::Path;

    use super::ResolvedStore;
    use super::resolve_url;

    impl ResolvedStore {
        fn unwrap_store(self) -> (Arc<dyn object_store::ObjectStore>, Path) {
            match self {
                ResolvedStore::ObjectStore(store, path) => (store, path),
                ResolvedStore::Path(_) => panic!("cannot unwrap ResolvedStore::Path as store"),
            }
        }

        fn unwrap_path(self) -> std::path::PathBuf {
            match self {
                ResolvedStore::ObjectStore(..) => {
                    panic!("cannot unwrap ResolvedStore::ObjectStore as path")
                }
                ResolvedStore::Path(path) => path,
            }
        }
    }

    #[test]
    fn test_resolve() -> vortex_error::VortexResult<()> {
        assert_eq!(
            resolve_url("/my/absolute/path", None)?.unwrap_path(),
            std::path::PathBuf::from("/my/absolute/path")
        );

        assert_eq!(
            resolve_url("file:///my/absolute/path", None)?.unwrap_path(),
            std::path::PathBuf::from("/my/absolute/path")
        );

        let (_store, path) =
            resolve_url("s3://my-bucket/first/second/third/", None)?.unwrap_store();
        assert_eq!(path, Path::from("first/second/third"));

        let local_store = Arc::new(LocalFileSystem::default());
        let (_store, path) = resolve_url("/root/test", Some(local_store))?.unwrap_store();
        assert_eq!(path, Path::from("root/test"));

        Ok(())
    }
}
