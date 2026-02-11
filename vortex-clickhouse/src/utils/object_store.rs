// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Object store utilities for remote file access.
//!
//! This module provides utilities for:
//! - Detecting remote URLs (S3, GCS, Azure, HTTP)
//! - Creating `ObjectStore` instances from URLs
//! - Parsing and validating remote paths

use std::str::FromStr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Duration;

use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::http::HttpBuilder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{ClientOptions, ObjectStore, ObjectStoreScheme};
use parking_lot::Mutex;
use url::Url;
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex_utils::aliases::hash_map::HashMap;

/// Cached object stores to avoid recreating connections.
static OBJECT_STORE_CACHE: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Parse a URL string into a `Url`.
pub fn parse_url(url_str: &str) -> VortexResult<Url> {
    Url::parse(url_str).map_err(|e| vortex_err!("Invalid URL '{}': {}", url_str, e))
}

/// Check if a path is a remote URL (S3, HTTP, etc.).
pub fn is_remote_path(path: &str) -> bool {
    path.starts_with("s3://")
        || path.starts_with("http://")
        || path.starts_with("https://")
        || path.starts_with("gs://")
        || path.starts_with("az://")
        || path.starts_with("abfs://")
        || path.starts_with("abfss://")
        || path.starts_with("hdfs://")
}

/// Extract the object path from a URL.
///
/// For example, `s3://bucket/path/to/file.vortex` returns `path/to/file.vortex`.
pub fn extract_object_path(url: &Url) -> VortexResult<ObjectPath> {
    ObjectPath::from_url_path(url.path())
        .map_err(|e| vortex_err!("Invalid object path in URL '{}': {}", url, e))
}

/// Result of creating an object store from a URL.
pub struct ObjectStoreWithPath {
    /// The object store instance.
    pub store: Arc<dyn ObjectStore>,
    /// The path within the object store.
    pub path: ObjectPath,
    /// The original URL.
    pub url: Url,
}

/// Generate a cache key from a URL.
///
/// The cache key is based on scheme + host + port, so different paths
/// on the same server share the same ObjectStore instance.
fn url_cache_key(url: &Url) -> String {
    format!(
        "{}://{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort],
    )
}

/// Create an `ObjectStore` from a URL.
///
/// This function creates and caches object store instances. It supports:
/// - S3 (`s3://bucket/path`)
/// - Google Cloud Storage (`gs://bucket/path`)
/// - Azure Blob Storage (`az://container/path`, `abfs://`, `abfss://`)
/// - HTTP/HTTPS (`http://`, `https://`)
/// - Local filesystem (`file://` or plain paths)
///
/// # Arguments
/// * `url_str` - The URL string to parse
///
/// # Returns
/// An `ObjectStoreWithPath` containing the store, path, and parsed URL.
#[expect(clippy::cognitive_complexity)]
pub fn make_object_store(url_str: &str) -> VortexResult<ObjectStoreWithPath> {
    make_object_store_with_options(url_str, &HashMap::new())
}

/// Create an `ObjectStore` from a URL with custom configuration options.
///
/// This function creates and caches object store instances with custom configuration.
///
/// # Arguments
/// * `url_str` - The URL string to parse
/// * `properties` - Configuration options (e.g., AWS credentials, region, etc.)
///
/// # Returns
/// An `ObjectStoreWithPath` containing the store, path, and parsed URL.
#[expect(clippy::cognitive_complexity)]
pub fn make_object_store_with_options(
    url_str: &str,
    properties: &HashMap<String, String>,
) -> VortexResult<ObjectStoreWithPath> {
    let url = parse_url(url_str)?;
    let path = extract_object_path(&url)?;

    let (scheme, _) = ObjectStoreScheme::parse(&url)
        .map_err(|error| vortex_err!("Failed to parse object store scheme: {}", error))?;

    let cache_key = url_cache_key(&url);

    // Check cache first
    {
        if let Some(cached) = OBJECT_STORE_CACHE.lock().get(&cache_key) {
            return Ok(ObjectStoreWithPath {
                store: cached.clone(),
                path,
                url,
            });
        }
    }

    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            tracing::trace!("using LocalFileSystem object store");
            Arc::new(LocalFileSystem::default())
        }
        ObjectStoreScheme::AmazonS3 => {
            tracing::trace!("using AmazonS3 object store for URL: {}", url);
            let mut builder = AmazonS3Builder::new()
                .with_url(url.to_string())
                .with_virtual_hosted_style_request(false);

            // Try to load credentials from environment if not provided in properties
            if !properties.contains_key("access_key_id") {
                if let Ok(access_key) = std::env::var("AWS_ACCESS_KEY_ID") {
                    builder = builder.with_access_key_id(access_key);
                }
            }
            if !properties.contains_key("secret_access_key") {
                if let Ok(secret_key) = std::env::var("AWS_SECRET_ACCESS_KEY") {
                    builder = builder.with_secret_access_key(secret_key);
                }
            }
            if !properties.contains_key("region") {
                if let Ok(region) = std::env::var("AWS_DEFAULT_REGION") {
                    builder = builder.with_region(region);
                }
            }

            for (key, val) in properties {
                if let Ok(config_key) = AmazonS3ConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Amazon S3 config key: {}", key);
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::MicrosoftAzure => {
            tracing::trace!("using MicrosoftAzure object store for URL: {}", url);

            let client_opts = ClientOptions::new().with_timeout(Duration::from_secs(120));
            let mut builder = MicrosoftAzureBuilder::new()
                .with_url(url.to_string())
                .with_client_options(client_opts);

            for (key, val) in properties {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Azure config key: {}", key);
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            tracing::trace!("using GoogleCloudStorage object store for URL: {}", url);

            let mut builder = GoogleCloudStorageBuilder::new().with_url(url.to_string());

            for (key, val) in properties {
                if let Ok(config_key) = GoogleConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Google Cloud Storage config key: {}", key);
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::Http => {
            tracing::trace!("using HTTP object store for URL: {}", url);

            let client_opts = ClientOptions::new().with_timeout(Duration::from_secs(120));
            let builder = HttpBuilder::new()
                .with_url(url.to_string())
                .with_client_options(client_opts);

            Arc::new(builder.build()?)
        }
        store => {
            vortex_bail!("Unsupported object store scheme: {:?}", store);
        }
    };

    // Cache the store
    {
        OBJECT_STORE_CACHE.lock().insert(cache_key, store.clone());
    }

    Ok(ObjectStoreWithPath { store, path, url })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_remote_path() {
        // Remote paths
        assert!(is_remote_path("s3://bucket/path/to/file.vortex"));
        assert!(is_remote_path("https://example.com/data.vortex"));
        assert!(is_remote_path("http://localhost:8080/data.vortex"));
        assert!(is_remote_path("gs://my-bucket/file.vortex"));
        assert!(is_remote_path("az://container/blob.vortex"));
        assert!(is_remote_path("abfs://container/blob.vortex"));
        assert!(is_remote_path("abfss://container/blob.vortex"));

        // Local paths
        assert!(!is_remote_path("/path/to/file.vortex"));
        assert!(!is_remote_path("./relative/path.vortex"));
        assert!(!is_remote_path("file.vortex"));
        assert!(!is_remote_path("C:\\Windows\\path.vortex"));
    }

    #[test]
    fn test_parse_url() {
        let url = parse_url("s3://bucket/path/to/file.vortex").unwrap();
        assert_eq!(url.scheme(), "s3");
        assert_eq!(url.host_str(), Some("bucket"));
        assert_eq!(url.path(), "/path/to/file.vortex");

        let url = parse_url("https://example.com/data.vortex").unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("example.com"));
        assert_eq!(url.path(), "/data.vortex");

        // Invalid URL
        assert!(parse_url("not a valid url").is_err());
    }

    #[test]
    fn test_extract_object_path() {
        let url = parse_url("s3://bucket/path/to/file.vortex").unwrap();
        let path = extract_object_path(&url).unwrap();
        assert_eq!(path.as_ref(), "path/to/file.vortex");

        let url = parse_url("https://example.com/data/nested/file.vortex").unwrap();
        let path = extract_object_path(&url).unwrap();
        assert_eq!(path.as_ref(), "data/nested/file.vortex");
    }

    #[test]
    fn test_url_cache_key() {
        let url = parse_url("s3://bucket/path/to/file.vortex").unwrap();
        assert_eq!(url_cache_key(&url), "s3://bucket");

        let url = parse_url("https://example.com:8080/data.vortex").unwrap();
        assert_eq!(url_cache_key(&url), "https://example.com:8080");

        // Different paths on same server should have same cache key
        let url1 = parse_url("s3://bucket/path1/file1.vortex").unwrap();
        let url2 = parse_url("s3://bucket/path2/file2.vortex").unwrap();
        assert_eq!(url_cache_key(&url1), url_cache_key(&url2));
    }

    #[test]
    fn test_make_object_store_local() {
        let result = make_object_store("file:///tmp/test.vortex");
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.path.as_ref(), "tmp/test.vortex");
    }
}
