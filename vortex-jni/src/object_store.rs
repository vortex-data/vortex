// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use parking_lot::Mutex;
use url::Url;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::Handle;
use vortex::utils::aliases::hash_map::HashMap;

pub(crate) fn object_store_fs(
    url: &Url,
    properties: &HashMap<String, String>,
    handle: Handle,
) -> VortexResult<FileSystemRef> {
    let object_store = make_object_store(url, properties)?;
    let object_store = Arc::new(Compat::new(object_store)) as Arc<dyn ObjectStore>;

    Ok(Arc::new(ObjectStoreFileSystem::new(object_store, handle)))
}

#[expect(clippy::cognitive_complexity)]
pub(crate) fn make_object_store(
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

    // Configure extra properties on that scheme instead.
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            tracing::trace!("using LocalFileSystem object store");
            Arc::new(LocalFileSystem::default())
        }
        ObjectStoreScheme::AmazonS3 => {
            tracing::trace!("using AmazonS3 object store");
            let mut builder = AmazonS3Builder::new()
                .with_url(url.to_string())
                // Use generic S3 endpoint to avoid DNS resolution issues with region-specific endpoints
                .with_endpoint("https://s3.amazonaws.com")
                // Use path-style URLs
                .with_virtual_hosted_style_request(false)
                // Allow user to override endpoint to HTTP endpoints, e.g. LocalStack, Minio
                .with_allow_http(true);

            // Try to load credentials from environment if not provided in properties
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

            // NOTE(aduffy): anecdotally Azure often times out after 30 seconds, this bumps us up
            //  to avoid that.
            let client_opts = ClientOptions::new().with_timeout(Duration::from_secs(120));
            let mut builder = MicrosoftAzureBuilder::new()
                .with_url(url.to_string())
                .with_client_options(client_opts);
            for (key, val) in properties {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    tracing::warn!("setting azure config {key:?} = {val}");
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
