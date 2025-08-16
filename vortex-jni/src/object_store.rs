// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::str::FromStr;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::local::LocalFileSystem;
use object_store::{ClientOptions, ObjectStore, ObjectStoreScheme};
use parking_lot::Mutex;
use url::Url;
use vortex::error::{VortexError, VortexResult, vortex_bail};
use vortex::utils::aliases::hash_map::HashMap;

pub(crate) fn make_object_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> VortexResult<(Arc<dyn ObjectStore>, ObjectStoreScheme)> {
    static OBJECT_STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let start = std::time::Instant::now();

    let (scheme, _) = ObjectStoreScheme::parse(url)
        .map_err(|error| VortexError::from(object_store::Error::from(error)))?;

    let cache_key = url_cache_key(url);

    {
        if let Some(cached) = OBJECT_STORES.lock().get(&cache_key) {
            return Ok((cached.clone(), scheme));
        }
        // guard dropped at close of scope
    }

    // Configure extra properties on that scheme instead.
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            log::trace!("using LocalFileSystem object store");
            Arc::new(LocalFileSystem::default())
        }
        ObjectStoreScheme::AmazonS3 => {
            log::trace!("using AmazonS3 object store");
            let mut builder = AmazonS3Builder::new()
                .with_url(url.to_string())
                // Use generic S3 endpoint to avoid DNS resolution issues with region-specific endpoints
                .with_endpoint("https://s3.amazonaws.com")
                .with_virtual_hosted_style_request(false); // Use path-style URLs

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
                    log::warn!("Skipping unknown Amazon S3 config key: {key}");
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::MicrosoftAzure => {
            log::trace!("using MicrosoftAzure object store");

            // NOTE(aduffy): anecdotally Azure often times out after 30 seconds, this bumps us up
            //  to avoid that.
            let client_opts = ClientOptions::new().with_timeout(Duration::from_secs(120));
            let mut builder = MicrosoftAzureBuilder::new()
                .with_url(url.to_string())
                .with_client_options(client_opts);
            for (key, val) in properties {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    log::warn!("setting azure config {key:?} = {val}");
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Azure config key: {key}");
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            log::trace!("using GoogleCloudStorage object store");

            let mut builder = GoogleCloudStorageBuilder::new().with_url(url.to_string());
            for (key, val) in properties {
                if let Ok(config_key) = GoogleConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Google Cloud Storage config key: {key}");
                }
            }

            Arc::new(builder.build()?)
        }
        store => {
            vortex_bail!("Unsupported store scheme: {store:?}");
        }
    };

    {
        OBJECT_STORES.lock().insert(cache_key, store.clone());
        // Guard dropped at close of scope.
    }

    let duration = std::time::Instant::now().duration_since(start);
    log::debug!("make_object_store latency = {duration:?}");

    Ok((store, scheme))
}

fn url_cache_key(url: &Url) -> String {
    format!(
        "{}://{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort],
    )
}
