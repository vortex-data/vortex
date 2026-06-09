// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use object_store::ObjectStore;
use parking_lot::Mutex;
use url::Url;
use vortex::error::VortexResult;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::FileLocation;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::Handle;
use vortex::utils::aliases::hash_map::HashMap;

/// Process-global cache of object stores keyed by (url authority, sorted properties).
/// Avoids recreating HTTP connection pools across Java calls to the same bucket.
static OBJECT_STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub(crate) fn object_store_fs(
    url: &Url,
    properties: &HashMap<String, String>,
    handle: Handle,
) -> VortexResult<FileSystemRef> {
    let object_store = get_or_create_store(url, properties)?;
    let object_store = Arc::new(Compat::new(object_store)) as Arc<dyn ObjectStore>;
    Ok(Arc::new(ObjectStoreFileSystem::new(object_store, handle)))
}

/// Returns a cached store for `(url, properties)`, creating one via [`FileLocation::resolve_with_props`]
/// on first access. The cache is keyed by URL authority and sorted properties to ensure
/// the same credentials always reuse the same connection pool.
fn get_or_create_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> VortexResult<Arc<dyn ObjectStore>> {
    let cache_key = store_cache_key(url, properties);

    {
        if let Some(cached) = OBJECT_STORES.lock().get(&cache_key) {
            return Ok(Arc::clone(cached));
        }
        // guard dropped at close of scope
    }

    let (store, _) = FileLocation::resolve_with_props(
        url.as_str(),
        properties.iter().map(|(k, v)| (k.as_str(), v.as_str())),
    )?
    .into_remote()?;

    OBJECT_STORES.lock().insert(cache_key, Arc::clone(&store));
    Ok(store)
}

fn store_cache_key(url: &Url, properties: &HashMap<String, String>) -> String {
    let mut sorted_props: Vec<_> = properties.iter().collect();
    sorted_props.sort_by_key(|(k, _)| *k);
    let props_str = sorted_props
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
