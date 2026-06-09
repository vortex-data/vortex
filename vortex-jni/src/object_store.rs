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

pub(crate) fn object_store_fs(
    url: &Url,
    properties: &HashMap<String, String>,
    handle: Handle,
) -> VortexResult<FileSystemRef> {
    let object_store = make_object_store(url, properties)?;
    let object_store = Arc::new(Compat::new(object_store)) as Arc<dyn ObjectStore>;

    Ok(Arc::new(ObjectStoreFileSystem::new(object_store, handle)))
}

pub(crate) fn make_object_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> VortexResult<Arc<dyn ObjectStore>> {
    static OBJECT_STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let cache_key = url_cache_key(url, properties);

    {
        if let Some(cached) = OBJECT_STORES.lock().get(&cache_key) {
            return Ok(Arc::clone(cached));
        }
        // guard dropped at close of scope
    }

    let start = std::time::Instant::now();

    let (store, _) = FileLocation::resolve_with_props(
        url.as_str(),
        properties.iter().map(|(k, v)| (k.as_str(), v.as_str())),
    )?
    .into_remote()?;

    OBJECT_STORES.lock().insert(cache_key, Arc::clone(&store));

    tracing::debug!("make_object_store latency = {:?}", start.elapsed());

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
