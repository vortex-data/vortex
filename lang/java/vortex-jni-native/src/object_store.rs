// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use object_store::ObjectStore;
use object_store::path::Path;
use object_store::registry::ObjectStoreRegistry;
use parking_lot::Mutex;
use url::Url;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::Handle;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_cloud::Registry;

/// Resolve `url` to a filesystem plus the path of the URL *within* it — not every scheme mounts
/// its store at the URL authority, so callers must key their reads by the returned path.
pub(crate) fn object_store_fs(
    url: &Url,
    properties: &HashMap<String, String>,
    handle: Handle,
) -> VortexResult<(FileSystemRef, String)> {
    let (object_store, path) = make_object_store(url, properties)?;
    let object_store = Arc::new(Compat::new(object_store)) as Arc<dyn ObjectStore>;

    Ok((
        Arc::new(ObjectStoreFileSystem::new(object_store, handle)),
        path.to_string(),
    ))
}

/// Registries keyed by the caller's properties: a store built with one caller's credentials
/// must not serve another's requests.
static REGISTRIES: LazyLock<Mutex<HashMap<String, Arc<Registry>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Process-wide cache of OpenDAL-backed stores, keyed by URL authority + properties.
static OPENDAL_STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Resolve `url` to a store plus the path of the URL within it, configured from the caller's
/// `object_store` properties over the process environment.
pub(crate) fn make_object_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> VortexResult<(Arc<dyn ObjectStore>, Path)> {
    let start = std::time::Instant::now();

    // OpenDAL schemes take the services' own property names (e.g. `secret_id`), not environment
    // names, and mount at the URL authority, which makes the authority-keyed cache sound.
    if vortex_cloud::opendal::supports_scheme(url.scheme()) {
        let path = Path::from_url_path(url.path())
            .map_err(|e| vortex_err!("cannot parse url path as object_store Path: {e}"))?;
        let cache_key = url_cache_key(url, properties);
        {
            if let Some(cached) = OPENDAL_STORES.lock().get(&cache_key) {
                return Ok((Arc::clone(cached), path));
            }
        }
        let store = vortex_cloud::opendal::make_opendal_store(url, properties)
            .map_err(|e| VortexError::from(object_store::Error::from(e)))?;
        OPENDAL_STORES.lock().insert(cache_key, Arc::clone(&store));
        return Ok((store, path));
    }

    let (store, path) = registry_for(properties)
        .resolve(url)
        .map_err(VortexError::from)?;

    let duration = start.elapsed();
    tracing::debug!("make_object_store latency = {duration:?}");

    Ok((store, path))
}

/// The registry serving `properties`, created on first use.
fn registry_for(properties: &HashMap<String, String>) -> Arc<Registry> {
    let mut sorted_props: Vec<_> = properties.iter().collect();
    sorted_props.sort_by_key(|(k, _)| *k);
    let key: String = sorted_props
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",");

    let mut registries = REGISTRIES.lock();
    Arc::clone(registries.entry(key).or_insert_with(|| {
        // Later inserts win; keys are lowercased because the registry matches case-insensitively
        // and requires each key at most once.
        let mut vars: HashMap<String, String> = std::env::vars()
            .map(|(k, v)| (k.to_ascii_lowercase(), v))
            .collect();
        vars.extend(
            properties
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone())),
        );
        Arc::new(Registry::with_vars(vars))
    }))
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
    use std::fmt::Write;

    use vortex::error::vortex_err;

    use super::*;

    fn parse(url: &str) -> VortexResult<Url> {
        Url::parse(url).map_err(|e| vortex_err!("{e}"))
    }

    #[test]
    fn test_hf_url_reports_the_in_repository_path() -> VortexResult<()> {
        let url = parse("hf://datasets/org/name/data/train.vortex")?;
        let (_store, path) = make_object_store(&url, &HashMap::new())?;

        assert_eq!(path.as_ref(), "data/train.vortex");
        Ok(())
    }

    #[test]
    fn test_hf_repositories_do_not_share_a_store() -> VortexResult<()> {
        let a = parse("hf://datasets/org/one/train.vortex")?;
        let b = parse("hf://datasets/org/two/train.vortex")?;

        let (store_a, _) = make_object_store(&a, &HashMap::new())?;
        let (store_b, _) = make_object_store(&b, &HashMap::new())?;

        assert!(!Arc::ptr_eq(&store_a, &store_b));
        Ok(())
    }

    /// `object_store` offers no way to read a store's configuration back; the Debug output is
    /// the only observable.
    ///
    /// The properties use canonical `aws_*` spellings: a property overrides the environment by
    /// exact key, and CI runners export `AWS_REGION`-style variables, so a short spelling
    /// (`region`) would race its environment alias at the store builder.
    #[test]
    #[expect(clippy::use_debug)]
    fn test_s3_properties_reach_the_store() -> VortexResult<()> {
        let url = parse("s3://bucket/dir/data%20file.vortex")?;
        let properties = HashMap::from_iter([
            ("aws_region".to_string(), "eu-central-9".to_string()),
            (
                "aws_endpoint".to_string(),
                "http://localhost:9000".to_string(),
            ),
            ("aws_allow_http".to_string(), "true".to_string()),
        ]);

        let (store, path) = make_object_store(&url, &properties)?;
        assert_eq!(path.as_ref(), "dir/data file.vortex");

        let mut debug_str = String::new();
        write!(&mut debug_str, "{store:?}").map_err(|e| vortex_err!("{e}"))?;
        assert!(debug_str.contains("eu-central-9"), "{debug_str}");
        assert!(debug_str.contains("localhost:9000"), "{debug_str}");
        Ok(())
    }

    #[test]
    fn test_stores_are_shared_per_property_set() -> VortexResult<()> {
        let url = parse("s3://bucket/key.vortex")?;
        let a = HashMap::from_iter([("aws_region".to_string(), "eu-central-9".to_string())]);
        let b = HashMap::from_iter([("aws_region".to_string(), "us-west-7".to_string())]);

        let (store_a1, _) = make_object_store(&url, &a)?;
        let (store_a2, _) = make_object_store(&url, &a)?;
        let (store_b, _) = make_object_store(&url, &b)?;

        assert!(Arc::ptr_eq(&store_a1, &store_a2));
        assert!(!Arc::ptr_eq(&store_a1, &store_b));
        Ok(())
    }
}
