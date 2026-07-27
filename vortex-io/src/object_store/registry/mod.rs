// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Apache Software Foundation (ASF)

//! A URL-to-[`ObjectStore`] registry.
//!
//! This is an adapted version of the `DefaultObjectStoreRegistry` from the `object_store` crate,
//! modified in two ways:
//!
//! 1. configuration is resolved out of environment variables case-insensitively, matching how the
//!    various `Store::from_env` builders behave (see
//!    <https://github.com/apache/arrow-rs-object-store/issues/529>);
//! 2. schemes that `object_store` does not recognize natively — the OpenDAL-backed `cos://` and
//!    `oss://` — are served under the `opendal` feature.
//!
//! Every Vortex language binding resolves URLs through this one type, so a scheme added here is
//! reachable from Python, Java and DuckDB alike.

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::parse_url_opts;
use object_store::path::Path;
use object_store::path::PathPart;
use object_store::registry::ObjectStoreRegistry;
use parking_lot::RwLock;
use url::Url;
use vortex_utils::aliases::hash_map::HashMap;

#[derive(Debug, Default)]
struct PathEntry {
    /// Store, if defined at this path
    store: Option<Arc<dyn ObjectStore>>,
    /// Child [`PathEntry`], keyed by the next path segment in their path
    children: HashMap<String, Self>,
}

impl PathEntry {
    /// Lookup a store based on URL path
    ///
    /// Returns the store and its path segment depth
    fn lookup(&self, to_resolve: &Url) -> Option<(&Arc<dyn ObjectStore>, usize)> {
        let mut current = self;
        let mut ret = self.store.as_ref().map(|store| (store, 0));
        let mut depth = 0;
        // Traverse the PathEntry tree to find the longest match
        for segment in path_segments(to_resolve.path()) {
            match current.children.get(segment) {
                Some(e) => {
                    current = e;
                    depth += 1;
                    if let Some(store) = &current.store {
                        ret = Some((store, depth))
                    }
                }
                None => break,
            }
        }
        ret
    }
}

/// An [`ObjectStoreRegistry`] that normalizes environment variables before doing lookups, and
/// resolves OpenDAL-backed schemes under the `opendal` feature.
///
/// Stores are cached per URL authority, so repeated resolves against the same bucket share one
/// client. A store can also be registered explicitly at a path prefix with
/// [`ObjectStoreRegistry::register`], in which case it wins over the built-and-cached fallback.
///
/// ```
/// # use object_store::registry::ObjectStoreRegistry;
/// # use vortex_io::object_store::Registry;
/// # use url::Url;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let registry = Registry::default();
/// let store = object_store::memory::InMemory::new();
/// registry.register(Url::parse("memory://bucket/")?, std::sync::Arc::new(store));
///
/// let (_store, path) = registry.resolve(&Url::parse("memory://bucket/a/b.vortex")?)?;
/// assert_eq!(path.as_ref(), "a/b.vortex");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct Registry {
    /// Mapping from [`url_key`] to [`PathEntry`]
    map: RwLock<HashMap<String, PathEntry>>,
    /// Where store configuration is read from when building a store.
    env: EnvSource,
}

/// Source of the configuration variables consulted when building a store.
///
/// Tests construct a registry over a fixed set of variables rather than mutating the process
/// environment, which is unsound when tests run on multiple threads within one process (the
/// `std::env::set_var` block became `unsafe` in Rust 2024 for exactly this reason).
#[derive(Debug, Default)]
enum EnvSource {
    /// Read from the process environment.
    #[default]
    Process,
    /// A fixed set of variables.
    #[cfg(test)]
    Fixed(Vec<(String, String)>),
}

impl EnvSource {
    /// Case-insensitive lookup of a single configuration variable.
    #[cfg(feature = "opendal")]
    fn lookup(&self, key: &str) -> Option<String> {
        match self {
            EnvSource::Process => std::env::var(key).ok(),
            #[cfg(test)]
            EnvSource::Fixed(vars) => vars
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(key))
                .map(|(_, v)| v.clone()),
        }
    }

    /// The configuration variables, with keys lowercased so that lookups are case-insensitive.
    fn normalized_vars(&self) -> Vec<(String, String)> {
        match self {
            EnvSource::Process => std::env::vars()
                .map(|(k, v)| (k.to_ascii_lowercase(), v))
                .collect(),
            #[cfg(test)]
            EnvSource::Fixed(vars) => vars
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                .collect(),
        }
    }
}

impl ObjectStoreRegistry for Registry {
    fn register(&self, url: Url, store: Arc<dyn ObjectStore>) -> Option<Arc<dyn ObjectStore>> {
        let mut map = self.map.write();
        let entry = entry_at(
            &mut map,
            url_key(&url),
            url.path(),
            num_segments(url.path()),
        );
        entry.store.replace(store)
    }

    fn resolve(&self, to_resolve: &Url) -> object_store::Result<(Arc<dyn ObjectStore>, Path)> {
        let key = url_key(to_resolve);

        // 1. Look up the user-registered map first. Every other scheme does this, so an explicit
        //    `Registry::register("cos://...", store)` should also win over the build-and-cache
        //    fallback below. This also means a previously-resolved store (built by us for the
        //    same URL) is served from the cache here, without rebuilding the client.
        {
            let map = self.map.read();
            if let Some((store, depth)) = map.get(key).and_then(|entry| entry.lookup(to_resolve)) {
                let path = path_suffix(to_resolve, depth)?;
                return Ok((Arc::clone(store), path));
            }
        }

        // 2. Build the store and the path *within* that store. `build_store` reports the path
        //    the way `parse_url_opts` does — i.e. the path is what the store will see as the
        //    key — which is what keeps every scheme on the one caching rule in
        //    `cache_and_resolve`. The depth is then just the difference in segment counts.
        let (store, path) = self.build_store(to_resolve)?;
        let depth = num_segments(to_resolve.path()) - num_segments(path.as_ref());

        // 3. Cache the store and return it alongside the path that lives inside the store.
        self.cache_and_resolve(to_resolve, store, depth)
    }
}

impl Registry {
    /// Create a registry that reads store configuration from the process environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry over a fixed set of configuration variables.
    #[cfg(test)]
    fn with_env<I>(vars: I) -> Self
    where
        I: IntoIterator<Item = (String, String)>,
    {
        Self {
            map: RwLock::default(),
            env: EnvSource::Fixed(vars.into_iter().collect()),
        }
    }

    /// Caches `store` as the store serving the first `depth` path segments of `to_resolve`, and
    /// returns it alongside the remaining path.
    ///
    /// If a racing `resolve` cached a store at the same position first, that store wins and the
    /// one just built is dropped, so all callers of a given prefix share a single client.
    fn cache_and_resolve(
        &self,
        to_resolve: &Url,
        store: Arc<dyn ObjectStore>,
        depth: usize,
    ) -> object_store::Result<(Arc<dyn ObjectStore>, Path)> {
        let path = path_suffix(to_resolve, depth)?;

        let mut map = self.map.write();
        let entry = entry_at(&mut map, url_key(to_resolve), to_resolve.path(), depth);
        let stored = Arc::clone(match &entry.store {
            None => entry.store.insert(store),
            Some(existing) => existing, // Racing creation - use existing
        });

        Ok((stored, path))
    }

    /// Builds the [`ObjectStore`] that serves `to_resolve`, with the path within that store.
    ///
    /// This mirrors [`parse_url_opts`], extended with schemes the `object_store` crate does not
    /// recognize. Reporting the path the way `parse_url_opts` does is what keeps every scheme on
    /// the one caching rule in [`Registry::resolve`]: a scheme says where its store is rooted, and
    /// the registry decides how to cache it.
    fn build_store(&self, to_resolve: &Url) -> object_store::Result<(Arc<dyn ObjectStore>, Path)> {
        // OpenDAL-backed schemes (Tencent COS, Alibaba OSS) are not recognized by `object_store`,
        // so build them from OpenDAL's own environment-variable configuration (e.g.
        // `TENCENTCLOUD_SECRET_ID`). The operator is rooted at the bucket, which lives in the URL
        // authority, so — exactly as for `s3://bucket/path` — the whole URL path is the object key.
        #[cfg(feature = "opendal")]
        if vortex_object_store_opendal::supports_scheme(to_resolve.scheme()) {
            let store = vortex_object_store_opendal::make_opendal_store_with_env(
                to_resolve,
                &HashMap::new(),
                |key| self.env.lookup(key),
            )?;
            return Ok((store, Path::from_url_path(to_resolve.path())?));
        }

        let (store, path) = parse_url_opts(to_resolve, self.env.normalized_vars())?;
        Ok((Arc::from(store), path))
    }
}

/// Extracts the scheme and authority of a URL (components before the Path)
fn url_key(url: &Url) -> &str {
    &url[..url::Position::AfterPort]
}

/// Returns the non-empty segments of a path
///
/// Note: We don't use [`Url::path_segments`] as we only want non-empty paths
fn path_segments(s: &str) -> impl Iterator<Item = &str> {
    s.split('/').filter(|x| !x.is_empty())
}

/// Returns the number of non-empty path segments in a path
fn num_segments(s: &str) -> usize {
    path_segments(s).count()
}

/// Returns the path of `url` skipping the first `depth` segments
fn path_suffix(url: &Url, depth: usize) -> Result<Path, object_store::Error> {
    let segments = path_segments(url.path()).skip(depth);
    let path = segments
        .map(PathPart::parse)
        .collect::<Result<_, _>>()
        .map_err(|e| object_store::Error::Generic {
            store: "ObjectStoreRegistry",
            source: Box::new(e),
        })?;
    Ok(path)
}

/// Walks to the [`PathEntry`] for `key` sitting `depth` segments into `path`, creating the
/// intermediate entries as needed.
fn entry_at<'a>(
    map: &'a mut HashMap<String, PathEntry>,
    key: &str,
    path: &str,
    depth: usize,
) -> &'a mut PathEntry {
    let mut current = map.entry(key.to_string()).or_default();
    for segment in path_segments(path).take(depth) {
        current = current.children.entry(segment.to_string()).or_default();
    }
    current
}

#[cfg(test)]
mod tests;
