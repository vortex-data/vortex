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
//! 2. schemes that `object_store` does not recognize natively — the OpenDAL-backed `cos://`,
//!    `oss://`, and `goosefs://`, and the Hugging Face Hub's `hf://` — are served by the crate's
//!    `opendal` and `hf` modules under the matching service feature.

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::parse_url_opts;
use object_store::path::Path;
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
/// resolves OpenDAL-backed schemes under the enabled service features.
///
/// Stores are cached per URL authority, so repeated resolves against the same bucket share one
/// client. A store can also be registered explicitly at a path prefix with
/// [`ObjectStoreRegistry::register`], in which case it wins over the built-and-cached fallback.
///
/// ```
/// # use object_store::registry::ObjectStoreRegistry;
/// # use vortex_cloud::Registry;
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
#[derive(Debug, Default)]
enum EnvSource {
    /// Read from the process environment.
    #[default]
    Process,
    /// A fixed set of variables.
    Fixed(Vec<(String, String)>),
}

impl EnvSource {
    /// Case-insensitive lookup of a single configuration variable.
    #[cfg(any(feature = "cos", feature = "goosefs", feature = "hf", feature = "oss"))]
    fn lookup(&self, key: &str) -> Option<String> {
        match self {
            EnvSource::Process => std::env::var(key).ok(),
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

        // 2. Build the store and determine how many raw URL path segments form its mount prefix.
        let (store, depth) = self.build_store(to_resolve)?;

        // 3. Cache the store and return it alongside the path that lives inside the store.
        self.cache_and_resolve(to_resolve, store, depth)
    }
}

impl Registry {
    /// Create a registry that reads store configuration from the process environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry over a fixed set of configuration variables, consulted instead of the
    /// process environment when building stores.
    ///
    /// Lookups are case-insensitive. Pass each key at most once — the set's consumers disagree
    /// on which duplicate wins.
    pub fn with_vars<I>(vars: I) -> Self
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

    /// Builds the [`ObjectStore`] that serves `to_resolve`, along with how many raw URL path
    /// segments form the prefix where the store is mounted.
    ///
    /// Schemes handled directly (`hf://`, OpenDAL) return their raw prefix depth without
    /// subtraction. For `parse_url_opts`, we subtract decoded key segments from the total decoded
    /// segment count rather than raw segment count, which works because bucket and container names
    /// cannot contain `/`.
    fn build_store(&self, to_resolve: &Url) -> object_store::Result<(Arc<dyn ObjectStore>, usize)> {
        // OpenDAL-backed schemes (Tencent COS, Alibaba OSS, Tencent GooseFS) are not recognized
        // by `object_store`, so build them from OpenDAL's own environment-variable configuration
        // (e.g. `TENCENTCLOUD_SECRET_ID`, `GOOSEFS_MASTER_ADDR`). The operator is rooted at the
        // bucket, which lives in the URL authority, so — exactly as for `s3://bucket/path` —
        // the whole URL path is the object key (depth 0).
        #[cfg(any(feature = "cos", feature = "goosefs", feature = "oss"))]
        if crate::opendal::supports_scheme(to_resolve.scheme()) {
            Path::from_url_path(to_resolve.path())?;
            let store =
                crate::opendal::make_opendal_store_with_env(to_resolve, &HashMap::new(), |key| {
                    self.env.lookup(key)
                })?;
            return Ok((store, 0));
        }

        // The Hugging Face Hub is not recognized by `object_store` either. Unlike the OpenDAL
        // schemes its store is not rooted at the URL authority — a repository and revision occupy
        // path segments too — so the parser returns the exact number of raw URL segments it
        // consumed for the mount prefix.
        #[cfg(feature = "hf")]
        if crate::hf::supports_scheme(to_resolve.scheme()) {
            let (store, _path, depth) =
                crate::hf::make_hf_store_from_url_with_env(to_resolve, |key| self.env.lookup(key))?;
            return Ok((store, depth));
        }

        let (store, path) = parse_url_opts(to_resolve, self.env.normalized_vars())?;
        // Count decoded segments per non-empty raw segment so `//` or leading `%2F` after the
        // container/bucket segment do not fail here when `parse_url_opts` accepted them.
        let decoded_segments: usize = path_segments(to_resolve.path())
            .map(|seg| {
                // `parse_url_opts` already validated the URL path; fall back to 1 defensively.
                Path::from_url_path(seg)
                    .map(|p| num_segments(p.as_ref()))
                    .unwrap_or(1)
            })
            .sum();
        let depth = decoded_segments.saturating_sub(num_segments(path.as_ref()));
        Ok((Arc::from(store), depth))
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
    // The segments come from a URL, so percent-decode them (one raw segment may decode to
    // several path parts, e.g. `refs%2Fconvert%2Fparquet`).
    let suffix = path_segments(url.path())
        .skip(depth)
        .collect::<Vec<_>>()
        .join("/");
    Path::from_url_path(suffix).map_err(|e| object_store::Error::Generic {
        store: "ObjectStoreRegistry",
        source: Box::new(e),
    })
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
