// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Apache Software Foundation (ASF)

//! This file is an adapted version of the `DefaultObjectStoreRegistry` from the object_store crate,
//! but modified to resolve configurations out of environment variables case-insensitively. This
//! is similar to how all the `Store::from_env` builders work for the various object stores.
//!
//! See also <https://github.com/apache/arrow-rs-object-store/issues/529>

#![expect(clippy::disallowed_types)]

use std::collections::HashMap;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::parse_url_opts;
use object_store::path::Path;
use object_store::registry::ObjectStoreRegistry;
use parking_lot::RwLock;
use url::Url;
#[cfg(feature = "opendal")]
use vortex_utils::aliases::hash_map::HashMap as VortexHashMap;

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

/// An implementation of the [`ObjectStoreRegistry`] that normalizes environment variables
/// before doing lookups.
#[derive(Debug, Default)]
pub(crate) struct Registry {
    /// Mapping from [`url_key`] to [`PathEntry`]
    map: RwLock<HashMap<String, PathEntry>>,
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
        //    `cache_and_resolve`. The depth is then the difference in segment counts.
        let (store, path) = build_store(to_resolve)?;
        // The raw URL path counts a percent-encoded segment (e.g. `refs%2Fconvert%2Fparquet`)
        // as one segment while `path` is percent-decoded and may count more, so saturate to
        // mount the store at the URL root rather than underflowing.
        let depth = num_segments(to_resolve.path()).saturating_sub(num_segments(path.as_ref()));

        // 3. Cache the store and return it alongside the path that lives inside the store.
        self.cache_and_resolve(to_resolve, store, depth)
    }
}

impl Registry {
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
}

/// Builds the [`ObjectStore`] that serves `to_resolve`, with the path within that store.
///
/// This mirrors [`parse_url_opts`], extended with schemes the `object_store` crate does not
/// recognize. Reporting the path the way `parse_url_opts` does is what keeps every scheme on
/// the one caching rule in [`Registry::resolve`]: a scheme says where its store is rooted, and
/// the registry decides how to cache it.
fn build_store(to_resolve: &Url) -> object_store::Result<(Arc<dyn ObjectStore>, Path)> {
    // OpenDAL-backed schemes (Tencent COS) are not recognized by `object_store`, so build them
    // from OpenDAL's own environment-variable configuration (e.g. `TENCENTCLOUD_SECRET_ID`).
    // The operator is rooted at the bucket, which lives in the URL authority, so — exactly as
    // for `s3://bucket/path` — the whole URL path is the object key.
    #[cfg(feature = "opendal")]
    if to_resolve.scheme() == vortex_object_store_opendal::COS_SCHEME {
        let store =
            vortex_object_store_opendal::make_opendal_store(to_resolve, &VortexHashMap::new())?;
        return Ok((store, Path::from_url_path(to_resolve.path())?));
    }

    let normalized_env = std::env::vars().map(|(k, v)| (k.to_ascii_lowercase(), v));
    let (store, path) = parse_url_opts(to_resolve, normalized_env)?;
    Ok((Arc::from(store), path))
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
mod tests {
    use std::fmt::Write;
    use std::sync::Arc;

    use object_store::ObjectStore;
    use object_store::path::Path;
    use object_store::registry::ObjectStoreRegistry;
    use url::Url;

    use crate::object_store::registry::Registry;

    fn with_var<F>(key: &str, value: &str, func: F)
    where
        F: FnOnce(),
    {
        let old_val = std::env::var(key).ok();

        // SAFETY: these unit tests run single-threaded.
        unsafe { std::env::set_var(key, value) };

        func();

        // Set the variable back to its original value
        match old_val {
            None => {
                unsafe { std::env::remove_var(key) };
            }
            Some(val) => {
                unsafe { std::env::set_var(key, val) };
            }
        }
    }

    #[test]
    fn test_resolve_percent_encoded_path() {
        let registry = Registry::default();
        let url =
            Url::parse("https://example.com/datasets/org/name/resolve/refs%2Fconvert%2Fparquet/dir/file.vortex")
                .unwrap();
        let expected = Path::from("datasets/org/name/resolve/refs/convert/parquet/dir/file.vortex");

        // First resolution parses and registers the store; it must decode the path, not panic.
        let (_store, path) = registry.resolve(&url).unwrap();
        assert_eq!(path, expected);

        // Second resolution takes the cached-store branch and must agree.
        let (_store, path) = registry.resolve(&url).unwrap();
        assert_eq!(path, expected);
    }

    #[test]
    #[expect(clippy::use_debug)]
    fn test_resolve_url() {
        with_var("AWS_REGION", "us-east-3", || {
            let registry = Registry::default();
            let (store, _) = registry
                .resolve(&Url::parse("s3://my-bucket/test").unwrap())
                .unwrap();

            // NOTE(aduffy): object_store doesn't let us downcast stores, the only way to verify
            //  that a configuration was added was to validate that it ends up in the Debug
            //  output :/
            let mut debug_str = String::new();
            write!(&mut debug_str, "{store:?}").unwrap();

            assert!(debug_str.contains("us-east-3"));
        });
    }

    /// Two resolves of the same URL must return the same `Arc` (the cached client) and the same
    /// path, and two different keys must get distinct stores. This pins the symmetry the registry
    /// promises: cache hit returns the same client and the same key.
    #[test]
    fn test_resolve_url_caches_per_key() {
        with_var("AWS_REGION", "us-east-3", || {
            let registry = Registry::default();
            let first = Url::parse("s3://my-bucket/first/second").unwrap();
            let second = Url::parse("s3://my-bucket/first/second").unwrap();

            let (store_a, path_a) = registry.resolve(&first).unwrap();
            let (store_b, path_b) = registry.resolve(&second).unwrap();
            assert!(Arc::ptr_eq(&store_a, &store_b));
            assert_eq!(path_a, path_b);

            // A different bucket gets its own client.
            let other = Url::parse("s3://other-bucket/first/second").unwrap();
            let (store_c, _) = registry.resolve(&other).unwrap();
            assert!(!Arc::ptr_eq(&store_a, &store_c));
        });
    }

    /// Two objects in the same bucket must share a cached client. This is the regression that
    /// the bespoke "walk every segment then return the whole key" path caused: by walking all
    /// segments it stored the store at full depth, so the next resolve saw the whole key
    /// already consumed and returned an empty path. Pinning the path here guards against that.
    #[test]
    fn test_resolve_url_shared_client_same_bucket() {
        with_var("AWS_REGION", "us-east-3", || {
            let registry = Registry::default();
            let a = Url::parse("s3://my-bucket/path/to/data.vortex").unwrap();
            let b = Url::parse("s3://my-bucket/other/data.vortex").unwrap();

            let (store_a, path_a) = registry.resolve(&a).unwrap();
            let (store_b, path_b) = registry.resolve(&b).unwrap();
            assert!(Arc::ptr_eq(&store_a, &store_b));
            assert_eq!(path_a.as_ref(), "path/to/data.vortex");
            assert_eq!(path_b.as_ref(), "other/data.vortex");
        });
    }

    /// Pinning the `entry_at` helper at depth > 0 (the path-prefix case) protects the shared
    /// `entry_at` / `cache_and_resolve` helpers from regressing prefix registration.
    #[test]
    fn test_register_at_prefix_shares_store() {
        let registry = Registry::default();
        let prefix = Url::parse("s3://my-bucket/prefix/").unwrap();
        let (inner, _) = registry.resolve(&prefix).unwrap();
        let cached: Arc<dyn ObjectStore> = Arc::clone(&inner);
        let replaced = registry.register(prefix.clone(), Arc::clone(&cached));
        // First registration at this prefix replaces nothing.
        assert!(replaced.is_none());

        // A resolve at a deeper key still walks through the registered prefix's store.
        let deeper = Url::parse("s3://my-bucket/prefix/inner/key").unwrap();
        let (store_deeper, path_deeper) = registry.resolve(&deeper).unwrap();
        assert!(Arc::ptr_eq(&store_deeper, &cached));
        assert_eq!(path_deeper.as_ref(), "inner/key");
    }
}
