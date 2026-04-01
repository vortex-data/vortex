// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Apache Software Foundation (ASF)

//! This file is an adapted version of the `DefaultObjectStoreRegistry` from the object_store crate,
//! but modified to resolve configurations out of environment variables case-insensitively. This
//! is similar to how all the `Store::from_env` builders work for the various object stores.
//!
//! See also <https://github.com/apache/arrow-rs-object-store/issues/529>

#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::sync::Arc;

use object_store::ObjectStore;
use object_store::parse_url_opts;
use object_store::path::Path;
use object_store::path::PathPart;
use object_store::registry::ObjectStoreRegistry;
use parking_lot::RwLock;
use url::Url;

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
        let key = url_key(&url);
        let mut entry = map.entry(key.to_string()).or_default();

        for segment in path_segments(url.path()) {
            entry = entry.children.entry(segment.to_string()).or_default();
        }
        entry.store.replace(store)
    }

    fn resolve(&self, to_resolve: &Url) -> object_store::Result<(Arc<dyn ObjectStore>, Path)> {
        let key = url_key(to_resolve);
        {
            let map = self.map.read();

            if let Some((store, depth)) = map.get(key).and_then(|entry| entry.lookup(to_resolve)) {
                let path = path_suffix(to_resolve, depth)?;
                return Ok((Arc::clone(store), path));
            }
        }

        let normalized_env = std::env::vars().map(|(k, v)| (k.to_ascii_lowercase(), v));

        if let Ok((store, path)) = parse_url_opts(to_resolve, normalized_env) {
            let depth = num_segments(to_resolve.path()) - num_segments(path.as_ref());

            let mut map = self.map.write();
            let mut entry = map.entry(key.to_string()).or_default();
            for segment in path_segments(to_resolve.path()).take(depth) {
                entry = entry.children.entry(segment.to_string()).or_default();
            }
            let store = Arc::clone(match &entry.store {
                None => entry.store.insert(Arc::from(store)),
                Some(x) => x, // Racing creation - use existing
            });

            let path = path_suffix(to_resolve, depth)?;
            return Ok((store, path));
        }

        Err(object_store::Error::Generic {
            store: "ObjectStoreRegistry",
            source: "URL could not be resolved".into(),
        })
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

#[cfg(test)]
mod tests {
    use std::fmt::Write;

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
}
