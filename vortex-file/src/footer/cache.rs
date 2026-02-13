// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_utils::aliases::hash_map::HashMap;

use super::Footer;

/// A shared reference to a [`FooterCache`].
pub type FooterCacheRef = Arc<dyn FooterCache>;

/// A cache for [`Footer`]s keyed by file path.
///
/// Implementations can wrap engine-specific caches (e.g. DuckDB's object cache) or use the
/// provided [`InMemoryFooterCache`] for simple in-process caching.
pub trait FooterCache: Send + Sync {
    /// Retrieve a cached footer for the given key, or `None` if not cached.
    fn get(&self, key: &str) -> Option<Footer>;

    /// Store a footer under the given key.
    fn put(&self, key: &str, footer: Footer);
}

/// A simple in-memory [`FooterCache`] backed by a [`HashMap`].
#[derive(Default)]
pub struct InMemoryFooterCache {
    cache: RwLock<HashMap<String, Footer>>,
}

impl InMemoryFooterCache {
    /// Create an empty in-memory footer cache.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FooterCache for InMemoryFooterCache {
    fn get(&self, key: &str) -> Option<Footer> {
        self.cache.read().get(key).cloned()
    }

    fn put(&self, key: &str, footer: Footer) {
        self.cache.write().insert(key.to_string(), footer);
    }
}
