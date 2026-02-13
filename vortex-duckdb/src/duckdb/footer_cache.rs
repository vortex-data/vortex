// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::file::Footer;
use vortex::file::FooterCache;

use crate::duckdb::ObjectCacheRef;

/// A [`FooterCache`] backed by DuckDB's internal object cache.
pub struct DuckDbFooterCache {
    object_cache: ObjectCacheRef<'static>,
}

impl DuckDbFooterCache {
    pub fn new(object_cache: ObjectCacheRef<'static>) -> Self {
        Self { object_cache }
    }

    fn key(path: &str) -> String {
        format!("vx_footer://{path}")
    }
}

impl FooterCache for DuckDbFooterCache {
    fn get(&self, key: &str) -> Option<Footer> {
        self.object_cache.get::<Footer>(&Self::key(key)).cloned()
    }

    fn put(&self, key: &str, footer: Footer) {
        self.object_cache.put(&Self::key(key), footer);
    }
}

// SAFETY: ObjectCacheRef<'static> is Send + Sync (DuckDB object cache is mutex-protected).
unsafe impl Send for DuckDbFooterCache {}
unsafe impl Sync for DuckDbFooterCache {}
