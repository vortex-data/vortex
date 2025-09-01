// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_file::Footer;

use crate::duckdb::ObjectCacheRef;

#[derive(Clone)]
pub struct FooterCache<'a> {
    object_cache: ObjectCacheRef<'a>,
}

impl<'a> FooterCache<'a> {
    pub fn new(object_cache: ObjectCacheRef<'a>) -> Self {
        Self { object_cache }
    }

    pub fn get(&self, key: &str) -> Option<&Footer> {
        self.object_cache.get(&Self::key(key))
    }

    pub fn insert(&self, key: &str, footer: Footer) {
        self.object_cache.put(&Self::key(key), footer);
    }

    fn key(key: &str) -> String {
        format!("vx_cache_key://{key}")
    }
}
