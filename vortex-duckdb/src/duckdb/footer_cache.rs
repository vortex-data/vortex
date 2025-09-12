// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::file::{FileType, Footer, VortexOpenOptions};

use crate::duckdb::ObjectCacheRef;

pub struct FooterCache<'a> {
    object_cache: ObjectCacheRef<'a>,
}

pub struct Entry<'a> {
    object_cache: ObjectCacheRef<'a>,
    key: String,
    value: Option<&'a Footer>,
}

impl Entry<'_> {
    pub fn put_if_absent(self, value: impl FnOnce() -> Footer) {
        if self.value.is_some() {
            return;
        }
        self.object_cache.put(&self.key, value());
    }

    pub fn apply_to_file<F: FileType>(&self, file: VortexOpenOptions<F>) -> VortexOpenOptions<F> {
        if let Some(footer) = self.value {
            file.with_footer(footer.clone())
        } else {
            file
        }
    }
}

impl<'a> FooterCache<'a> {
    pub fn new(object_cache: ObjectCacheRef<'a>) -> Self {
        Self { object_cache }
    }

    pub fn entry(&self, key: &str) -> Entry<'_> {
        let key = Self::key(key);
        let value = self.object_cache.get(&key);
        Entry {
            object_cache: self.object_cache,
            key,
            value,
        }
    }

    fn key(key: &str) -> String {
        format!("vx_cache_key://{key}")
    }
}
