// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`FooterCache`] adapter backed by DataFusion's [`FileMetadataCache`].

use std::sync::Arc;

use chrono::DateTime;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use object_store::ObjectMeta;
use object_store::path::Path;
use vortex::file::Footer;
use vortex::file::FooterCache;

use crate::persistent::cache::CachedVortexMetadata;

/// A [`FooterCache`] backed by DataFusion's [`FileMetadataCache`].
///
/// Participates in DataFusion's cache memory accounting. File paths are mapped to
/// [`ObjectMeta`] keys with synthetic metadata (zero size, epoch timestamp) so entries
/// created here are distinct from entries created by DataFusion's persistent file scanning.
pub struct DataFusionFooterCache {
    cache: Arc<dyn FileMetadataCache>,
}

impl DataFusionFooterCache {
    /// Create a new adapter wrapping a DataFusion [`FileMetadataCache`].
    pub fn new(cache: Arc<dyn FileMetadataCache>) -> Self {
        Self { cache }
    }
}

impl FooterCache for DataFusionFooterCache {
    fn get(&self, key: &str) -> Option<Footer> {
        let meta = object_meta_for_key(key);
        let cached = self.cache.get(&meta)?;
        let vortex_meta = cached.as_any().downcast_ref::<CachedVortexMetadata>()?;
        Some(vortex_meta.footer().clone())
    }

    fn put(&self, key: &str, footer: Footer) {
        let meta = object_meta_for_key(key);
        self.cache
            .put(&meta, Arc::new(CachedVortexMetadata::from_footer(footer)));
    }
}

fn object_meta_for_key(key: &str) -> ObjectMeta {
    ObjectMeta {
        location: Path::from(key),
        last_modified: DateTime::default(),
        size: 0,
        e_tag: None,
        version: None,
    }
}
