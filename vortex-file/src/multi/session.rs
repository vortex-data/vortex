// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session extension for multi-file scanning, providing a shared footer cache.

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;
use vortex_utils::aliases::hash_map::HashMap;

use vortex_buffer::ByteBuffer;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::footer::Footer;

/// Session state for multi-file scanning.
///
/// Provides a shared, in-memory footer cache so that repeated scans over the same files
/// avoid redundant footer I/O. A file opened with its user-defined metadata also keeps that
/// metadata in the cache, so later opens that include metadata do not read it again. The cache is bounded by entry count and lives as long as
/// the [`VortexSession`](vortex_session::VortexSession).
///
/// # Future Work
///
/// Consider generalizing this cache into [`VortexOpenOptions`](crate::VortexOpenOptions) so
/// that single-file opens also benefit from session-level footer caching.
#[derive(Clone)]
pub struct MultiFileSession {
    footer_cache: moka::sync::Cache<String, CachedFile>,
}

/// The cached parts of an opened file.
#[derive(Clone)]
pub(crate) struct CachedFile {
    pub(crate) footer: Footer,
    /// The user-defined metadata segments, when an open has loaded them.
    pub(crate) metadata: Option<Arc<HashMap<String, ByteBuffer>>>,
}

impl CachedFile {
    /// The approximate size of the entry in KiB, as the cache weigher expects.
    fn weight(&self) -> u32 {
        let footer_bytes = self.footer.approx_byte_size().unwrap_or(10 * 1024);
        let metadata_bytes: usize = self
            .metadata
            .iter()
            .flat_map(|metadata| metadata.iter())
            .map(|(key, value)| key.len() + value.len())
            .sum();

        u32::try_from((footer_bytes + metadata_bytes) / 1024).unwrap_or(u32::MAX)
    }
}

impl Default for MultiFileSession {
    fn default() -> Self {
        Self {
            footer_cache: moka::sync::Cache::builder()
                // Capacity and weigher are in KB
                .max_capacity(100 * 1024) // 100MB
                .weigher(|_k, file: &CachedFile| file.weight())
                .build(),
        }
    }
}

impl Debug for MultiFileSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiFileSession")
            .field("footer_cache_entry_count", &self.footer_cache.entry_count())
            .finish()
    }
}

impl MultiFileSession {
    /// Retrieve a cached footer for the given file path.
    pub fn get_footer(&self, path: &str) -> Option<Footer> {
        self.footer_cache.get(path).map(|file| file.footer)
    }

    /// Retrieve the cached footer and metadata for the given file path.
    pub(crate) fn get_file(&self, path: &str) -> Option<CachedFile> {
        self.footer_cache.get(path)
    }

    /// Store a footer, and the metadata when it was loaded, under the given file path.
    pub(crate) fn put_file(&self, path: &str, file: CachedFile) {
        self.footer_cache.insert(path.to_string(), file);
    }
}

impl SessionVar for MultiFileSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing the [`MultiFileSession`] from a session.
pub(super) trait MultiFileSessionExt: SessionExt {
    /// Returns a reference to the [`MultiFileSession`] state.
    fn multi_file(&self) -> SessionGuard<'_, MultiFileSession> {
        self.get::<MultiFileSession>()
    }
}

impl<S: SessionExt> MultiFileSessionExt for S {}
