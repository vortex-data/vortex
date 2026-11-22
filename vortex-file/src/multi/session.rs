// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session extension for multi-file scanning, providing a shared footer cache.

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;

use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::footer::Footer;

/// Cache key identifying a specific immutable version of a file within a filesystem namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FooterCacheKey {
    /// Filesystem namespace isolating identical relative paths across different filesystems.
    pub namespace: Arc<str>,
    /// File path or URI within the namespace.
    pub path: Arc<str>,
    /// Version token bound to the opened object's bytes.
    pub version: Arc<str>,
}

impl FooterCacheKey {
    /// Create a new [`FooterCacheKey`].
    pub fn new(
        namespace: impl Into<Arc<str>>,
        path: impl Into<Arc<str>>,
        version: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            path: path.into(),
            version: version.into(),
        }
    }
}

/// Session state for multi-file scanning.
///
/// Provides a shared, in-memory footer cache so that repeated scans over the same immutable
/// file version avoid redundant footer I/O. The cache is bounded by approximate memory weight
/// and lives as long as the [`VortexSession`](vortex_session::VortexSession).
///
/// # Future Work
///
/// Consider generalizing this cache into [`VortexOpenOptions`](crate::VortexOpenOptions) so
/// that single-file opens also benefit from session-level footer caching.
#[derive(Clone)]
pub struct MultiFileSession {
    footer_cache: moka::sync::Cache<FooterCacheKey, Footer>,
}

impl Default for MultiFileSession {
    fn default() -> Self {
        Self {
            footer_cache: moka::sync::Cache::builder()
                // Capacity and weigher are in KB
                .max_capacity(100 * 1024) // 100MB
                .weigher(|_k, footer: &Footer| {
                    footer
                        .approx_byte_size()
                        .and_then(|bytes| u32::try_from(bytes / 1024).ok())
                        .unwrap_or(10)
                })
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
    /// Retrieve a cached footer for the given versioned file identity.
    pub fn get_footer(&self, key: &FooterCacheKey) -> Option<Footer> {
        self.footer_cache.get(key)
    }

    /// Store a footer under the given versioned file identity.
    pub(crate) fn put_footer(&self, key: FooterCacheKey, footer: Footer) {
        self.footer_cache.insert(key, footer);
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
