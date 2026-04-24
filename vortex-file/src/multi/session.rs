// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session extension for multi-file scanning, providing a shared footer cache.

use std::fmt;
use std::fmt::Debug;

use vortex_session::SessionExt;

use crate::footer::Footer;

/// Session state for multi-file scanning.
///
/// Provides a shared, in-memory footer cache so that repeated scans over the same files
/// avoid redundant footer I/O. The cache is bounded by entry count and lives as long as
/// the [`VortexSession`](vortex_session::VortexSession).
///
/// # Future Work
///
/// Consider generalizing this cache into [`VortexOpenOptions`](crate::VortexOpenOptions) so
/// that single-file opens also benefit from session-level footer caching.
pub(super) struct MultiFileSession {
    footer_cache: moka::sync::Cache<String, Footer>,
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
    /// Retrieve a cached footer for the given file path.
    pub fn get_footer(&self, path: &str) -> Option<Footer> {
        self.footer_cache.get(path)
    }

    /// Store a footer under the given file path.
    pub fn put_footer(&self, path: &str, footer: Footer) {
        self.footer_cache.insert(path.to_string(), footer);
    }
}

/// Extension trait for accessing the [`MultiFileSession`] from a session.
pub(super) trait MultiFileSessionExt: SessionExt {
    /// Returns a reference to the [`MultiFileSession`] state.
    fn multi_file(&self) -> vortex_session::Ref<'_, MultiFileSession> {
        self.get::<MultiFileSession>()
    }
}

impl<S: SessionExt> MultiFileSessionExt for S {}
