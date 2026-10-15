// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session extension for multi-file scanning, providing shared footer and segment caches.

use std::any::Any;
use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use crate::footer::Footer;

/// Session state for multi-file scanning.
///
/// Provides a shared, in-memory footer cache so that repeated scans over the same files
/// avoid redundant footer I/O. The cache is bounded by entry count and lives as long as
/// the [`VortexSession`](vortex_session::VortexSession).
///
/// An optional segment cache can be enabled with [`Self::enable_segment_cache`]. It is shared by
/// every file opened through the session and keyed by file path, so files must not be rewritten
/// in place while it is enabled.
///
/// # Future Work
///
/// Consider generalizing this cache into [`VortexOpenOptions`](crate::VortexOpenOptions) so
/// that single-file opens also benefit from session-level footer caching.
#[derive(Clone)]
pub struct MultiFileSession {
    footer_cache: moka::sync::Cache<String, Footer>,
    segment_cache: Option<SharedSegmentCache>,
}

/// Segment buffers shared by every file opened through a session, keyed by file path.
///
/// This is a plain concurrent map rather than an evicting cache: lookups sit on the scan's hot
/// path, so they must stay cheap. Once the byte budget is spent, further segments are not cached.
/// The budget counts segment bytes, while a cached segment may keep a larger coalesced read alive.
#[derive(Clone)]
struct SharedSegmentCache(Arc<SharedSegmentCacheInner>);

struct SharedSegmentCacheInner {
    entries: DashMap<(Arc<str>, SegmentId), ByteBuffer>,
    used_bytes: AtomicU64,
    max_bytes: u64,
}

impl SharedSegmentCache {
    fn new(max_bytes: u64) -> Self {
        Self(Arc::new(SharedSegmentCacheInner {
            entries: DashMap::default(),
            used_bytes: AtomicU64::new(0),
            max_bytes,
        }))
    }

    fn get(&self, key: &(Arc<str>, SegmentId)) -> Option<ByteBuffer> {
        self.0.entries.get(key).map(|buffer| buffer.clone())
    }

    fn insert(&self, key: (Arc<str>, SegmentId), buffer: ByteBuffer) {
        let inner = &self.0;
        let len = buffer.len() as u64;
        if inner.used_bytes.fetch_add(len, Ordering::Relaxed) + len > inner.max_bytes {
            inner.used_bytes.fetch_sub(len, Ordering::Relaxed);
            return;
        }
        match inner.entries.entry(key) {
            Entry::Occupied(_) => {
                inner.used_bytes.fetch_sub(len, Ordering::Relaxed);
            }
            Entry::Vacant(entry) => {
                entry.insert(buffer);
            }
        }
    }

    /// Remove every entry. Inserts racing with the clear may survive it.
    fn clear(&self) {
        self.0.entries.clear();
        self.0.used_bytes.store(0, Ordering::Relaxed);
    }

    fn len(&self) -> usize {
        self.0.entries.len()
    }
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
            segment_cache: None,
        }
    }
}

impl Debug for MultiFileSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiFileSession")
            .field("footer_cache_entry_count", &self.footer_cache.entry_count())
            .field(
                "segment_cache_entry_count",
                &self.segment_cache.as_ref().map(SharedSegmentCache::len),
            )
            .finish()
    }
}

impl MultiFileSession {
    /// Retrieve a cached footer for the given file path.
    pub fn get_footer(&self, path: &str) -> Option<Footer> {
        self.footer_cache.get(path)
    }

    /// Store a footer under the given file path.
    pub(crate) fn put_footer(&self, path: &str, footer: Footer) {
        self.footer_cache.insert(path.to_string(), footer);
    }

    /// Enable a segment cache shared by all files opened through this session, capped at
    /// `max_capacity_bytes` of segment data.
    ///
    /// Replaces any previously enabled segment cache.
    pub fn enable_segment_cache(&mut self, max_capacity_bytes: u64) {
        self.segment_cache = Some(SharedSegmentCache::new(max_capacity_bytes));
    }

    /// Returns a [`SegmentCache`] scoped to the file at `path`, if a segment cache is enabled.
    pub fn segment_cache(&self, path: &str) -> Option<Arc<dyn SegmentCache>> {
        self.segment_cache.as_ref().map(|cache| {
            Arc::new(FileSegmentCache {
                path: Arc::from(path),
                cache: cache.clone(),
            }) as Arc<dyn SegmentCache>
        })
    }

    /// Remove every entry from the segment cache, if one is enabled.
    pub fn clear_segment_cache(&self) {
        if let Some(cache) = &self.segment_cache {
            cache.clear();
        }
    }
}

/// A view of the session-wide segment cache for a single file.
struct FileSegmentCache {
    path: Arc<str>,
    cache: SharedSegmentCache,
}

#[async_trait]
impl SegmentCache for FileSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.cache.get(&(Arc::clone(&self.path), id)))
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.cache.insert((Arc::clone(&self.path), id), buffer);
        Ok(())
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

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_layout::segments::SegmentId;

    use super::MultiFileSession;

    #[tokio::test]
    async fn segment_cache_is_scoped_per_file() -> VortexResult<()> {
        let mut session = MultiFileSession::default();
        assert!(session.segment_cache("a.vortex").is_none());

        session.enable_segment_cache(1 << 20);
        let a = session.segment_cache("a.vortex").expect("cache enabled");
        let b = session.segment_cache("b.vortex").expect("cache enabled");
        let id = SegmentId::from(0u32);

        a.put(id, ByteBuffer::copy_from(b"a")).await?;
        assert_eq!(a.get(id).await?.as_deref(), Some(b"a".as_slice()));
        assert!(b.get(id).await?.is_none());

        session.clear_segment_cache();
        assert!(a.get(id).await?.is_none());
        Ok(())
    }
}
