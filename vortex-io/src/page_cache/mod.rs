// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A bounded memory cache of the byte pages of immutable objects, shared by every reader that
//! wraps the same object.
//!
//! A [`PageCache`] cuts each object into fixed-size pages. A read through a cache-wrapped reader
//! fetches only the pages its range touches, serves pages that are already kept, and shares a
//! page that another read is still fetching. After each read, the cache can fetch the pages that
//! follow it before they are asked for.

mod read_ahead;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::future::WeakShared;
use futures::stream;
use parking_lot::Mutex;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use self::read_ahead::ReadAhead;
use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;

/// The memory and IO limits of a [`PageCache`].
#[derive(Clone, Copy, Debug)]
pub struct PageCacheConfig {
    /// The most page bytes the cache keeps after their reads complete. Zero disables the cache.
    pub capacity_bytes: usize,
    /// The size of a page. A read fetches only the pages that its range touches.
    pub page_bytes: usize,
    /// The most physical reads in flight at the same time, over every object, including the
    /// reads that fetch pages ahead.
    pub max_in_flight: usize,
    /// The most pages after a read that the cache fetches before they are asked for. Zero
    /// disables read-ahead.
    pub read_ahead_pages: usize,
    /// The most pages fetched ahead at the same time, over every object.
    pub max_prefetch_in_flight: usize,
}

impl Default for PageCacheConfig {
    fn default() -> Self {
        Self {
            capacity_bytes: 128 * 1024 * 1024,
            page_bytes: 256 * 1024,
            max_in_flight: 16,
            read_ahead_pages: 4,
            max_prefetch_in_flight: 2,
        }
    }
}

/// The activity of a [`PageCache`]. Every clone of a cache shares one set of counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct PageCacheStats {
    /// The bytes the cache keeps now. It does not count buffers that callers hold or reads in
    /// flight.
    pub cached_bytes: usize,
    /// Page requests that a kept page served without a physical read.
    pub hits: u64,
    /// Page loads that started a physical read, including loads that were cancelled later.
    pub misses: u64,
    /// Page requests that shared a load already in flight.
    pub coalesced: u64,
    /// Physical reads that got a permit from the IO budget.
    pub physical_reads: u64,
    /// The bytes those physical reads asked for, including the reads that fetch pages ahead.
    pub physical_bytes: u64,
}

/// A memory cache of the byte pages of immutable objects, shared by many readers, that keeps at
/// most [`PageCacheConfig::capacity_bytes`] and evicts the least recently used page first.
///
/// The caller names each object with a key when it [wraps](Self::wrap) a reader. Readers with the
/// same key share pages, so the key must identify the object's bytes exactly, for example by its
/// location and an immutable version or digest. The cache does not check access: a caller that
/// must keep the pages of different principals apart puts the principal in the key.
#[derive(Clone)]
pub struct PageCache {
    pages: Arc<CachePages>,
    read_ahead: Arc<ReadAhead>,
    handle: Option<Handle>,
}

struct CachePages {
    config: PageCacheConfig,
    state: Mutex<CacheState>,
    reads: Arc<Semaphore>,
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct PageKey {
    object: Arc<str>,
    page: u64,
}

struct CachedPage {
    bytes: ByteBuffer,
    accessed: u64,
}

type PageResult = Result<ByteBuffer, Arc<VortexError>>;
type PageFuture = BoxFuture<'static, PageResult>;

#[derive(Default)]
struct CacheState {
    pages: HashMap<PageKey, CachedPage>,
    inflight: HashMap<PageKey, WeakShared<PageFuture>>,
    clock: u64,
    stats: PageCacheStats,
}

impl PageCache {
    /// A cache with the memory and IO limits of `config`.
    ///
    /// The IO limits bound the page buffers in flight by `page_bytes * max_in_flight`. Buffers
    /// already returned to callers do not count against them. The cache starts empty and keeps
    /// pages in memory only.
    pub fn new(config: PageCacheConfig) -> VortexResult<Self> {
        if config.page_bytes == 0 || config.max_in_flight == 0 {
            vortex_bail!("Page cache page size and IO concurrency must be positive");
        }

        if config.max_in_flight > Semaphore::MAX_PERMITS
            || config.max_prefetch_in_flight > Semaphore::MAX_PERMITS
            || config
                .page_bytes
                .checked_mul(config.max_in_flight)
                .is_none()
            || config
                .page_bytes
                .checked_mul(config.max_prefetch_in_flight)
                .is_none()
        {
            vortex_bail!("Page cache IO limits exceed platform limits");
        }

        if config.capacity_bytes != 0 && config.capacity_bytes < config.page_bytes {
            vortex_bail!("Page cache capacity must hold at least one page");
        }

        Ok(Self {
            pages: Arc::new(CachePages {
                config,
                state: Mutex::new(CacheState::default()),
                reads: Arc::new(Semaphore::new(config.max_in_flight)),
            }),
            read_ahead: Arc::new(ReadAhead::new(
                config.read_ahead_pages,
                config.max_prefetch_in_flight,
            )),
            handle: None,
        })
    }

    /// Fetch pages ahead on the tasks of `handle`. Without a handle, the cache fetches ahead on
    /// the runtime that [`Handle::find`] returns, and not at all when it returns none.
    pub fn with_handle(mut self, handle: Handle) -> Self {
        self.handle = Some(handle);
        self
    }

    /// The bytes the cache keeps now and its counts of reads since it was made.
    pub fn stats(&self) -> PageCacheStats {
        self.pages.state.lock().stats
    }

    /// Whether the cache keeps pages. A cache with no capacity returns readers unchanged from
    /// [`wrap`](Self::wrap).
    pub fn is_enabled(&self) -> bool {
        self.pages.config.capacity_bytes != 0
    }

    /// A reader of the `size` bytes of `source` through the cache, where `object` names exactly
    /// those bytes.
    pub fn wrap(
        &self,
        object: Arc<str>,
        size: u64,
        source: Arc<dyn VortexReadAt>,
    ) -> Arc<dyn VortexReadAt> {
        if !self.is_enabled() {
            return source;
        }

        Arc::new(CachedReadAt {
            cache: self.clone(),
            object,
            size,
            source,
        })
    }
}

impl CachePages {
    fn page(
        self: &Arc<Self>,
        object: Arc<str>,
        page: u64,
        size: u64,
        source: Arc<dyn VortexReadAt>,
        speculative: bool,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let cache = Arc::clone(self);
        let key = PageKey { object, page };
        async move {
            let cached = cache.state.lock().lookup(&key);
            if let Some(cached) = cached {
                return cached.await;
            }

            // A page fetched ahead never waits for a permit, so demand reads keep priority.
            let permit = if speculative {
                Arc::clone(&cache.reads)
                    .try_acquire_owned()
                    .map_err(|error| vortex_err!("Page cache read-ahead has no permit: {error}"))?
            } else {
                Arc::clone(&cache.reads)
                    .acquire_owned()
                    .await
                    .map_err(|error| vortex_err!("Page cache IO budget is closed: {error}"))?
            };
            cache.start_page(key, size, source, permit).await
        }
        .boxed()
    }

    fn start_page(
        self: &Arc<Self>,
        key: PageKey,
        size: u64,
        source: Arc<dyn VortexReadAt>,
        permit: OwnedSemaphorePermit,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let mut state = self.state.lock();
        if let Some(cached) = state.lookup(&key) {
            return cached;
        }

        state
            .inflight
            .retain(|_, future| future.upgrade().is_some());
        state.stats.misses += 1;

        let cache = Arc::clone(self);
        let load_key = key.clone();
        let load = async move {
            let _permit = permit;
            let result = cache.load_page(&load_key, size, source).await;

            let mut state = cache.state.lock();
            state.inflight.remove(&load_key);
            if let Ok(bytes) = &result {
                state.store(load_key, bytes.clone(), cache.config.capacity_bytes);
            }
            result.map_err(Arc::new)
        }
        .boxed()
        .shared();

        state.inflight.insert(
            key,
            load.downgrade()
                .vortex_expect("a new shared load has not completed"),
        );
        async move { load.await.map_err(VortexError::from) }.boxed()
    }

    async fn load_page(
        &self,
        key: &PageKey,
        size: u64,
        source: Arc<dyn VortexReadAt>,
    ) -> VortexResult<ByteBuffer> {
        let offset = key.page * self.config.page_bytes as u64;
        let length = usize::try_from((size - offset).min(self.config.page_bytes as u64))
            .map_err(|error| vortex_err!("Page length is not a usize: {error}"))?;

        {
            let mut state = self.state.lock();
            state.stats.physical_reads += 1;
            state.stats.physical_bytes += length as u64;
        }

        Ok(source
            .read_at(offset, length, Alignment::none())
            .await?
            .to_host()
            .await)
    }
}

impl CacheState {
    fn lookup(&mut self, key: &PageKey) -> Option<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        self.clock = self.clock.wrapping_add(1);
        if let Some(cached) = self.pages.get_mut(key) {
            cached.accessed = self.clock;
            let bytes = cached.bytes.clone();
            self.stats.hits += 1;
            return Some(async move { Ok(bytes) }.boxed());
        }

        if let Some(pending) = self.inflight.get(key).and_then(WeakShared::upgrade) {
            self.stats.coalesced += 1;
            return Some(async move { pending.await.map_err(VortexError::from) }.boxed());
        }

        None
    }

    fn store(&mut self, key: PageKey, bytes: ByteBuffer, capacity: usize) {
        if bytes.len() > capacity {
            return;
        }

        while self.stats.cached_bytes + bytes.len() > capacity {
            let Some(oldest) = self
                .pages
                .iter()
                .min_by_key(|(_, page)| page.accessed)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(old) = self.pages.remove(&oldest) {
                self.stats.cached_bytes -= old.bytes.len();
            }
        }

        self.clock = self.clock.wrapping_add(1);
        self.stats.cached_bytes += bytes.len();
        self.pages.insert(
            key,
            CachedPage {
                bytes,
                accessed: self.clock,
            },
        );
    }
}

/// A reader of one object through a [`PageCache`].
struct CachedReadAt {
    cache: PageCache,
    object: Arc<str>,
    size: u64,
    source: Arc<dyn VortexReadAt>,
}

impl VortexReadAt for CachedReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        self.source.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.source.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.cache.pages.config.max_in_flight
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.size;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let cache = self.cache.clone();
        let object = Arc::clone(&self.object);
        let source = Arc::clone(&self.source);
        let size = self.size;
        async move {
            let end = offset
                .checked_add(length as u64)
                .filter(|end| *end <= size)
                .ok_or_else(|| {
                    vortex_err!(
                        "Read of {length} bytes at {offset} exceeds the object's {size} bytes"
                    )
                })?;
            if length == 0 {
                return Ok(BufferHandle::new_host(
                    ByteBuffer::from(Vec::<u8>::new()).aligned(alignment),
                ));
            }

            let page_bytes = cache.pages.config.page_bytes as u64;
            let first = offset / page_bytes;
            let last = (end - 1) / page_bytes;
            let pages = stream::iter(first..=last)
                .map(|page| {
                    cache
                        .pages
                        .page(Arc::clone(&object), page, size, Arc::clone(&source), false)
                })
                .buffered(cache.pages.config.max_in_flight)
                .try_collect::<Vec<_>>()
                .await?;

            let start = usize::try_from(offset % page_bytes)?;
            let result = if pages.len() == 1 {
                pages[0]
                    .slice_unaligned(start..start + length)
                    .aligned(alignment)
            } else {
                let mut bytes = Vec::with_capacity(length);
                for (index, page) in pages.iter().enumerate() {
                    let start = if index == 0 { start } else { 0 };
                    let take = (length - bytes.len()).min(page.len() - start);
                    bytes.extend_from_slice(&page[start..start + take]);
                }
                ByteBuffer::from(bytes).aligned(alignment)
            };

            if let Some(handle) = cache.handle.clone().or_else(Handle::find) {
                let pages = Arc::clone(&cache.pages);
                let prefetch_object = Arc::clone(&object);
                cache.read_ahead.observe(
                    &handle,
                    object,
                    first,
                    last,
                    size.div_ceil(page_bytes),
                    move |page| {
                        let pages = Arc::clone(&pages);
                        let object = Arc::clone(&prefetch_object);
                        let source = Arc::clone(&source);
                        async move {
                            // A page fetched ahead that fails is fetched again on demand.
                            drop(pages.page(object, page, size, source, true).await);
                        }
                        .boxed()
                    },
                );
            }

            Ok(BufferHandle::new_host(result))
        }
        .boxed()
    }
}
