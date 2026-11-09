// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use tokio::sync::Semaphore;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::PageCache;
use super::PageCacheConfig;
use crate::VortexReadAt;

/// A 64-byte object whose byte `i` is `i`, that counts its reads and can fail one or wait on a
/// gate.
#[derive(Clone)]
struct TestReadAt {
    bytes: ByteBuffer,
    calls: Arc<AtomicUsize>,
    fail: Arc<AtomicBool>,
    gate: Option<Arc<Semaphore>>,
}

impl TestReadAt {
    fn new(gate: Option<Arc<Semaphore>>) -> Arc<Self> {
        Arc::new(Self {
            bytes: ByteBuffer::from((0..64u8).collect::<Vec<_>>()),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: Arc::new(AtomicBool::new(false)),
            gate,
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl VortexReadAt for TestReadAt {
    fn concurrency(&self) -> usize {
        8
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.bytes.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let source = self.clone();
        async move {
            source.calls.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;

            if let Some(gate) = &source.gate {
                gate.acquire()
                    .await
                    .map_err(|error| vortex_err!("{error}"))?
                    .forget();
            }

            if source.fail.swap(false, Ordering::SeqCst) {
                vortex_bail!("injected read error");
            }
            source.bytes.read_at(offset, length, alignment).await
        }
        .boxed()
    }
}

fn cache(capacity_bytes: usize) -> VortexResult<PageCache> {
    PageCache::new(PageCacheConfig {
        capacity_bytes,
        page_bytes: 8,
        max_in_flight: 2,
        read_ahead_pages: 0,
        max_prefetch_in_flight: 0,
    })
}

fn wrap(cache: &PageCache, object: &str, source: &Arc<TestReadAt>) -> Arc<dyn VortexReadAt> {
    cache.wrap(Arc::from(object), 64, Arc::clone(source) as _)
}

async fn read(reader: &Arc<dyn VortexReadAt>, offset: u64, length: usize) -> VortexResult<Vec<u8>> {
    Ok(reader
        .read_at(offset, length, Alignment::none())
        .await?
        .to_host()
        .await
        .to_vec())
}

#[test]
fn rejects_invalid_limits() {
    let invalid = [
        PageCacheConfig {
            page_bytes: 0,
            ..PageCacheConfig::default()
        },
        PageCacheConfig {
            max_in_flight: 0,
            ..PageCacheConfig::default()
        },
        PageCacheConfig {
            capacity_bytes: 4,
            page_bytes: 8,
            ..PageCacheConfig::default()
        },
    ];
    for config in invalid {
        assert!(PageCache::new(config).is_err(), "{config:?}");
    }
}

#[tokio::test]
async fn disabled_cache_returns_the_source() -> VortexResult<()> {
    let cache = cache(0)?;
    assert!(!cache.is_enabled());

    let source = TestReadAt::new(None);
    let reader = wrap(&cache, "object", &source);
    read(&reader, 0, 8).await?;
    read(&reader, 0, 8).await?;
    assert_eq!(source.calls(), 2);
    Ok(())
}

#[tokio::test]
async fn readers_of_one_key_share_pages_and_reads_in_flight() -> VortexResult<()> {
    let cache = cache(32)?;
    let source = TestReadAt::new(None);
    let a = wrap(&cache, "scope-a:digest", &source);
    let reopened = wrap(&cache, "scope-a:digest", &source);

    let (first, second) = tokio::join!(read(&a, 2, 3), read(&reopened, 4, 2));
    assert_eq!(first?, [2, 3, 4]);
    assert_eq!(second?, [4, 5]);
    assert_eq!(source.calls(), 1);
    assert_eq!(cache.stats().coalesced, 1);

    read(&reopened, 0, 8).await?;
    assert_eq!(source.calls(), 1);
    assert_eq!(cache.stats().hits, 1);

    // Other keys never share pages, even over the same bytes.
    read(&wrap(&cache, "scope-b:digest", &source), 0, 8).await?;
    read(&wrap(&cache, "scope-a:new-digest", &source), 0, 8).await?;
    assert_eq!(source.calls(), 3);
    Ok(())
}

#[tokio::test]
async fn bounded_pages_keep_ranges_across_pages_and_retry_errors() -> VortexResult<()> {
    let cache = cache(16)?;
    let source = TestReadAt::new(None);
    let reader = wrap(&cache, "object", &source);

    assert_eq!(read(&reader, 3, 29).await?, (3..32u8).collect::<Vec<_>>());
    assert!(cache.stats().cached_bytes <= 16);

    assert!(read(&reader, u64::MAX, 4).await.is_err());
    assert!(read(&reader, 63, 2).await.is_err());

    source.fail.store(true, Ordering::SeqCst);
    assert!(read(&reader, 40, 3).await.is_err());
    assert_eq!(read(&reader, 40, 3).await?, [40, 41, 42]);
    assert!(cache.stats().cached_bytes <= 16);
    Ok(())
}

#[tokio::test]
async fn a_skipped_read_ahead_never_fails_a_demand_read() -> VortexResult<()> {
    let cache = cache(16)?;
    let source = TestReadAt::new(None);
    let held = Arc::clone(&cache.pages.reads)
        .acquire_many_owned(2)
        .await
        .map_err(|error| vortex_err!("{error}"))?;

    let speculative = cache
        .pages
        .page(Arc::from("object"), 0, 64, Arc::clone(&source) as _, true);
    let demand = cache
        .pages
        .page(Arc::from("object"), 0, 64, Arc::clone(&source) as _, false);
    assert!(speculative.await.is_err());

    drop(held);
    assert_eq!(demand.await?.as_ref(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(source.calls(), 1);
    assert_eq!(cache.stats().coalesced, 0);
    Ok(())
}

#[tokio::test]
async fn a_cancelled_miss_frees_its_permit_and_does_not_fail_later_reads() -> VortexResult<()> {
    let cache = cache(16)?;
    let gate = Arc::new(Semaphore::new(0));
    let source = TestReadAt::new(Some(Arc::clone(&gate)));
    let reader = wrap(&cache, "object", &source);

    let pending = (0..8)
        .map(|page| tokio::spawn(reader.read_at(page * 8, 4, Alignment::none())))
        .collect::<Vec<_>>();
    while source.calls() < 2 {
        tokio::task::yield_now().await;
    }
    assert_eq!(cache.pages.state.lock().inflight.len(), 2);

    for task in &pending {
        task.abort();
    }
    for task in pending {
        assert!(task.await.is_err_and(|error| error.is_cancelled()));
    }

    gate.add_permits(1);
    assert_eq!(read(&reader, 0, 4).await?, [0, 1, 2, 3]);
    assert_eq!(source.calls(), 3);
    assert_eq!(cache.stats().cached_bytes, 8);

    // A read that shares a load survives the cancellation of the read that started it.
    let original = tokio::spawn(reader.read_at(8, 4, Alignment::none()));
    while source.calls() < 4 {
        tokio::task::yield_now().await;
    }
    let surviving = {
        let reader = Arc::clone(&reader);
        tokio::spawn(async move { read(&reader, 8, 4).await })
    };
    while cache.stats().coalesced == 0 {
        tokio::task::yield_now().await;
    }

    original.abort();
    assert!(original.await.is_err_and(|error| error.is_cancelled()));

    gate.add_permits(1);
    let surviving = surviving.await.map_err(|error| vortex_err!("{error}"))??;
    assert_eq!(surviving, [8, 9, 10, 11]);
    assert_eq!(source.calls(), 4);
    assert_eq!(cache.stats().cached_bytes, 16);
    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn reads_fetch_the_following_pages_ahead() -> VortexResult<()> {
    let cache = PageCache::new(PageCacheConfig {
        capacity_bytes: 64,
        page_bytes: 8,
        max_in_flight: 4,
        read_ahead_pages: 2,
        max_prefetch_in_flight: 2,
    })?;
    let source = TestReadAt::new(None);
    let reader = wrap(&cache, "object", &source);

    read(&reader, 0, 4).await?;
    while cache.stats().physical_reads < 3 {
        tokio::task::yield_now().await;
    }

    let calls = source.calls();
    assert_eq!(read(&reader, 8, 16).await?, (8..24u8).collect::<Vec<_>>());
    assert_eq!(source.calls(), calls, "pages 1 and 2 were fetched ahead");
    Ok(())
}
