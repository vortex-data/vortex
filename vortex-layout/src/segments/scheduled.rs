// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::WeakShared;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_scan::read::CancelGroup;
use vortex_scan::read::ReadRequestKey;
use vortex_scan::read::ScanIoPhase;
use vortex_scan::read::ScanPriority;
use vortex_scan::read::ScanRead;
use vortex_scan::read::ScanReadRequest;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;
use vortex_utils::aliases::hash_set::HashSet;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Scheduler-visible metadata for one logical segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentInfo {
    /// Number of bytes in the logical segment payload.
    pub bytes: u64,
}

impl SegmentInfo {
    /// Create metadata for a segment with `bytes` payload bytes.
    pub fn new(bytes: u64) -> Self {
        Self { bytes }
    }
}

/// A scheduler-visible request for one logical segment payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentRequest {
    /// Logical segment id within the source.
    pub segment: SegmentId,
    /// Number of bytes in the logical segment payload.
    pub bytes: u64,
    /// High-level scan phase that needs this segment.
    pub phase: ScanIoPhase,
    /// Scheduler priority for this request.
    pub priority: ScanPriority,
    /// Cancellation scope for this request.
    pub cancel_group: CancelGroup,
}

/// Dedupe key for exact segment requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentRequestKey {
    /// Logical segment id within the source.
    pub segment: SegmentId,
}

impl SegmentRequestKey {
    /// Create a key for deduping exact segment requests.
    pub fn new(segment: SegmentId) -> Self {
        Self { segment }
    }
}

impl From<&SegmentRequest> for SegmentRequestKey {
    fn from(request: &SegmentRequest) -> Self {
        Self::new(request.segment)
    }
}

impl From<SegmentRequestKey> for ReadRequestKey {
    fn from(key: SegmentRequestKey) -> Self {
        Self::new(u64::from(*key.segment))
    }
}

impl From<&SegmentRequest> for ScanReadRequest {
    fn from(request: &SegmentRequest) -> Self {
        Self::new(
            ReadRequestKey::from(SegmentRequestKey::from(request)),
            request.bytes,
            request.phase,
        )
        .with_priority(request.priority)
        .with_cancel_group(request.cancel_group)
    }
}

impl SegmentRequest {
    /// Create a segment request from source, segment metadata, and phase.
    pub fn new(segment: SegmentId, info: SegmentInfo, phase: ScanIoPhase) -> Self {
        Self {
            segment,
            bytes: info.bytes,
            phase,
            priority: ScanPriority::NORMAL,
            cancel_group: CancelGroup::NONE,
        }
    }

    /// Return a copy of this request with the provided priority.
    pub fn with_priority(mut self, priority: ScanPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Return a copy of this request with the provided cancellation group.
    pub fn with_cancel_group(mut self, cancel_group: CancelGroup) -> Self {
        self.cancel_group = cancel_group;
        self
    }
}

type SharedSegmentFuture = BoxFuture<'static, SharedVortexResult<BufferHandle>>;

/// Snapshot of [`SegmentFutureCache`] activity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentFutureCacheStats {
    /// Number of requests passed to [`SegmentFutureCache::register`].
    pub registered_reads: u64,
    /// Number of same-batch duplicate requests skipped by [`SegmentFutureCache::register`].
    pub register_duplicates: u64,
    /// Number of calls to [`SegmentFutureCache::request_segment`].
    pub request_segment_reads: u64,
    /// Number of reads satisfied from an in-flight shared future.
    pub in_flight_hits: u64,
    /// Number of cache entries whose weak future had already expired.
    pub stale_in_flight: u64,
    /// Number of read futures submitted to the underlying [`SegmentSource`].
    pub submitted_reads: u64,
}

#[derive(Default)]
struct SegmentFutureCacheCounters {
    registered_reads: AtomicU64,
    register_duplicates: AtomicU64,
    request_segment_reads: AtomicU64,
    in_flight_hits: AtomicU64,
    stale_in_flight: AtomicU64,
    submitted_reads: AtomicU64,
}

impl SegmentFutureCacheCounters {
    fn snapshot(&self) -> SegmentFutureCacheStats {
        SegmentFutureCacheStats {
            registered_reads: self.registered_reads.load(Ordering::Relaxed),
            register_duplicates: self.register_duplicates.load(Ordering::Relaxed),
            request_segment_reads: self.request_segment_reads.load(Ordering::Relaxed),
            in_flight_hits: self.in_flight_hits.load(Ordering::Relaxed),
            stale_in_flight: self.stale_in_flight.load(Ordering::Relaxed),
            submitted_reads: self.submitted_reads.load(Ordering::Relaxed),
        }
    }

    fn increment(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn add(counter: &AtomicU64, value: usize) {
        counter.fetch_add(u64::try_from(value).unwrap_or(u64::MAX), Ordering::Relaxed);
    }
}

/// Scan-local cache of in-flight segment futures keyed by logical segment request.
///
/// The cache only stores weak references. Scheduled morsel futures and read calls hold the strong
/// futures that define lifetime; once they are dropped, a future cache entry may be replaced by a
/// later request for the same segment.
#[derive(Default)]
pub struct SegmentFutureCache {
    in_flight: DashMap<SegmentRequestKey, WeakShared<SharedSegmentFuture>>,
    counters: SegmentFutureCacheCounters,
}

impl SegmentFutureCache {
    /// Create an empty segment future cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a snapshot of cache activity counters.
    pub fn stats(&self) -> SegmentFutureCacheStats {
        self.counters.snapshot()
    }

    /// Request one segment from a scheduled source, reusing an in-flight future when present.
    pub fn request_segment(&self, source: &dyn SegmentSource, request: SegmentRequest) -> ScanRead {
        SegmentFutureCacheCounters::increment(&self.counters.request_segment_reads);

        loop {
            match self.in_flight.entry(SegmentRequestKey::from(&request)) {
                Entry::Occupied(entry) => {
                    if let Some(future) = entry.get().upgrade() {
                        SegmentFutureCacheCounters::increment(&self.counters.in_flight_hits);
                        return shared_segment_handle(request, future);
                    }
                    SegmentFutureCacheCounters::increment(&self.counters.stale_in_flight);
                    entry.remove();
                }
                Entry::Vacant(entry) => {
                    SegmentFutureCacheCounters::increment(&self.counters.submitted_reads);
                    let shared = source
                        .request(request.segment)
                        .map_err(Arc::new)
                        .boxed()
                        .shared();
                    entry.insert(
                        shared
                            .downgrade()
                            .vortex_expect("shared future was just created"),
                    );
                    return shared_segment_handle(request, shared);
                }
            }
        }
    }

    /// Register segment reads with a source, returning handles that keep the futures alive.
    pub fn register(
        &self,
        source: &dyn SegmentSource,
        requests: impl IntoIterator<Item = SegmentRequest>,
    ) -> Vec<ScanRead> {
        let mut seen: HashSet<SegmentRequestKey> = HashSet::default();
        let mut handles = Vec::new();
        let mut misses = Vec::new();
        for request in requests {
            SegmentFutureCacheCounters::increment(&self.counters.registered_reads);
            if !seen.insert(SegmentRequestKey::from(&request)) {
                SegmentFutureCacheCounters::increment(&self.counters.register_duplicates);
                continue;
            }
            if let Some(handle) = self.cached_handle(request) {
                handles.push(handle);
            } else {
                misses.push(request);
            }
        }

        handles.extend(self.submit_misses(source, misses));
        handles
    }

    fn cached_handle(&self, request: SegmentRequest) -> Option<ScanRead> {
        let key = SegmentRequestKey::from(&request);
        let entry = self.in_flight.get(&key)?;
        let Some(future) = entry.upgrade() else {
            SegmentFutureCacheCounters::increment(&self.counters.stale_in_flight);
            return None;
        };
        SegmentFutureCacheCounters::increment(&self.counters.in_flight_hits);
        Some(shared_segment_handle(request, future))
    }

    fn submit_misses(
        &self,
        source: &dyn SegmentSource,
        misses: Vec<SegmentRequest>,
    ) -> Vec<ScanRead> {
        SegmentFutureCacheCounters::add(&self.counters.submitted_reads, misses.len());
        self.insert_submitted(misses.into_iter().map(|request| {
            let future = source.request(request.segment);
            (request, future)
        }))
    }

    fn insert_submitted(
        &self,
        handles: impl IntoIterator<Item = (SegmentRequest, SegmentFuture)>,
    ) -> Vec<ScanRead> {
        handles
            .into_iter()
            .map(|(request, future)| {
                let shared = future.map_err(Arc::new).boxed().shared();
                self.in_flight.insert(
                    SegmentRequestKey::from(&request),
                    shared
                        .downgrade()
                        .vortex_expect("shared future was just created"),
                );
                shared_segment_handle(request, shared)
            })
            .collect()
    }
}

fn shared_segment_handle(request: SegmentRequest, future: Shared<SharedSegmentFuture>) -> ScanRead {
    shared_read_handle(ScanReadRequest::from(&request), future)
}

fn shared_read_handle(request: ScanReadRequest, future: Shared<SharedSegmentFuture>) -> ScanRead {
    ScanRead::new(request, future.map_err(VortexError::from).boxed())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use futures::executor::block_on;
    use parking_lot::Mutex;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use super::*;
    struct CountingSegmentSource {
        info: SegmentInfo,
        submit_count: AtomicUsize,
    }

    impl CountingSegmentSource {
        fn new(info: SegmentInfo) -> Self {
            Self {
                info,
                submit_count: AtomicUsize::new(0),
            }
        }

        fn submit_count(&self) -> usize {
            self.submit_count.load(Ordering::Relaxed)
        }
    }

    struct CountingMissSegmentSource {
        info: SegmentInfo,
        batches: Mutex<Vec<usize>>,
    }

    impl CountingMissSegmentSource {
        fn new(info: SegmentInfo) -> Self {
            Self {
                info,
                batches: Mutex::new(Vec::new()),
            }
        }

        fn batches(&self) -> Vec<usize> {
            self.batches.lock().clone()
        }
    }

    impl SegmentSource for CountingSegmentSource {
        fn segment_info(&self, _id: SegmentId) -> VortexResult<SegmentInfo> {
            Ok(self.info)
        }

        fn request(&self, _id: SegmentId) -> SegmentFuture {
            self.submit_count.fetch_add(1, Ordering::Relaxed);
            async move { Ok(BufferHandle::new_host(ByteBuffer::from(vec![0]))) }.boxed()
        }
    }

    impl SegmentSource for CountingMissSegmentSource {
        fn segment_info(&self, _id: SegmentId) -> VortexResult<SegmentInfo> {
            Ok(self.info)
        }

        fn request(&self, _id: SegmentId) -> SegmentFuture {
            self.batches.lock().push(1);
            async move { Ok(BufferHandle::new_host(ByteBuffer::from(vec![0]))) }.boxed()
        }
    }

    #[test]
    fn register_segment_reads_dedupes_exact_segments() -> VortexResult<()> {
        let source = Arc::new(CountingSegmentSource::new(SegmentInfo::new(8)));
        let segment = SegmentId::from(0);
        let request = SegmentRequest::new(
            segment,
            source.segment_info(segment)?,
            ScanIoPhase::ProjectionRead,
        );
        let cache = SegmentFutureCache::new();

        let reads = cache.register(source.as_ref(), vec![request, request]);

        assert_eq!(reads.len(), 1);
        assert_eq!(source.submit_count(), 1);
        assert_eq!(
            cache.stats(),
            SegmentFutureCacheStats {
                registered_reads: 2,
                register_duplicates: 1,
                submitted_reads: 1,
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn register_segment_reads_registers_each_miss() -> VortexResult<()> {
        let source = Arc::new(CountingMissSegmentSource::new(SegmentInfo::new(8)));
        let requests = (0..5)
            .map(|segment| {
                let segment = SegmentId::from(segment);
                Ok(SegmentRequest::new(
                    segment,
                    source.segment_info(segment)?,
                    ScanIoPhase::ProjectionRead,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let cache = SegmentFutureCache::new();

        let reads = cache.register(source.as_ref(), requests);

        assert_eq!(reads.len(), 5);
        assert_eq!(source.batches(), vec![1, 1, 1, 1, 1]);
        assert_eq!(
            cache.stats(),
            SegmentFutureCacheStats {
                registered_reads: 5,
                submitted_reads: 5,
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn segment_future_cache_reuses_prefetched_segment() -> VortexResult<()> {
        let source = Arc::new(CountingSegmentSource::new(SegmentInfo::new(8)));
        let segment = SegmentId::from(0);
        let request = SegmentRequest::new(
            segment,
            source.segment_info(segment)?,
            ScanIoPhase::ProjectionRead,
        );
        let cache = Arc::new(SegmentFutureCache::new());

        let reads = cache.register(source.as_ref(), vec![request]);
        let read = cache.request_segment(source.as_ref(), request);

        assert_eq!(reads.len(), 1);
        assert_eq!(source.submit_count(), 1);
        assert_eq!(block_on(read.future)?.as_host().len(), 1);
        assert_eq!(source.submit_count(), 1);
        assert_eq!(
            cache.stats(),
            SegmentFutureCacheStats {
                registered_reads: 1,
                request_segment_reads: 1,
                in_flight_hits: 1,
                submitted_reads: 1,
                ..Default::default()
            }
        );

        Ok(())
    }

    #[test]
    fn segment_future_cache_counts_stale_entries() -> VortexResult<()> {
        let source = Arc::new(CountingSegmentSource::new(SegmentInfo::new(8)));
        let segment = SegmentId::from(0);
        let request = SegmentRequest::new(
            segment,
            source.segment_info(segment)?,
            ScanIoPhase::ProjectionRead,
        );
        let cache = SegmentFutureCache::new();

        let reads = cache.register(source.as_ref(), vec![request]);
        drop(reads);

        let reads = cache.register(source.as_ref(), vec![request]);

        assert_eq!(reads.len(), 1);
        assert_eq!(source.submit_count(), 2);
        assert_eq!(
            cache.stats(),
            SegmentFutureCacheStats {
                registered_reads: 2,
                stale_in_flight: 1,
                submitted_reads: 2,
                ..Default::default()
            }
        );

        Ok(())
    }
}
