// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::WeakShared;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_scan::read::CancelGroup;
use vortex_scan::read::ReadRequestKey;
use vortex_scan::read::ReadResults;
use vortex_scan::read::ScanIoPhase;
use vortex_scan::read::ScanPriority;
use vortex_scan::read::ScanRead;
use vortex_scan::read::ScanReadRequest;
use vortex_session::VortexSession;
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

/// Planning result for segment request introspection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentRequests {
    exact: Option<Vec<SegmentRequest>>,
}

impl SegmentRequests {
    /// Return an unknown segment request set.
    pub fn unknown() -> Self {
        Self { exact: None }
    }

    /// Return an exact segment request set.
    pub fn exact(requests: Vec<SegmentRequest>) -> Self {
        Self {
            exact: Some(requests),
        }
    }

    /// Return an exact empty segment request set.
    pub fn none() -> Self {
        Self::exact(Vec::new())
    }

    /// Return whether this plan could not describe its segment requests.
    pub fn is_unknown(&self) -> bool {
        self.exact.is_none()
    }

    /// Borrow the exact request set, if known.
    pub fn as_exact(&self) -> Option<&[SegmentRequest]> {
        self.exact.as_deref()
    }

    /// Consume this value and return the exact request set, if known.
    pub fn into_exact(self) -> Option<Vec<SegmentRequest>> {
        self.exact
    }

    /// Append another request set, preserving `unknown` if either side is unknown.
    pub fn extend(&mut self, other: SegmentRequests) {
        match (&mut self.exact, other.exact) {
            (Some(requests), Some(mut other)) => requests.append(&mut other),
            _ => self.exact = None,
        }
    }
}

/// Context used by plans when producing scheduler-visible segment requests.
#[derive(Clone)]
pub struct SegmentPlanCtx {
    source: Arc<dyn SegmentSource>,
    session: VortexSession,
    phase: ScanIoPhase,
    priority: ScanPriority,
    cancel_group: CancelGroup,
}

impl fmt::Debug for SegmentPlanCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentPlanCtx")
            .field("phase", &self.phase)
            .field("priority", &self.priority)
            .field("cancel_group", &self.cancel_group)
            .finish_non_exhaustive()
    }
}

impl SegmentPlanCtx {
    /// Create a request-planning context for a registered segment source.
    pub fn new(source: Arc<dyn SegmentSource>, session: VortexSession) -> Self {
        Self {
            source,
            session,
            phase: ScanIoPhase::default(),
            priority: ScanPriority::NORMAL,
            cancel_group: CancelGroup::NONE,
        }
    }

    /// Return the source used to resolve segment metadata.
    pub fn source(&self) -> &Arc<dyn SegmentSource> {
        &self.source
    }

    /// Return the scan session used by lazy plans that must instantiate child plans.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Return a copy of this context using the provided scan phase.
    pub fn with_phase(mut self, phase: ScanIoPhase) -> Self {
        self.phase = phase;
        self
    }

    /// Return a copy of this context using the provided priority.
    pub fn with_priority(mut self, priority: ScanPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Return a copy of this context using the provided cancellation group.
    pub fn with_cancel_group(mut self, cancel_group: CancelGroup) -> Self {
        self.cancel_group = cancel_group;
        self
    }

    /// Create a segment request with this context's source and scheduling metadata.
    pub fn request(&self, segment: SegmentId, info: SegmentInfo) -> SegmentRequest {
        SegmentRequest::new(segment, info, self.phase)
            .with_priority(self.priority)
            .with_cancel_group(self.cancel_group)
    }

    /// Create a segment request after resolving metadata from the registered source.
    pub fn request_for_segment(&self, segment: SegmentId) -> VortexResult<SegmentRequest> {
        let info = self.source.segment_info(segment)?;
        Ok(self.request(segment, info))
    }
}

type SharedSegmentFuture = BoxFuture<'static, SharedVortexResult<BufferHandle>>;

/// Scan-local cache of in-flight segment futures keyed by logical segment request.
///
/// The cache only stores weak references. Scheduled morsel futures and read calls hold the strong
/// futures that define lifetime; once they are dropped, a future cache entry may be replaced by a
/// later request for the same segment.
#[derive(Default)]
pub struct SegmentFutureCache {
    in_flight: DashMap<SegmentRequestKey, WeakShared<SharedSegmentFuture>>,
}

impl SegmentFutureCache {
    /// Create an empty segment future cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request one segment from a scheduled source, reusing an in-flight future when present.
    pub fn request_segment(&self, source: &dyn SegmentSource, request: SegmentRequest) -> ScanRead {
        if let Some(handle) = self.cached_handle(request) {
            return handle;
        }

        loop {
            match self.in_flight.entry(SegmentRequestKey::from(&request)) {
                Entry::Occupied(entry) => {
                    if let Some(future) = entry.get().upgrade() {
                        return shared_segment_handle(request, future);
                    }
                    entry.remove();
                }
                Entry::Vacant(entry) => {
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

    /// Register exact segment reads with a source, returning handles that keep the futures alive.
    pub fn register(&self, source: &dyn SegmentSource, requests: SegmentRequests) -> Vec<ScanRead> {
        let Some(requests) = requests.into_exact() else {
            return Vec::new();
        };

        let mut seen: HashSet<SegmentRequestKey> = HashSet::default();
        let mut handles = Vec::new();
        let mut misses = Vec::new();
        for request in requests {
            if !seen.insert(SegmentRequestKey::from(&request)) {
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
        let future = self.in_flight.get(&key)?.upgrade()?;
        Some(shared_segment_handle(request, future))
    }

    fn submit_misses(
        &self,
        source: &dyn SegmentSource,
        misses: Vec<SegmentRequest>,
    ) -> Vec<ScanRead> {
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

/// Register exact segment reads through a shared in-flight future cache.
pub fn register_segment_reads_cached(
    cache: &SegmentFutureCache,
    source: &dyn SegmentSource,
    requests: SegmentRequests,
) -> Vec<ScanRead> {
    cache.register(source, requests)
}

fn shared_segment_handle(request: SegmentRequest, future: Shared<SharedSegmentFuture>) -> ScanRead {
    shared_read_handle(ScanReadRequest::from(&request), future)
}

fn shared_read_handle(request: ScanReadRequest, future: Shared<SharedSegmentFuture>) -> ScanRead {
    ScanRead::new(request, future.map_err(VortexError::from).boxed())
}

/// Segment-source view backed by another source and a [`SegmentFutureCache`].
pub struct CachedSegmentSource {
    source: Arc<dyn SegmentSource>,
    cache: Arc<SegmentFutureCache>,
    phase: ScanIoPhase,
}

/// Segment source backed by scheduler-resolved read results.
pub struct ReadResultsSegmentSource {
    source: Arc<dyn SegmentSource>,
    results: ReadResults,
}

impl ReadResultsSegmentSource {
    /// Create a segment source over already-resolved scan read results.
    pub fn new(source: Arc<dyn SegmentSource>, results: ReadResults) -> Self {
        Self { source, results }
    }
}

impl SegmentSource for ReadResultsSegmentSource {
    fn segment_info(&self, id: SegmentId) -> VortexResult<SegmentInfo> {
        self.source.segment_info(id)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        let key = ReadRequestKey::from(SegmentRequestKey::new(id));
        let results = self.results.clone();
        async move { results.get(key) }.boxed()
    }

    fn resolved(&self, id: SegmentId) -> VortexResult<BufferHandle> {
        self.results
            .get(ReadRequestKey::from(SegmentRequestKey::new(id)))
    }
}

impl CachedSegmentSource {
    /// Create a cached source using projection reads as the default late-request phase.
    pub fn new(source: Arc<dyn SegmentSource>, cache: Arc<SegmentFutureCache>) -> Self {
        Self {
            source,
            cache,
            phase: ScanIoPhase::ProjectionRead,
        }
    }

    /// Return a copy of this source with a different phase for late segment requests.
    pub fn with_phase(mut self, phase: ScanIoPhase) -> Self {
        self.phase = phase;
        self
    }
}

impl SegmentSource for CachedSegmentSource {
    fn segment_info(&self, id: SegmentId) -> VortexResult<SegmentInfo> {
        self.source.segment_info(id)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        let info = match self.source.segment_info(id) {
            Ok(info) => info,
            Err(error) => return async move { Err(error) }.boxed(),
        };
        self.cache
            .request_segment(
                self.source.as_ref(),
                SegmentRequest::new(id, info, self.phase),
            )
            .future
    }
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
        let segment_source: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);
        let ctx = SegmentPlanCtx::new(segment_source, VortexSession::empty());
        let request = ctx.request_for_segment(SegmentId::from(0))?;

        let reads = SegmentFutureCache::new().register(
            source.as_ref(),
            SegmentRequests::exact(vec![request, request]),
        );

        assert_eq!(reads.len(), 1);
        assert_eq!(source.submit_count(), 1);

        Ok(())
    }

    #[test]
    fn register_segment_reads_registers_each_miss() -> VortexResult<()> {
        let source = Arc::new(CountingMissSegmentSource::new(SegmentInfo::new(8)));
        let segment_source: Arc<dyn SegmentSource> =
            Arc::<CountingMissSegmentSource>::clone(&source);
        let ctx = SegmentPlanCtx::new(segment_source, VortexSession::empty());
        let requests = (0..5)
            .map(|segment| ctx.request_for_segment(SegmentId::from(segment)))
            .collect::<VortexResult<Vec<_>>>()?;

        let reads =
            SegmentFutureCache::new().register(source.as_ref(), SegmentRequests::exact(requests));

        assert_eq!(reads.len(), 5);
        assert_eq!(source.batches(), vec![1, 1, 1, 1, 1]);

        Ok(())
    }

    #[test]
    fn segment_future_cache_reuses_prefetched_segment() -> VortexResult<()> {
        let source = Arc::new(CountingSegmentSource::new(SegmentInfo::new(8)));
        let segment_source: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);
        let ctx = SegmentPlanCtx::new(Arc::clone(&segment_source), VortexSession::empty());
        let request = ctx.request_for_segment(SegmentId::from(0))?;
        let cache = Arc::new(SegmentFutureCache::new());

        let reads = cache.register(source.as_ref(), SegmentRequests::exact(vec![request]));
        let reader = CachedSegmentSource::new(segment_source, Arc::clone(&cache));
        let read = reader.request(SegmentId::from(0));

        assert_eq!(reads.len(), 1);
        assert_eq!(source.submit_count(), 1);
        assert_eq!(block_on(read)?.as_host().len(), 1);
        assert_eq!(source.submit_count(), 1);

        Ok(())
    }
}
