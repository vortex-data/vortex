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
use vortex_error::vortex_err;
use vortex_scan::SegmentSourceId;
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
    /// Whether this segment is eligible for segment-cache lookup and admission.
    pub cacheable: bool,
}

impl SegmentInfo {
    /// Create cacheable metadata for a segment with `bytes` payload bytes.
    pub fn cacheable(bytes: u64) -> Self {
        Self {
            bytes,
            cacheable: true,
        }
    }

    /// Create non-cacheable metadata for a segment with `bytes` payload bytes.
    pub fn non_cacheable(bytes: u64) -> Self {
        Self {
            bytes,
            cacheable: false,
        }
    }
}

/// High-level scan phase associated with a segment request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScanIoPhase {
    /// Shared evidence setup, such as loading a stats table.
    EvidenceSetup,
    /// Per-morsel evidence probe.
    EvidenceProbe,
    /// Residual predicate value read.
    PredicateRead,
    /// Projected output value read.
    #[default]
    ProjectionRead,
    /// Aggregate input or metadata read.
    AggregateRead,
}

/// Scheduler priority for a segment request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScanPriority(i32);

impl ScanPriority {
    /// Normal request priority.
    pub const NORMAL: Self = Self(0);

    /// Create a priority from a signed integer value.
    pub fn new(value: i32) -> Self {
        Self(value)
    }

    /// Return the signed integer priority value.
    pub fn get(self) -> i32 {
        self.0
    }
}

/// Cancellation scope for a group of related segment requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CancelGroup(u64);

impl CancelGroup {
    /// A request that is not associated with a finer cancellation group.
    pub const NONE: Self = Self(0);

    /// Create a cancellation group from an integer id.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the integer cancellation group id.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// A scheduler-visible request for one logical segment payload.
///
/// The first scheduler API intentionally only models segment payloads. If a future custom
/// `ScanNode` needs opaque or non-segment I/O, add that request shape next to this type rather
/// than smuggling physical locations into `SegmentRequest`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentRequest {
    /// Registered source that owns the segment.
    pub source: SegmentSourceId,
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
    /// Registered source that owns the segment.
    pub source: SegmentSourceId,
    /// Logical segment id within the source.
    pub segment: SegmentId,
}

impl SegmentRequestKey {
    /// Create a key for deduping exact segment requests.
    pub fn new(source: SegmentSourceId, segment: SegmentId) -> Self {
        Self { source, segment }
    }
}

impl From<&SegmentRequest> for SegmentRequestKey {
    fn from(request: &SegmentRequest) -> Self {
        Self::new(request.source, request.segment)
    }
}

impl SegmentRequest {
    /// Create a segment request from source, segment metadata, and phase.
    pub fn new(
        source: SegmentSourceId,
        segment: SegmentId,
        info: SegmentInfo,
        phase: ScanIoPhase,
    ) -> Self {
        Self {
            source,
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
    source_id: SegmentSourceId,
    source: Arc<dyn ScheduledSegmentSource>,
    session: VortexSession,
    phase: ScanIoPhase,
    priority: ScanPriority,
    cancel_group: CancelGroup,
}

impl fmt::Debug for SegmentPlanCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentPlanCtx")
            .field("source_id", &self.source_id)
            .field("phase", &self.phase)
            .field("priority", &self.priority)
            .field("cancel_group", &self.cancel_group)
            .finish_non_exhaustive()
    }
}

impl SegmentPlanCtx {
    /// Create a request-planning context for a registered segment source.
    pub fn new(
        source_id: SegmentSourceId,
        source: Arc<dyn ScheduledSegmentSource>,
        session: VortexSession,
    ) -> Self {
        Self {
            source_id,
            source,
            session,
            phase: ScanIoPhase::default(),
            priority: ScanPriority::NORMAL,
            cancel_group: CancelGroup::NONE,
        }
    }

    /// Return the registered source id used for requests created by this context.
    pub fn source_id(&self) -> SegmentSourceId {
        self.source_id
    }

    /// Return the scheduled source used to resolve segment metadata.
    pub fn source(&self) -> &Arc<dyn ScheduledSegmentSource> {
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
        SegmentRequest::new(self.source_id, segment, info, self.phase)
            .with_priority(self.priority)
            .with_cancel_group(self.cancel_group)
    }

    /// Create a segment request after resolving metadata from the registered source.
    pub fn request_for_segment(&self, segment: SegmentId) -> VortexResult<SegmentRequest> {
        let info = self.source.segment_info(segment)?;
        Ok(self.request(segment, info))
    }
}

/// Backend capabilities relevant to scheduled segment submission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentSourceCapabilities {
    /// Maximum number of logical segment requests preferred in one batch.
    pub max_batch_len: Option<usize>,
    /// Maximum number of logical segment bytes preferred in one batch.
    pub max_batch_bytes: Option<u64>,
    /// Whether the backend can observe best-effort cancellation.
    pub supports_cancellation: bool,
}

/// A batch of segment requests submitted to one scheduled segment source.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SegmentBatch {
    requests: Vec<SegmentRequest>,
}

impl SegmentBatch {
    /// Create a batch from requests that all target the same source.
    pub fn new(requests: Vec<SegmentRequest>) -> Self {
        Self { requests }
    }

    /// Borrow the requests in this batch.
    pub fn requests(&self) -> &[SegmentRequest] {
        &self.requests
    }

    /// Consume this batch and return its requests.
    pub fn into_requests(self) -> Vec<SegmentRequest> {
        self.requests
    }

    /// Return the number of requests in this batch.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Return whether this batch contains no requests.
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// One logical segment result returned by a scheduled source submission.
pub struct SegmentHandle {
    /// The logical request this handle resolves.
    pub request: SegmentRequest,
    /// Future resolving to the requested segment payload.
    pub future: SegmentFuture,
}

impl SegmentHandle {
    /// Create a handle for one logical segment request.
    pub fn new(request: SegmentRequest, future: SegmentFuture) -> Self {
        Self { request, future }
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
    pub fn request_segment(
        &self,
        source: &dyn ScheduledSegmentSource,
        request: SegmentRequest,
    ) -> SegmentHandle {
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
                    let Some(handle) = source
                        .submit(SegmentBatch::new(vec![request]))
                        .into_iter()
                        .next()
                    else {
                        return missing_segment_handle(request);
                    };
                    let shared = handle.future.map_err(Arc::new).boxed().shared();
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

    /// Submit exact segment requests to a source, returning strong handles that keep them alive.
    pub fn submit(
        &self,
        source: &dyn ScheduledSegmentSource,
        requests: SegmentRequests,
    ) -> SubmittedSegmentRequests {
        let Some(requests) = requests.into_exact() else {
            return SubmittedSegmentRequests::default();
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
        SubmittedSegmentRequests::new(handles)
    }

    fn cached_handle(&self, request: SegmentRequest) -> Option<SegmentHandle> {
        let key = SegmentRequestKey::from(&request);
        let future = self.in_flight.get(&key)?.upgrade()?;
        Some(shared_segment_handle(request, future))
    }

    fn submit_misses(
        &self,
        source: &dyn ScheduledSegmentSource,
        misses: Vec<SegmentRequest>,
    ) -> Vec<SegmentHandle> {
        let capabilities = source.capabilities();
        let mut handles = Vec::new();
        let mut batch = Vec::new();
        let mut batch_bytes = 0_u64;
        for request in misses {
            let len_limit_reached = capabilities
                .max_batch_len
                .is_some_and(|max_len| !batch.is_empty() && batch.len() >= max_len);
            let bytes_limit_reached = capabilities.max_batch_bytes.is_some_and(|max_bytes| {
                !batch.is_empty() && batch_bytes.saturating_add(request.bytes) > max_bytes
            });
            if len_limit_reached || bytes_limit_reached {
                handles.extend(self.insert_submitted(
                    source.submit(SegmentBatch::new(std::mem::take(&mut batch))),
                ));
                batch_bytes = 0;
            }
            batch_bytes = batch_bytes.saturating_add(request.bytes);
            batch.push(request);
        }
        if !batch.is_empty() {
            handles.extend(self.insert_submitted(source.submit(SegmentBatch::new(batch))));
        }
        handles
    }

    fn insert_submitted(&self, handles: Vec<SegmentHandle>) -> Vec<SegmentHandle> {
        handles
            .into_iter()
            .map(|handle| {
                let request = handle.request;
                let shared = handle.future.map_err(Arc::new).boxed().shared();
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

/// Submitted segment request handles.
///
/// Holding this value keeps pre-submitted segment futures alive. The existing file-backed source
/// registers reads when those futures are created, so a later layout read can coalesce with them
/// even if this value never awaits the handles directly.
#[derive(Default)]
pub struct SubmittedSegmentRequests {
    handles: Vec<SegmentHandle>,
    bytes: u64,
}

impl SubmittedSegmentRequests {
    /// Create a submitted request set from handles.
    pub fn new(handles: Vec<SegmentHandle>) -> Self {
        let bytes = handles
            .iter()
            .map(|handle| handle.request.bytes)
            .sum::<u64>();
        Self { handles, bytes }
    }

    /// Borrow submitted segment handles.
    pub fn handles(&self) -> &[SegmentHandle] {
        &self.handles
    }

    /// Return the number of submitted segment handles.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Return the total logical segment bytes represented by these handles.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Return whether no segment handles were submitted.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Extend this submitted request set with another, keeping all handles alive.
    pub fn extend(&mut self, other: SubmittedSegmentRequests) {
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.handles.extend(other.handles);
    }
}

/// Submit exact segment requests to a source after deduping by `(source, segment)`.
///
/// Unknown request sets are left to the normal lazy read path and submit no work. The source still
/// owns physical coalescing; this helper only removes duplicate logical segment requests.
pub fn submit_segment_requests(
    source: &dyn ScheduledSegmentSource,
    requests: SegmentRequests,
) -> SubmittedSegmentRequests {
    SegmentFutureCache::new().submit(source, requests)
}

/// Submit exact segment requests through a shared in-flight future cache.
pub fn submit_segment_requests_cached(
    cache: &SegmentFutureCache,
    source: &dyn ScheduledSegmentSource,
    requests: SegmentRequests,
) -> SubmittedSegmentRequests {
    cache.submit(source, requests)
}

fn shared_segment_handle(
    request: SegmentRequest,
    future: Shared<SharedSegmentFuture>,
) -> SegmentHandle {
    SegmentHandle::new(request, future.map_err(VortexError::from).boxed())
}

fn missing_segment_handle(request: SegmentRequest) -> SegmentHandle {
    SegmentHandle::new(
        request,
        async move {
            Err(vortex_err!(
                "scheduled source did not return a handle for segment {}",
                request.segment
            ))
        }
        .boxed(),
    )
}

/// Segment-source view backed by a [`ScheduledSegmentSource`] and [`SegmentFutureCache`].
pub struct ScheduledSegmentSourceReader {
    source_id: SegmentSourceId,
    source: Arc<dyn ScheduledSegmentSource>,
    cache: Arc<SegmentFutureCache>,
    phase: ScanIoPhase,
}

impl ScheduledSegmentSourceReader {
    /// Create a segment source reader using projection reads as the default late-request phase.
    pub fn new(
        source_id: SegmentSourceId,
        source: Arc<dyn ScheduledSegmentSource>,
        cache: Arc<SegmentFutureCache>,
    ) -> Self {
        Self {
            source_id,
            source,
            cache,
            phase: ScanIoPhase::ProjectionRead,
        }
    }

    /// Return a copy of this reader with a different phase for late segment requests.
    pub fn with_phase(mut self, phase: ScanIoPhase) -> Self {
        self.phase = phase;
        self
    }
}

impl SegmentSource for ScheduledSegmentSourceReader {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let info = match self.source.segment_info(id) {
            Ok(info) => info,
            Err(error) => return async move { Err(error) }.boxed(),
        };
        self.cache
            .request_segment(
                self.source.as_ref(),
                SegmentRequest::new(self.source_id, id, info, self.phase),
            )
            .future
    }
}

/// Source that accepts explicit scheduler-visible segment batches.
pub trait ScheduledSegmentSource: Send + Sync + 'static {
    /// Return scheduler-visible metadata for a segment.
    fn segment_info(&self, id: SegmentId) -> VortexResult<SegmentInfo>;

    /// Return backend capabilities relevant to scheduling and batching.
    fn capabilities(&self) -> SegmentSourceCapabilities {
        SegmentSourceCapabilities::default()
    }

    /// Submit a batch of segment requests to this source.
    fn submit(&self, batch: SegmentBatch) -> Vec<SegmentHandle>;
}

/// Adapter that exposes an existing [`SegmentSource`] as a scheduled segment source.
pub struct ScheduledSegmentSourceAdapter {
    source: Arc<dyn SegmentSource>,
    segments: Arc<[SegmentInfo]>,
    capabilities: SegmentSourceCapabilities,
}

impl ScheduledSegmentSourceAdapter {
    /// Create a scheduled adapter over an existing segment source.
    pub fn new(source: Arc<dyn SegmentSource>, segments: Arc<[SegmentInfo]>) -> Self {
        Self {
            source,
            segments,
            capabilities: SegmentSourceCapabilities::default(),
        }
    }

    /// Return a copy of this adapter with explicit capabilities.
    pub fn with_capabilities(mut self, capabilities: SegmentSourceCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Return the wrapped segment source.
    pub fn source(&self) -> &Arc<dyn SegmentSource> {
        &self.source
    }
}

impl ScheduledSegmentSource for ScheduledSegmentSourceAdapter {
    fn segment_info(&self, id: SegmentId) -> VortexResult<SegmentInfo> {
        let idx = usize::try_from(*id).map_err(|_| vortex_err!("segment id exceeds usize"))?;
        self.segments
            .get(idx)
            .copied()
            .ok_or_else(|| vortex_err!("missing segment: {}", id))
    }

    fn capabilities(&self) -> SegmentSourceCapabilities {
        self.capabilities
    }

    fn submit(&self, batch: SegmentBatch) -> Vec<SegmentHandle> {
        batch
            .into_requests()
            .into_iter()
            .map(|request| {
                let future = self.source.request(request.segment);
                SegmentHandle::new(request, future)
            })
            .collect()
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
    use vortex_scan::ScanMeta;
    use vortex_scan::ScanScheduler;
    use vortex_scan::SegmentSourceMeta;

    use super::*;

    struct TestSegmentSource;

    impl SegmentSource for TestSegmentSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            async move {
                let id = u8::try_from(*id).map_err(|_| vortex_err!("segment id exceeds u8"))?;
                Ok(BufferHandle::new_host(ByteBuffer::from(vec![id])))
            }
            .boxed()
        }
    }

    struct CountingScheduledSegmentSource {
        info: SegmentInfo,
        submit_count: AtomicUsize,
    }

    impl CountingScheduledSegmentSource {
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

    struct BatchingScheduledSegmentSource {
        info: SegmentInfo,
        batches: Mutex<Vec<usize>>,
    }

    impl BatchingScheduledSegmentSource {
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

    impl ScheduledSegmentSource for CountingScheduledSegmentSource {
        fn segment_info(&self, _id: SegmentId) -> VortexResult<SegmentInfo> {
            Ok(self.info)
        }

        fn submit(&self, batch: SegmentBatch) -> Vec<SegmentHandle> {
            self.submit_count.fetch_add(batch.len(), Ordering::Relaxed);
            batch
                .into_requests()
                .into_iter()
                .map(|request| {
                    let future =
                        async move { Ok(BufferHandle::new_host(ByteBuffer::from(vec![0]))) }
                            .boxed();
                    SegmentHandle::new(request, future)
                })
                .collect()
        }
    }

    impl ScheduledSegmentSource for BatchingScheduledSegmentSource {
        fn segment_info(&self, _id: SegmentId) -> VortexResult<SegmentInfo> {
            Ok(self.info)
        }

        fn capabilities(&self) -> SegmentSourceCapabilities {
            SegmentSourceCapabilities {
                max_batch_len: Some(2),
                max_batch_bytes: Some(16),
                supports_cancellation: false,
            }
        }

        fn submit(&self, batch: SegmentBatch) -> Vec<SegmentHandle> {
            self.batches.lock().push(batch.len());
            batch
                .into_requests()
                .into_iter()
                .map(|request| {
                    let future =
                        async move { Ok(BufferHandle::new_host(ByteBuffer::from(vec![0]))) }
                            .boxed();
                    SegmentHandle::new(request, future)
                })
                .collect()
        }
    }

    #[test]
    fn adapter_reports_metadata_and_submits_handles() -> VortexResult<()> {
        let adapter = Arc::new(ScheduledSegmentSourceAdapter::new(
            Arc::new(TestSegmentSource),
            vec![SegmentInfo::cacheable(4), SegmentInfo::cacheable(8)].into(),
        ));
        let scheduler = ScanScheduler::unbounded();
        let ticket = scheduler.register_scan(ScanMeta::default());
        let source_id =
            ticket.register_segment_source(Arc::clone(&adapter), SegmentSourceMeta::default());

        let scheduled: Arc<dyn ScheduledSegmentSource> =
            Arc::<ScheduledSegmentSourceAdapter>::clone(&adapter);
        let ctx = SegmentPlanCtx::new(source_id, scheduled, VortexSession::empty())
            .with_phase(ScanIoPhase::PredicateRead);
        let request = ctx.request_for_segment(SegmentId::from(1))?;

        assert_eq!(request.bytes, 8);
        assert_eq!(request.phase, ScanIoPhase::PredicateRead);

        let mut handles = adapter.submit(SegmentBatch::new(vec![request]));
        let handle = handles
            .pop()
            .ok_or_else(|| vortex_err!("scheduled adapter did not return a handle"))?;
        assert_eq!(handle.request.segment, SegmentId::from(1));
        assert_eq!(block_on(handle.future)?.as_host().len(), 1);

        Ok(())
    }

    #[test]
    fn submit_segment_requests_dedupes_exact_segments() -> VortexResult<()> {
        let scheduler = ScanScheduler::unbounded();
        let ticket = scheduler.register_scan(ScanMeta::default());
        let source = Arc::new(CountingScheduledSegmentSource::new(SegmentInfo::cacheable(
            8,
        )));
        let source_id =
            ticket.register_segment_source(Arc::clone(&source), SegmentSourceMeta::default());
        let scheduled: Arc<dyn ScheduledSegmentSource> =
            Arc::<CountingScheduledSegmentSource>::clone(&source);
        let ctx = SegmentPlanCtx::new(source_id, scheduled, VortexSession::empty());
        let request = ctx.request_for_segment(SegmentId::from(0))?;

        let submitted = submit_segment_requests(
            source.as_ref(),
            SegmentRequests::exact(vec![request, request]),
        );

        assert_eq!(submitted.len(), 1);
        assert_eq!(source.submit_count(), 1);

        Ok(())
    }

    #[test]
    fn submit_segment_requests_respects_batch_capabilities() -> VortexResult<()> {
        let scheduler = ScanScheduler::unbounded();
        let ticket = scheduler.register_scan(ScanMeta::default());
        let source = Arc::new(BatchingScheduledSegmentSource::new(SegmentInfo::cacheable(
            8,
        )));
        let source_id =
            ticket.register_segment_source(Arc::clone(&source), SegmentSourceMeta::default());
        let scheduled: Arc<dyn ScheduledSegmentSource> =
            Arc::<BatchingScheduledSegmentSource>::clone(&source);
        let ctx = SegmentPlanCtx::new(source_id, scheduled, VortexSession::empty());
        let requests = (0..5)
            .map(|segment| ctx.request_for_segment(SegmentId::from(segment)))
            .collect::<VortexResult<Vec<_>>>()?;

        let submitted = submit_segment_requests(source.as_ref(), SegmentRequests::exact(requests));

        assert_eq!(submitted.len(), 5);
        assert_eq!(source.batches(), vec![2, 2, 1]);

        Ok(())
    }

    #[test]
    fn segment_future_cache_reuses_prefetched_segment() -> VortexResult<()> {
        let scheduler = ScanScheduler::unbounded();
        let ticket = scheduler.register_scan(ScanMeta::default());
        let source = Arc::new(CountingScheduledSegmentSource::new(SegmentInfo::cacheable(
            8,
        )));
        let source_id =
            ticket.register_segment_source(Arc::clone(&source), SegmentSourceMeta::default());
        let scheduled: Arc<dyn ScheduledSegmentSource> =
            Arc::<CountingScheduledSegmentSource>::clone(&source);
        let ctx = SegmentPlanCtx::new(source_id, Arc::clone(&scheduled), VortexSession::empty());
        let request = ctx.request_for_segment(SegmentId::from(0))?;
        let cache = Arc::new(SegmentFutureCache::new());

        let submitted = submit_segment_requests_cached(
            cache.as_ref(),
            source.as_ref(),
            SegmentRequests::exact(vec![request]),
        );
        let reader = ScheduledSegmentSourceReader::new(source_id, scheduled, Arc::clone(&cache));
        let read = reader.request(SegmentId::from(0));

        assert_eq!(submitted.len(), 1);
        assert_eq!(source.submit_count(), 1);
        assert_eq!(block_on(read)?.as_host().len(), 1);
        assert_eq!(source.submit_count(), 1);

        Ok(())
    }
}
