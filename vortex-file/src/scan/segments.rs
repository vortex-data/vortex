use std::cmp::Ordering;
use std::collections::{BTreeSet, VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, atomic};
use std::task::{Context, Poll, ready};
use std::time::Instant;

use dashmap::{DashMap, Entry};
use futures::channel::mpsc;
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::stream::ReadyChunks;
use futures::{FutureExt, StreamExt, TryFutureExt};
use itertools::Itertools;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
    vortex_panic,
};
use vortex_io::{PerformanceHint, VortexReadAt};
use vortex_layout::segments::{SegmentId, SegmentReader};
use vortex_metrics::{Counter, VortexMetrics};

use crate::SegmentSpec;
use crate::segments::SegmentCache;

/// Pre-fetch queue for segments for the generic file reader.
///
/// Segments are prioritised by the order in which they are first requested, with explicitly
/// polled segments jumping to the front of the queue.
pub struct SegmentQueue {
    pub inner: Arc<SegmentQueueInner>,
}

pub struct SegmentQueueInner {
    /// A map of pending segments, indexed by segment ID.
    segments: DashMap<SegmentId, PendingSegment>,
    /// Map of the segment locations within a file.
    segment_map: Arc<[SegmentSpec]>,
    /// Cache of segments passed to us from file open, typically from the initial read.
    segment_cache: Arc<dyn SegmentCache>,

    needed: Mutex<NeededSegments>,
    /// A queue of segment events as well as a condition variable to wake up the driver.
    events: (
        mpsc::UnboundedSender<SegmentEvent>,
        Mutex<ReadyChunks<mpsc::UnboundedReceiver<SegmentEvent>>>,
    ),
    metrics: VortexMetrics,
}

enum SegmentEvent {
    Registered(SegmentId),
    Polled(SegmentId),
    Dropped(SegmentId),
    Resolve(SegmentId, VortexResult<ByteBuffer>),
}

impl Debug for SegmentEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SegmentEvent::Registered(id) => write!(f, "SegmentEvent::Registered({:?})", id),
            SegmentEvent::Polled(id) => write!(f, "SegmentEvent::Polled({:?})", id),
            SegmentEvent::Dropped(id) => write!(f, "SegmentEvent::Dropped({:?})", id),
            SegmentEvent::Resolve(id, _) => write!(f, "SegmentEvent::Resolve({:?})", id),
        }
    }
}

impl SegmentQueueInner {
    /// Drive the segment queue to perform more I/O.
    ///
    /// The given performance hint helps identify coalescing opportunities.
    pub async fn drive(
        self: Arc<Self>,
        performance_hint: &PerformanceHint,
    ) -> VortexResult<Option<CoalescedSegmentRequest>> {
        // Process any outstanding events in order to bring our state up to date.
        let next_events = self
            .events
            .1
            .lock()
            .vortex_expect("poisoned lock")
            .next()
            .await
            .unwrap_or_default();

        // Since we buffer ready segments, we take all available segments on each iteration.
        log::debug!("Processing {} segment events", next_events.len());
        for event in next_events {
            log::debug!("Processing segment event {:?}", event);
            match event {
                SegmentEvent::Registered(id) => self.on_registered(id).await,
                SegmentEvent::Polled(id) => self.on_polled(id).await,
                SegmentEvent::Dropped(id) => self.on_dropped(id).await,
                SegmentEvent::Resolve(id, buffer_result) => {
                    self.on_resolve(id, buffer_result).await
                }
            }
        }

        let Some(next) = self.next_segment_request() else {
            // There's no more work to do
            return Ok(None);
        };
        let next_spec = self
            .segment_map
            .get(*next.id as usize)
            .ok_or_else(|| vortex_err!("SegmentID {} not found", next.id))?;

        // We build up a single coalesced read from the pending segments.
        // Since pending segments are ordered by priority, we _always_ launch a request
        // for the highest priority segment.
        let mut coalesced = CoalescedSegmentRequest {
            alignment: next_spec.alignment,
            byte_range: next_spec.offset..next_spec.offset + next_spec.length as u64,
            requests: vec![next],
        };

        let window = performance_hint.coalescing_window();
        let max_read = performance_hint.max_read();

        // We keep expanding our coalesced window until we reach max_read or no more segments
        // can be coalesced.
        loop {
            let lowest_segment = self.segment_map.partition_point(|s| {
                (s.offset + s.length as u64) < coalesced.byte_range.start.saturating_sub(window)
            });
            let highest_segment = self
                .segment_map
                .partition_point(|s| s.offset < coalesced.byte_range.end.saturating_add(window));

            // Loop over the segments within the coalescing range and add them into the request.
            for id in lowest_segment..highest_segment {
                let id = SegmentId::try_from(id)?;
                let Some(request) = self
                    .segments
                    .get_mut(&id)
                    .and_then(|mut pending| pending.send.take())
                    .map(|callback| SegmentRequest { id, callback })
                else {
                    continue;
                };

                let spec = self
                    .segment_map
                    .get(*id as usize)
                    .ok_or_else(|| vortex_err!("SegmentID {} not found", id))?;

                let segment_start = spec.offset;
                let segment_end = spec.offset + spec.length as u64;

                coalesced.byte_range.start = coalesced.byte_range.start.min(segment_start);
                coalesced.byte_range.end = coalesced.byte_range.end.max(segment_end);
                // Take the maximum alignment of all segments in the coalesced request.
                // FIXME(ngates): shouldn't this be the _first_ segment?
                coalesced.alignment = coalesced.alignment.max(spec.alignment);
                coalesced.requests.push(request);
            }

            if let Some(max_read) = max_read {
                if coalesced.byte_range.end - coalesced.byte_range.start > max_read {
                    break;
                }
            }
        }

        // Ensure the coalesced requests are sorted
        coalesced.requests.sort_by_key(|r| r.id);

        Ok(Some(coalesced))
    }

    fn next_segment_request(&self) -> Option<SegmentRequest> {
        let mut needed = self.needed.lock().vortex_expect("poisoned lock");

        // First, we sort the "need now" queue by priority.
        self.sort_by_priority(&mut needed.need_now);
        if let Some(next) = needed
            .need_now
            .iter()
            .filter_map(|id| {
                self.segments
                    .get_mut(id)
                    .and_then(|mut pending| pending.send.take())
                    .map(|send| SegmentRequest {
                        id: *id,
                        callback: send,
                    })
            })
            .next()
        {
            return Some(next);
        }

        // Otherwise, we fall back to the need later queue.
        self.sort_by_priority(&mut needed.need_later);
        needed
            .need_later
            .iter()
            .filter_map(|id| {
                self.segments
                    .get_mut(id)
                    .and_then(|mut pending| pending.send.take())
                    .map(|send| SegmentRequest {
                        id: *id,
                        callback: send,
                    })
            })
            .next()
    }

    fn sort_by_priority(&self, needed: &mut Vec<SegmentId>) {
        needed.sort_unstable_by_key(|id| {
            self.segments
                .get(id)
                .map(|pending| {
                    // We prioritize approximate filter above all else, and then we prioritize by
                    // row offset.
                    (
                        !matches!(pending.stage, ScanStage::ApproxFilter),
                        pending.row_range.start,
                    )
                })
                .unwrap_or_else(|| {
                    // If the segment has been dropped, we put it last.
                    (true, u64::MAX)
                });
        })
    }

    /// Event handler for [`SegmentEvent::Registered`].
    async fn on_registered(&self, id: SegmentId) {
        // On registration, check the cache to see if we can resolve it immediately.
        if let Some(buffer_result) = self.segment_cache.get(id).transpose() {
            self.submit_event(SegmentEvent::Resolve(id, buffer_result));
            return;
        }

        // Otherwise, insert it into the "need later" queue until it's polled.
        let mut needed = self.needed.lock().vortex_expect("poisoned lock");
        needed.need_later.push(id);
    }

    /// Event handler for [`SegmentEvent::Polled`].
    async fn on_polled(&self, id: SegmentId) {
        // The first time a segment is polled, we bump it to the front of the queue.
        let mut needed = self.needed.lock().vortex_expect("poisoned lock");
        needed.need_later.retain(|i| i != &id);
        needed.need_now.push(id);
        if let Some(mut pending) = self.segments.get_mut(&id) {
            pending.polled = true;
        }
    }

    /// Event handler for [`SegmentEvent::Dropped`].
    async fn on_dropped(&self, id: SegmentId) {
        if let Some(pending) = self.segments.get(&id) {
            match (pending.polled, pending.resolved) {
                (false, false) => {
                    log::debug!("Pending segment {}: DROPPED BEFORE POLL", id);
                }
                (false, true) => {
                    log::debug!("Pending segment {}: DROPPED BEFORE POLL AFTER RESOLVE", id);
                }
                (true, false) => {
                    log::debug!("Pending segment {}: DROPPED BEFORE RESOLVE", id);
                }
                (true, true) => {
                    // This is not an interesting case, the future resolved to completion.
                    log::trace!("Pending segment {}: DROPPED AFTER RESOLVE", id);
                }
            }
        }

        // If a segment is dropped, we need to remove it from the queue.
        let mut needed = self.needed.lock().vortex_expect("poisoned lock");
        needed.need_now.retain(|&x| x != id);
        needed.need_later.retain(|&x| x != id);
        self.segments.remove(&id);
    }

    /// Event handler for [`SegmentEvent::Resolve`].
    async fn on_resolve(&self, id: SegmentId, buffer_result: VortexResult<ByteBuffer>) {
        if let Some(mut pending) = self.segments.get_mut(&id) {
            pending.resolved = true;
            if let Some(send) = pending.send.take() {
                if let Err(e) = send.send(buffer_result) {
                    log::trace!("Segment future {} was dropped while resolving: {}", id, e);
                }
            }
        }
    }

    /// Get or create a segment future for the given segment ID.
    fn segment_future(
        self: Arc<Self>,
        id: SegmentId,
        for_whom: Arc<str>,
        row_range: Range<u64>,
        stage: ScanStage,
    ) -> Shared<SegmentFuture> {
        loop {
            // Loop in case the pending future has no strong references, in which case we clear it
            // out of the map and create a new one on the next iteration.
            match self.segments.entry(id) {
                Entry::Occupied(e) => {
                    if let Some(fut) = e.get().future() {
                        break fut;
                    } else {
                        log::debug!("Re-requesting dropped segment from segment reader {}", id);
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    log::debug!("Pending segment {} for {}: REGISTERED", id, &for_whom);
                    self.metrics.counter("vortex.scan.segments.requests").inc();
                    let (send, recv) = oneshot::channel::<VortexResult<ByteBuffer>>();

                    // Set up the segment future tied to the recv end of the channel.
                    let fut = SegmentFuture {
                        future: recv
                            .map_err(|e| vortex_err!("pending segment receiver dropped: {}", e))
                            .map(|r| r.unnest())
                            .map_err(Arc::new)
                            .boxed(),
                        id,
                        queue: self.clone(),
                        polled: AtomicBool::new(false),
                    }
                    .shared();

                    let pending = PendingSegment {
                        id,
                        row_range,
                        stage,
                        for_whom,
                        created_at: Instant::now(),
                        resolved: false,
                        polled: false,
                        fut: fut
                            .downgrade()
                            .vortex_expect("future must be alive, we only just created it"),
                        queue: self.clone(),
                        send: Some(send),
                    };
                    e.insert(pending);

                    self.submit_event(SegmentEvent::Registered(id));

                    break fut;
                }
            }
        }
    }

    /// Submit a segment event.
    fn submit_event(&self, event: SegmentEvent) {
        if self.events.0.unbounded_send(event).is_err() {
            log::trace!("Segment queue shutting down, no problem if we lose events")
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ScanStage {
    ApproxFilter,
    ExactFilter,
    Projection,
}

#[derive(Default)]
struct NeededSegments {
    /// A queue of segments that have been explicitly requested (polled), but not yet resolved,
    /// ordered by when they were polled.
    need_now: Vec<SegmentId>,
    // need_now: BTreeSet<SegmentId>,
    /// The set of known segments, ordered by SegmentID (which corresponds to byte offset).
    need_later: Vec<SegmentId>,
}

impl SegmentQueue {
    /// Create a new segment queue, returning the queue and a segment reader that can be used to
    /// populate it.
    pub fn new(
        segment_map: Arc<[SegmentSpec]>,
        segment_cache: Arc<dyn SegmentCache>,
        metrics: VortexMetrics,
    ) -> Self {
        let (send, recv) = mpsc::unbounded();

        let inner = Arc::new(SegmentQueueInner {
            segments: Default::default(),
            segment_map,
            segment_cache,
            needed: Default::default(),
            events: (send, Mutex::new(recv.ready_chunks(1024))),
            metrics: metrics.clone(),
        });

        Self { inner }
    }

    /// Create a new [`SegmentReader`] that can be used to populate the segment queue.
    pub fn segment_reader(
        &self,
        row_range: &Range<u64>,
        stage: ScanStage,
    ) -> Arc<dyn SegmentReader> {
        // We return a segment reader (instead of holding a strong reference to the send channel)
        // such that when all segment readers are dropped, the "send" end of the queue is closed,
        // and we can return `None` from the next function.
        Arc::new(SegmentQueueSegmentReader {
            queue: self.inner.clone(),
            row_range: row_range.clone(),
            stage,
            request_counter: self.inner.metrics.counter("vortex.scan.segments.requested"),
        })
    }
}

/// Segment reader that creates a [`PendingSegment`] in the segment queue.
struct SegmentQueueSegmentReader {
    queue: Arc<SegmentQueueInner>,
    row_range: Range<u64>,
    stage: ScanStage,

    request_counter: Arc<Counter>,
}

impl SegmentQueueSegmentReader {
    pub fn new(queue: Arc<SegmentQueueInner>, row_range: Range<u64>, stage: ScanStage) -> Self {
        let request_counter = queue.metrics.counter("vortex.scan.segments.requested");
        Self {
            queue,
            row_range,
            stage,
            request_counter,
        }
    }
}

impl SegmentReader for SegmentQueueSegmentReader {
    fn get(
        &self,
        id: SegmentId,
        for_whom: &Arc<str>,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        self.queue
            .clone()
            .segment_future(id, for_whom.clone(), self.row_range.clone(), self.stage)
            .map_err(VortexError::from)
            .boxed()
    }
}

/// A pending segment returned by the [`SegmentReader`].
pub struct PendingSegment {
    id: SegmentId,
    /// The row range of the scan that requested the segment.
    row_range: Range<u64>,
    /// The stage of the scan that requested the segment.
    stage: ScanStage,
    /// A debug string identifying which layout requested the segment.
    for_whom: Arc<str>,
    /// The time at which the segment was requested.
    created_at: Instant,

    /// Whether the segment has been resolved.
    resolved: bool,
    /// Whether the segment has been polled.
    polled: bool,

    /// A weak shared future that we hand out to all requesters. Once all requesters have been
    /// dropped, typically because their row split has completed (or been pruned), then the weak
    /// feature is no longer upgradable, and the segment can be dropped.
    fut: WeakShared<SegmentFuture>,

    /// A channel that can be used to resolve the segment future.
    send: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,

    /// Handle back into the queue state.
    queue: Arc<SegmentQueueInner>,
}

impl Debug for PendingSegment {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSegment")
            .field("id", &self.id)
            .finish()
    }
}

impl PendingSegment {
    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// Create a new future resolving this segment, provided the segment is still alive.
    pub fn future(&self) -> Option<Shared<SegmentFuture>> {
        self.fut.upgrade()
    }
}

/// A future that notifies the segment queue when it is first polled, as well as logging
/// when it is dropped.
pub struct SegmentFuture {
    future: BoxFuture<'static, SharedVortexResult<ByteBuffer>>,
    // FIXME(ngates): just call queue.on_poll(id).
    id: SegmentId,
    queue: Arc<SegmentQueueInner>,
    polled: AtomicBool,
}

impl Future for SegmentFuture {
    type Output = SharedVortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.polled.fetch_or(true, atomic::Ordering::Relaxed) {
            self.queue.submit_event(SegmentEvent::Polled(self.id));
        }
        self.future.poll_unpin(cx)
    }
}

impl Drop for SegmentFuture {
    fn drop(&mut self) {
        self.queue.submit_event(SegmentEvent::Dropped(self.id));
    }
}

#[derive(Debug)]
pub struct SegmentRequest {
    // The ID of the requested segment
    pub id: SegmentId,
    // The one-shot channel to send the segment back to the caller
    pub callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl SegmentRequest {
    pub fn resolve(self, buffer: VortexResult<ByteBuffer>) {
        self.callback
            .send(buffer)
            .map_err(|_| vortex_err!("send failed"))
            .vortex_expect("send failed");
    }
}

#[derive(Debug)]
struct CoalescedSegmentRequest {
    /// The alignment of the first segment.
    // TODO(ngates): is this the best alignment to use?
    pub(crate) alignment: Alignment,
    /// The range of the file to read.
    pub(crate) byte_range: Range<u64>,
    /// The original segment requests, ordered by segment ID.
    pub(crate) requests: Vec<SegmentRequest>,
}

impl CoalescedSegmentRequest {
    fn size_bytes(&self) -> u64 {
        self.byte_range.end - self.byte_range.start
    }
}

pub(crate) async fn evaluate<R: VortexReadAt + Send>(
    read: R,
    request: CoalescedSegmentRequest,
    segment_map: Arc<[SegmentSpec]>,
) -> VortexResult<()> {
    log::debug!(
        "Reading byte range for [{}] requests {:?} size={}",
        request.requests.iter().map(|r| r.id).join(", "),
        request.byte_range,
        request.byte_range.end - request.byte_range.start,
    );
    let buffer: ByteBuffer = read
        .read_byte_range(request.byte_range.clone(), request.alignment)
        .await?
        .aligned(Alignment::none());

    // Figure out the segments covered by the read.
    let start = segment_map.partition_point(|s| s.offset < request.byte_range.start);
    let end = segment_map.partition_point(|s| s.offset < request.byte_range.end);

    // Note that we may have multiple requests for the same segment.
    let mut requests_iter = request.requests.into_iter().peekable();

    for (i, segment) in segment_map[start..end].iter().enumerate() {
        let segment_id = SegmentId::from(u32::try_from(i + start).vortex_expect("segment id"));
        let offset = usize::try_from(segment.offset - request.byte_range.start)?;
        let buf = buffer
            .slice(offset..offset + segment.length as usize)
            .aligned(segment.alignment);

        // Find any request callbacks and send the buffer
        while let Some(req) = requests_iter.peek() {
            // If the request is before the current segment, we should have already resolved it.
            match req.id.cmp(&segment_id) {
                Ordering::Less => {
                    // This should never happen, it means we missed a segment request.
                    vortex_panic!("Skipped segment request");
                }
                Ordering::Equal => {
                    // Resolve the request
                    let _ = requests_iter
                        .next()
                        .vortex_expect("next request")
                        .callback //
                        .send(Ok(buf.clone()));
                }
                Ordering::Greater => {
                    // No request for this segment, so we continue
                    break;
                }
            }
        }
    }

    Ok(())
}
