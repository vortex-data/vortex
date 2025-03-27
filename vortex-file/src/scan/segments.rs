use std::collections::{BTreeSet, VecDeque};
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, ready};
use std::time::Instant;

use dashmap::{DashMap, Entry};
use futures::channel::mpsc;
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::{FutureExt, TryFutureExt};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
};
use vortex_layout::segments::{SegmentId, SegmentReader};
use vortex_metrics::{Counter, VortexMetrics};

/// Pre-fetch queue for segments for the generic file reader.
///
/// Segments are prioritised by the order in which they are first requested, with explicitly
/// polled segments jumping to the front of the queue.
pub struct SegmentQueue {
    pub(crate) inner: Arc<SegmentQueueInner>,
}

struct SegmentQueueInner {
    /// A map of pending segments, indexed by segment ID.
    segments: DashMap<SegmentId, PendingSegment>,
    needed: Mutex<NeededSegments>,
    /// A queue of segments whose futures have been dropped.
    dead_queue: (
        mpsc::UnboundedSender<SegmentId>,
        mpsc::UnboundedReceiver<SegmentId>,
    ),
    metrics: VortexMetrics,
}

impl SegmentQueueInner {
    /// Drive the segment queue to perform more I/O.
    pub async fn drive(self: Arc<Self>) -> VortexResult<()> {
        Ok(())
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
                    self.metrics.counter("vortex.scan.segments.requests").inc();

                    log::debug!("Pending segment {} for {}: REGISTERED", id, &for_whom);
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
                        resolved: AtomicBool::new(false),
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
                        send: Mutex::new(Some(send)),
                    };
                    e.insert(pending);

                    self.needed
                        .lock()
                        .vortex_expect("poisoned lock")
                        .need_later
                        .insert(id);

                    break fut;
                }
            }
        }
    }

    /// Callback invoked when a segment future is first polled.
    fn on_first_poll(&self, id: SegmentId) {
        log::debug!("Pending segment {}: POLLED", id);
        if let Some(mut pending) = self.segments.get_mut(&id) {
            pending.polled = true;
        }

        // Bump the segment to the front of the queue.
        {
            let mut needed = self.needed.lock().vortex_expect("poisoned lock");
            if needed.need_later.remove(&id) {
                needed.need_now.push_back(id);
            }
        }
    }

    fn on_resolve(&self, id: SegmentId) {
        log::debug!("Pending segment {}: RESOLVED", id);
        if let Some(mut pending) = self.segments.get_mut(&id) {
            pending.resolved = true;
        }
    }

    fn on_drop(&self, id: SegmentId) {
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
        // We cannot lock the pending segment in a drop handler, since we will deadlock.
        // Instead, we place the ID into a dead queue.
        if self.dead_queue.0.unbounded_send(id).is_err() {
            log::trace!("Cannot submit to dead queue after drop")
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
    need_now: VecDeque<SegmentId>,
    // need_now: BTreeSet<SegmentId>,
    /// The set of known segments, ordered by SegmentID (which corresponds to byte offset).
    need_later: BTreeSet<SegmentId>,
}

impl SegmentQueue {
    /// Create a new segment queue, returning the queue and a segment reader that can be used to
    /// populate it.
    pub fn new(metrics: VortexMetrics) -> Self {
        let inner = Arc::new(SegmentQueueInner {
            segments: Default::default(),
            needed: Default::default(),
            dead_queue: mpsc::unbounded(),
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

    /// Drive the segment queue to completion.
    pub async fn drive(&self) -> VortexResult<()> {
        Ok(())
    }

    /// Inspect all pending segments.
    pub fn pending(&self) -> impl Iterator<Item = PendingSegmentLease> + '_ {
        self.inner
            .segments
            .iter()
            // Iterate in sorted order
            .sorted_unstable_by_key(|e| *e.key())
            .filter_map(|e| e.value().clone().lease())
    }

    /// Returns a future that resolves to the highest priority segment.
    /// Returns `None` if the queue has been closed.
    pub async fn next(&mut self) -> Option<PendingSegmentLease> {
        loop {
            {
                let mut needed = self.inner.needed.lock().vortex_expect("poisoned lock");

                // First, check the need_now queue. These segments have been explicitly polled
                // and therefore there is CPU work waiting for them.
                loop {
                    // In a loop, we drain the need_now queue until we find a segment that
                    // hasn't been dropped or leased.
                    let Some(segment_id) = needed.need_now.pop_front() else {
                        // No more need_now segments, break out of the loop.
                        break;
                    };

                    if let Some(lease) = self
                        .inner
                        .segments
                        .get(&segment_id)
                        .and_then(|p| p.clone().lease())
                    {
                        log::trace!("Fetching requested segment: {}", lease.id());
                        return Some(lease);
                    };
                }

                // Otherwise, we start pre-fetching the need_later segments.
                // FIXME(ngates): make this configurable, and ideally we have a lot more awareness
                //  of what each segment will be used for.
                loop {
                    let Some(segment_id) = needed.need_later.pop_first() else {
                        // No more need_later segments, break out of the loop.
                        break;
                    };

                    if let Some(lease) = self
                        .inner
                        .segments
                        .get(&segment_id)
                        .and_then(|p| p.clone().lease())
                    {
                        log::trace!("Fetching unrequested segment: {}", lease.id());
                        return Some(lease);
                    };
                }
            }

            // Perform some cleanup and throw away any dropped segments (those whose weak
            // reference can no longer be upgraded).
            // self.inner.segments.retain(|_, v| v.fut.upgrade().is_some());

            // Otherwise, await a notification that there may be more work to do.
            // self.recv.next().await?;
        }
    }

    /// Lease all segments within a given byte offset range.
    pub fn lease_within_range(&self, segment_range: &Range<SegmentId>) -> Vec<PendingSegmentLease> {
        let mut leased = vec![];

        let needed = self.inner.needed.lock().vortex_expect("poisoned lock");
        for segment_id in &needed.need_now {
            if segment_range.contains(segment_id) {
                if let Some(lease) = self
                    .inner
                    .segments
                    .get(segment_id)
                    .and_then(|v| v.clone().lease())
                {
                    leased.push(lease);
                }
            }
        }

        for segment_id in needed.need_later.range(segment_range.clone()) {
            if let Some(lease) = self
                .inner
                .segments
                .get(segment_id)
                .and_then(|v| v.clone().lease())
            {
                leased.push(lease);
            }
        }

        leased
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
    send: Mutex<Option<oneshot::Sender<VortexResult<ByteBuffer>>>>,

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

    /// Take a unique lease on the pending segment to resolve it some time later.
    pub fn lease(self: Arc<Self>) -> Option<PendingSegmentLease> {
        let send = self.send.lock().vortex_expect("poisoned lock").take();
        send.map(|send| PendingSegmentLease {
            pending: self,
            send: Some(send),
        })
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
    resolved: AtomicBool,
}

impl Future for SegmentFuture {
    type Output = SharedVortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.polled.fetch_or(true, Ordering::Relaxed) {
            self.queue.on_first_poll(self.id);
        }

        let result = ready!(self.future.poll_unpin(cx));
        if !self.resolved.fetch_or(true, Ordering::Relaxed) {
            self.queue.on_resolve(self.id);
        }
        Poll::Ready(result)
    }
}

impl Drop for SegmentFuture {
    fn drop(&mut self) {
        self.queue.on_drop(self.id);
    }
}

/// Lease the pending segment such that we know there is only one resolver at a time.
pub struct PendingSegmentLease {
    pending: Arc<PendingSegment>,
    send: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,
}

impl PendingSegmentLease {
    pub fn id(&self) -> SegmentId {
        self.pending.id
    }

    pub fn resolve(mut self, buffer: VortexResult<ByteBuffer>) {
        if self
            .send
            .take()
            .vortex_expect("cannot resolve a segment twice")
            .send(buffer)
            .is_err()
        {
            // This occurs when the recv end of the channel was dropped while the segment was
            // leased, in other words, while the request was "in-flight".
            log::trace!("Pending segment {}: DROPPED WHILE LEASED", self.id());
        }
    }
}

impl Drop for PendingSegmentLease {
    fn drop(&mut self) {
        *self.pending.send.lock().vortex_expect("poisoned lock") = self.send.take();
    }
}
