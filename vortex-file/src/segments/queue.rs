use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::task::{Context, Poll, ready};
use std::time::Instant;

use dashmap::{DashMap, Entry};
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt, TryFutureExt};
use linked_hash_set::LinkedHashSet;
use vortex_buffer::ByteBuffer;
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};
use vortex_metrics::VortexMetrics;

type Segments = Arc<RwLock<VecDeque<Weak<PendingSegment>>>>;

/// Pre-fetch queue for segments for the generic file reader.
///
/// Segments are prioritised by the order in which they are first requested, with explicitly
/// polled segments jumping to the front of the queue.
pub struct SegmentQueue {
    /// Notification queue triggered whenever a new segment is requested
    recv: mpsc::UnboundedReceiver<()>,
    inner: Arc<SegmentQueueInner>,
}

struct SegmentQueueInner {
    /// A map of pending segments, indexed by segment ID.
    segments: DashMap<SegmentId, Weak<PendingSegment>>,
    needed: Mutex<NeededSegments>,
    metrics: VortexMetrics,
}

#[derive(Default)]
struct NeededSegments {
    /// A queue of segments that have been explicitly requested (polled), but not yet resolved.
    need_now: VecDeque<SegmentId>,
    /// The set of known segments, sorted by insertion order.
    need_later: LinkedHashSet<SegmentId>,
}

enum SegmentEvent {
    /// The initial request for a segment.
    Requested(PendingSegment),
    /// A segment has been polled for the first time.
    Polled(SegmentId),
    /// A segment has been dropped.
    Dropped(SegmentId),
}

impl SegmentQueue {
    /// Create a new segment queue, returning the queue and a segment reader that can be used to
    /// populate it.
    pub fn new(metrics: VortexMetrics) -> (Self, Arc<dyn AsyncSegmentReader>) {
        let (send, recv) = mpsc::unbounded();

        let inner = Arc::new(SegmentQueueInner {
            segments: Default::default(),
            needed: Default::default(),
            metrics,
        });

        // We return a segment reader (instead of holding a strong reference to the send channel)
        // such that when all segment readers are dropped, the "send" end of the queue is closed,
        // and we can return `None` from the next function.
        let segment_reader = Arc::new(SegmentQueueSegmentReader {
            queue: inner.clone(),
            notifier: send,
        });

        (Self { recv, inner }, segment_reader)
    }

    /// Inspect all pending segments.
    pub fn pending(&self) -> impl Iterator<Item = PendingSegmentLease> + '_ {
        self.inner
            .segments
            .iter()
            .filter_map(|p| p.value().upgrade())
            .filter_map(|p| p.lease())
    }

    /// Returns a future that resolves to the highest priority segment.
    /// Returns `None` if the queue has been closed.
    pub async fn next(&mut self) -> Option<PendingSegmentLease> {
        loop {
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
                    .and_then(|p| p.upgrade())
                    .and_then(|p| p.lease())
                {
                    log::trace!("Fetching requested segment: {}", lease.id());
                    return Some(lease);
                };
            }

            // Otherwise, we start pre-fetching the need_later segments.
            // loop {
            //     let Some(segment_id) = needed.need_later.pop_front() else {
            //         // No more need_later segments, break out of the loop.
            //         break;
            //     };
            //
            //     if let Some(lease) = self
            //         .inner
            //         .segments
            //         .get(&segment_id)
            //         .and_then(|p| p.upgrade())
            //         .and_then(|p| p.lease())
            //     {
            //         log::trace!("Fetching unrequested segment: {}", lease.id());
            //         return Some(lease);
            //     };
            // }

            // Perform some cleanup and throw away any dropped segments (those whose weak
            // reference can no longer be upgraded).
            self.inner.segments.retain(|_, v| v.strong_count() > 0);

            // Before we await the future, we ensure we unlock the needed mutex.
            drop(needed);

            // Otherwise, await a notification that there may be more work to do.
            // FIXME(ngates): segment leasing might end up putting work back in the queue....?
            //  Either we need to notify on PendingSegmentLease::drop, or we need to not allow
            //  returning a lease to the queue. Probably this one honestly...
            if self.recv.next().await.is_none() {
                // Or exit if the queue has been closed, and we've consumed all notifications.
                // assert!(
                //     self.inner
                //         .needed
                //         .lock()
                //         .vortex_expect("poisoned lock")
                //         .need_now
                //         .is_empty(),
                //     "Segment queue closed with pending _requested_ segments"
                // );
                return None;
            }
        }
    }

    /// Try to lease the given segment.
    ///
    /// If the segment is dropped, remove it from the queue and return `None`.
    /// If the segment is already leased, return `None`.
    fn maybe_lease(&self, segment_id: Option<SegmentId>) -> Option<PendingSegmentLease> {
        if let Some(segment_id) = segment_id {
            if let Some(pending) = self.inner.segments.get(&segment_id) {
                if let Some(pending) = pending.value().upgrade() {
                    if pending.lease().is_some() {
                        return None;
                    }
                } else {
                    // Segment was dropped, remove it from the queue.
                    self.inner.segments.remove(&segment_id);
                }
            }
        }

        segment_id
            .and_then(|segment_id| self.inner.segments.get(&segment_id))
            .and_then(|p| p.value().upgrade())
            .and_then(|p| p.lease())
    }
}

/// Segment reader that creates a [`PendingSegment`] in the segment queue.
struct SegmentQueueSegmentReader {
    queue: Arc<SegmentQueueInner>,
    notifier: mpsc::UnboundedSender<()>,
}

impl AsyncSegmentReader for SegmentQueueSegmentReader {
    fn get(
        &self,
        id: SegmentId,
        for_whom: &Arc<str>,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let pending = loop {
            // Loop in case the pending segment has no strong references, in which case we clear it
            // out of the map and create a new one on the next iteration.
            match self.queue.segments.entry(id) {
                Entry::Occupied(e) => {
                    if let Some(pending) = e.get().upgrade() {
                        break pending;
                    } else {
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    let pending = PendingSegment::new(id, for_whom.clone(), self.queue.clone());
                    // Insert the pending segment into the map, return the strong shared future to the caller.
                    e.insert(Arc::downgrade(&pending));
                    self.queue
                        .needed
                        .lock()
                        .vortex_expect("poisoned lock")
                        .need_later
                        .insert_if_absent(id);
                    break pending;
                }
            }
        };

        // Send a notification that there may be more work to do in the queue.
        self.notifier
            .unbounded_send(())
            .map_err(|e| vortex_err!("Failed to notify segment queue {}", e))
            .vortex_expect("Failed to notify segment queue");

        PendingSegmentFuture {
            pending,
            notifier: self.notifier.clone(),
        }
        .boxed()
    }
}

/// A pending segment returned by the [`AsyncSegmentReader`].
pub struct PendingSegment {
    id: SegmentId,
    for_whom: Arc<str>,
    inner: Mutex<PendingSegmentInner>,
    queue: Arc<SegmentQueueInner>,
    created_at: Instant,
}

struct PendingSegmentInner {
    /// The sender end of the one-shot channel can be taken by leasing the segment.
    /// If the lease is dropped before resolving, the sender is put back into this field to allow
    /// another lease.
    send: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,
    /// The receiver end of the one-shot channel.
    recv: BoxFuture<'static, VortexResult<ByteBuffer>>,
    /// The cached result of the pending segment.
    result: Option<SharedVortexResult<ByteBuffer>>,
    /// Whether the segment has been polled yet.
    polled: bool,
}

impl Debug for PendingSegment {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSegment")
            .field("id", &self.id)
            .finish()
    }
}

impl PendingSegment {
    /// Create a new [`PendingSegment`] that can be resolved later.
    fn new(
        id: SegmentId,
        for_whom: Arc<str>,
        queue: Arc<SegmentQueueInner>,
    ) -> Arc<PendingSegment> {
        log::debug!("Pending segment {} for {}: REGISTERED", id, &for_whom);
        queue
            .metrics
            .counter("vortex.scan.segments.requested")
            .inc();

        let (send, recv) = oneshot::channel();

        Arc::new(Self {
            id,
            for_whom,
            inner: Mutex::new(PendingSegmentInner {
                send: Some(send),
                recv: recv
                    .map_err(|e| vortex_err!("pending segment sender dropped: {}", e))
                    .map(|r| r.unnest())
                    .boxed(),
                polled: false,
                result: None,
            }),
            queue,
            created_at: Instant::now(),
        })
    }

    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// Take a unique lease on the pending segment to resolve it some time later.
    pub fn lease(self: Arc<Self>) -> Option<PendingSegmentLease> {
        let mut this = self.inner.lock().vortex_expect("poisoned lock");
        this.send.take().map(|send| PendingSegmentLease {
            id: self.id,
            pending: Arc::downgrade(&self),
            send: Some(send),
        })
    }
}

impl Drop for PendingSegment {
    fn drop(&mut self) {
        let inner = self.inner.lock().vortex_expect("poisoned lock");
        match (inner.result.is_some(), inner.send.is_some()) {
            (false, true) => {
                log::debug!(
                    "Pending segment {} for {}: DROPPED BEFORE LAUNCH",
                    self.id,
                    &self.for_whom
                );
                self.queue
                    .metrics
                    .counter("vortex.scan.segments.dropped_before_launch")
                    .inc();
            }
            (false, false) => {
                log::debug!(
                    "Pending segment {} for {}: DROPPED AFTER LAUNCH",
                    self.id,
                    &self.for_whom
                );
                self.queue
                    .metrics
                    .counter("vortex.scan.segments.dropped_after_launch")
                    .inc();
            }
            (true, _) => {
                log::trace!(
                    "Pending segment {} for {}: DROPPED AFTER RESOLUTION",
                    self.id,
                    &self.for_whom
                );
                self.queue
                    .metrics
                    .counter("vortex.scan.segments.dropped_after_resolution")
                    .inc();
            }
        }
    }
}

/// A future that resolves when a pending segment is resolved.
///
/// It supports being polled multiple times, and will return the same result.
pub struct PendingSegmentFuture {
    pending: Arc<PendingSegment>,
    notifier: mpsc::UnboundedSender<()>,
}

impl Future for PendingSegmentFuture {
    type Output = VortexResult<ByteBuffer>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = self.pending.inner.lock().vortex_expect("poisoned lock");

        // Continue to return the same result if it is already resolved.
        if let Some(result) = &inner.result {
            return Poll::Ready(result.clone().map_err(VortexError::from));
        }

        // Trigger the on-poll callback if it exists.
        if !inner.polled {
            inner.polled = true;

            // Bump the segment to the front of the queue.
            let mut needed = self
                .pending
                .queue
                .needed
                .lock()
                .vortex_expect("poisoned lock");
            if needed.need_later.remove(&self.pending.id) {
                needed.need_now.push_back(self.pending.id);
            }

            // Notify the queue that there may be more work to do.
            let _ = self.notifier.unbounded_send(());
        }

        // If the result is not resolved, poll the receiver.
        let result = ready!(inner.recv.poll_unpin(cx)).map_err(Arc::new);

        // Store the result in the inner state and return.
        inner.result = Some(result.clone());

        Poll::Ready(result.map_err(VortexError::from))
    }
}

/// Lease the pending segment such that we know there is only one resolver at a time.
pub struct PendingSegmentLease {
    id: SegmentId,
    pending: Weak<PendingSegment>,
    send: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,
}

impl PendingSegmentLease {
    pub fn id(&self) -> SegmentId {
        self.id
    }

    pub fn resolve(mut self, buffer: VortexResult<ByteBuffer>) {
        if let Err(_) = self
            .send
            .take()
            .vortex_expect("cannot resolve a segment twice")
            .send(buffer)
        {
            // This occurs when the recv end of the channel was dropped while the segment was
            // leased, in other words, while the request was "in-flight".
            log::trace!("Pending segment {}: DROPPED WHILE LEASED", self.id);
        }

        if let Some(pending) = self.pending.upgrade() {
            pending
                .queue
                .metrics
                .timer("vortex.scan.segments.resolve")
                .update(Instant::now() - pending.created_at);
        }
    }
}

impl Drop for PendingSegmentLease {
    fn drop(&mut self) {
        if let Some(pending) = self.pending.upgrade() {
            pending.inner.lock().vortex_expect("poisoned lock").send = self.send.take();
        }
    }
}
