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
use futures::{FutureExt, StreamExt, TryFutureExt};
use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};
use vortex_metrics::{Counter, VortexMetrics};

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
    segments: DashMap<SegmentId, Arc<PendingSegment>>,
    needed: Mutex<NeededSegments>,
    metrics: VortexMetrics,
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
    pub fn new(metrics: VortexMetrics) -> (Self, Arc<dyn AsyncSegmentReader>) {
        let (send, recv) = mpsc::unbounded();

        let inner = Arc::new(SegmentQueueInner {
            segments: Default::default(),
            needed: Default::default(),
            metrics: metrics.clone(),
        });

        // We return a segment reader (instead of holding a strong reference to the send channel)
        // such that when all segment readers are dropped, the "send" end of the queue is closed,
        // and we can return `None` from the next function.
        let segment_reader = Arc::new(SegmentQueueSegmentReader {
            queue: inner.clone(),
            notifier: send,
            request_counter: metrics.counter("vortex.scan.segments.requested"),
        });

        (Self { recv, inner }, segment_reader)
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
            }

            // Otherwise, we start pre-fetching the need_later segments.
            // FIXME(ngates): make this configurable, and ideally we have a lot more awareness
            //  of what each segment will be used for.
            // loop {
            //     let Some(segment_id) = needed.need_later.pop_first() else {
            //         // No more need_later segments, break out of the loop.
            //         break;
            //     };
            //
            //     if let Some(lease) = self
            //         .inner
            //         .segments
            //         .get(&segment_id)
            //         .and_then(|p| p.clone().lease())
            //     {
            //         log::trace!("Fetching unrequested segment: {}", lease.id());
            //         return Some(lease);
            //     };
            // }

            // Perform some cleanup and throw away any dropped segments (those whose weak
            // reference can no longer be upgraded).
            // self.inner.segments.retain(|_, v| v.fut.upgrade().is_some());

            // Otherwise, await a notification that there may be more work to do.
            if self.recv.next().await.is_none() {
                return None;
            }
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
                .get(&segment_id)
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
    notifier: mpsc::UnboundedSender<()>,

    request_counter: Arc<Counter>,
}

impl AsyncSegmentReader for SegmentQueueSegmentReader {
    fn get(
        &self,
        id: SegmentId,
        for_whom: &Arc<str>,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let fut = loop {
            // Loop in case the pending future has no strong references, in which case we clear it
            // out of the map and create a new one on the next iteration.
            match self.queue.segments.entry(id) {
                Entry::Occupied(e) => {
                    if let Some(fut) = e.get().future() {
                        break fut;
                    } else {
                        log::warn!("Removing dropped segment from segment reader {}", id);
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    self.request_counter.inc();
                    let (pending, fut) = PendingSegment::new(
                        id,
                        for_whom.clone(),
                        self.queue.clone(),
                        self.notifier.clone(),
                    );
                    e.insert(Arc::new(pending));
                    self.queue
                        .needed
                        .lock()
                        .vortex_expect("poisoned lock")
                        .need_later
                        .insert(id);
                    break fut;
                }
            }
        };

        // Send a notification that there may be more work to do in the queue.
        self.notifier
            .unbounded_send(())
            .map_err(|e| vortex_err!("Failed to notify segment queue {}", e))
            .vortex_expect("Failed to notify segment queue");

        fut.map_err(VortexError::from).boxed()
    }
}

/// A pending segment returned by the [`AsyncSegmentReader`].
pub struct PendingSegment {
    id: SegmentId,
    for_whom: Arc<str>,
    fut: WeakShared<SegmentFuture>,
    send: Mutex<Option<oneshot::Sender<VortexResult<ByteBuffer>>>>,
    // inner: Mutex<PendingSegmentInner>,
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
        notifier: mpsc::UnboundedSender<()>,
    ) -> (PendingSegment, Shared<SegmentFuture>) {
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
            queue: queue.clone(),
            notifier,
            polled: AtomicBool::new(false),
            resolved: AtomicBool::new(false),
            // resolved_timer: queue.metrics.timer("vortex.scan.segments.resolve"),
        }
        .shared();

        let this = Self {
            id,
            for_whom,
            fut: fut
                .downgrade()
                .vortex_expect("future has not been polled to completion"),
            queue,
            created_at: Instant::now(),
            send: Mutex::new(Some(send)),
        };

        (this, fut)
    }

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
    notifier: mpsc::UnboundedSender<()>,
    polled: AtomicBool,
    resolved: AtomicBool,
    // resolved_timer: Arc<Timer>,
}

impl Future for SegmentFuture {
    type Output = SharedVortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.polled.fetch_or(true, Ordering::Relaxed) {
            // Bump the segment to the front of the queue.
            {
                let mut needed = self.queue.needed.lock().vortex_expect("poisoned lock");
                if needed.need_later.remove(&self.id) {
                    needed.need_now.push_back(self.id);
                }
            }

            // Notify the queue that there may be more work to do.
            let _ = self.notifier.unbounded_send(());
        }

        let result = ready!(self.future.poll_unpin(cx));
        self.resolved.store(true, Ordering::Relaxed);
        // self.resolved_timer
        //     .update(Instant::now() - self.pending.created_at);
        Poll::Ready(result)
    }
}

impl Drop for SegmentFuture {
    fn drop(&mut self) {
        match (
            self.polled.load(Ordering::Relaxed),
            self.resolved.load(Ordering::Relaxed),
        ) {
            (false, false) => {
                log::debug!("Pending segment {}: DROPPED BEFORE POLL", self.id);
            }
            (false, true) => {
                log::debug!(
                    "Pending segment {}: DROPPED BEFORE POLL AFTER RESOLVE",
                    self.id
                );
            }
            (true, false) => {
                log::debug!("Pending segment {}: DROPPED BEFORE RESOLVE", self.id);
            }
            (true, true) => {
                // This is not an interesting case, the future resolved to completion.
                log::trace!("Pending segment {}: DROPPED AFTER RESOLVE", self.id);
            }
        }
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
        if let Err(_) = self
            .send
            .take()
            .vortex_expect("cannot resolve a segment twice")
            .send(buffer)
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
