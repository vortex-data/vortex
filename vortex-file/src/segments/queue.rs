use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock, Weak};

use dashmap::{DashMap, Entry};
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::segments::pending::{PendingSegment, PendingSegmentLease};

type Segments = Arc<RwLock<VecDeque<Weak<PendingSegment>>>>;

/// Pre-fetch queue for segments for the generic file reader.
///
/// Segments are prioritised by the order in which they are requested.
pub struct SegmentQueue {
    inner: Arc<SegmentQueueInner>,
    recv: mpsc::UnboundedReceiver<()>,
}

#[derive(Default)]
struct SegmentQueueInner {
    /// A map of pending segments, indexed by segment ID.
    segments: DashMap<SegmentId, Weak<PendingSegment>>,
    /// The set of outstanding segments, sorted by insertion order.
    inserted: Mutex<VecDeque<SegmentId>>,
    /// A queue of segments that have been explicitly requested (polled), but not yet resolved.
    requested: Mutex<VecDeque<SegmentId>>,
}

impl SegmentQueue {
    /// Create a new segment queue, returning the queue and a segment reader that can be used to
    /// populate it.
    pub fn new() -> (Self, Arc<dyn AsyncSegmentReader>) {
        let inner: Arc<SegmentQueueInner> = Arc::new(Default::default());

        let (send, recv) = mpsc::unbounded();
        let this = Self {
            inner: inner.clone(),
            recv,
        };

        // We return a segment reader (instead of holding a strong reference to the send channel)
        // such that when all segment readers are dropped, the "send" end of the queue is closed,
        // and we can return `None` from the next function.
        let segment_reader = Arc::new(SegmentQueueSegmentReader {
            inner,
            notifier: Mutex::new(send),
        });

        (this, segment_reader)
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
            // Perform some cleanup and throw away any dropped segments (those whose weak
            // reference can no longer be upgraded).
            self.inner.segments.retain(|_, v| v.upgrade().is_some());

            // Await a notification that there may be more work to do.
            // FIXME(ngates): segment leasing might end up putting work back in the queue....?
            //  Either we need to notify on PendingSegmentLease::drop, or we need to not allow
            //  returning a lease to the queue. Probably this one honestly...
            if self.recv.next().await.is_none() {
                // Or exit if the queue has been closed, and we've consumed all notifications.
                assert!(
                    self.inner
                        .requested
                        .lock()
                        .vortex_expect("poisoned lock")
                        .is_empty(),
                    "Segment queue closed with pending _requested_ segments"
                );
                return None;
            }

            // First, check the requested queue. These segments have been explicitly polled
            // and therefore there is CPU work waiting for them.
            if let Some(lease) = self.maybe_lease(
                self.inner
                    .requested
                    .lock()
                    .vortex_expect("poisoned lock")
                    .pop_front(),
            ) {
                log::info!("Fetching requested segment: {}", lease.id());
                return Some(lease);
            }

            // Otherwise, we start pre-fetching the remaining segments.
            if let Some(lease) = self.maybe_lease(
                self.inner
                    .inserted
                    .lock()
                    .vortex_expect("poisoned lock")
                    .pop_front(),
            ) {
                log::info!("Fetching unrequested segment: {}", lease.id());
                return Some(lease);
            }
        }
    }

    fn maybe_lease(&self, segment_id: Option<SegmentId>) -> Option<PendingSegmentLease> {
        segment_id
            .and_then(|segment_id| self.inner.segments.get(&segment_id))
            .and_then(|p| p.value().upgrade())
            .and_then(|p| p.lease())
    }
}

/// Segment reader that creates a [`PendingSegment`] in the segment queue.
struct SegmentQueueSegmentReader {
    inner: Arc<SegmentQueueInner>,
    notifier: Mutex<mpsc::UnboundedSender<()>>,
}

impl AsyncSegmentReader for SegmentQueueSegmentReader {
    fn get(&self, id: SegmentId) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let pending = loop {
            // Loop in case the pending segment has no strong references, in which case we clear it
            // out of the map and create a new one on the next iteration.
            match self.inner.segments.entry(id) {
                Entry::Occupied(e) => {
                    if let Some(pending) = e.get().upgrade() {
                        break pending;
                    } else {
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    let inner = self.inner.clone();
                    let pending = PendingSegment::new(
                        id,
                        move || {
                            // On-poll, we insert the segment into the requested queue.
                            inner
                                .requested
                                .lock()
                                .vortex_expect("poisoned lock")
                                .push_back(id);
                        },
                        move || {},
                    );

                    // Insert the pending segment into the map, return the strong shared future to the caller.
                    e.insert(Arc::downgrade(&pending));
                    self.inner
                        .inserted
                        .lock()
                        .vortex_expect("poisoned lock")
                        .push_back(id);

                    break pending;
                }
            }
        };

        // Send a notification that there may be more work to do in the queue.
        self.notifier
            .lock()
            .vortex_expect("poisoned lock")
            .unbounded_send(())
            .map_err(|e| vortex_err!("Failed to notify segment queue {}", e))
            .vortex_expect("Failed to notify segment queue");

        pending.new_future().boxed()
    }
}
