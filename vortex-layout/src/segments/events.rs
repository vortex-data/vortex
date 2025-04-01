use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, atomic};
use std::task::{Context, Poll};
use std::time::Instant;

use dashmap::{DashMap, Entry};
use futures::channel::{mpsc, oneshot};
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryFutureExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
};
use vortex_metrics::{Counter, VortexMetrics};

use crate::segments::{SegmentId, SegmentSource};

/// A utility for turning a [`SegmentSource`] into a stream of [`SegmentEvent`]s.
///
/// Also performs de-duplication of requests for the same segment, while tracking when the all
/// requesters have been dropped.
pub struct SegmentEvents {
    pending: DashMap<SegmentId, PendingSegment>,
    events: mpsc::UnboundedSender<SegmentEvent>,
    metrics: VortexMetrics,
}

impl SegmentEvents {
    pub fn create(
        metrics: VortexMetrics,
    ) -> (Arc<dyn SegmentSource>, BoxStream<'static, SegmentEvent>) {
        let (send, recv) = mpsc::unbounded();

        let events = Arc::new(Self {
            pending: Default::default(),
            events: send,
            metrics: metrics.clone(),
        });

        let source = Arc::new(EventsSegmentSource {
            events: events.clone(),
            request_counter: events.metrics.counter("vortex.scan.segments.requested"),
        });
        let stream = recv.boxed();

        (source, stream)
    }
}

pub enum SegmentEvent {
    Requested(SegmentRequest),
    Polled(SegmentId),
    Dropped(SegmentId),
}

impl Debug for SegmentEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SegmentEvent::Requested(id) => write!(f, "SegmentEvent::Registered({:?})", id),
            SegmentEvent::Polled(id) => write!(f, "SegmentEvent::Polled({:?})", id),
            SegmentEvent::Dropped(id) => write!(f, "SegmentEvent::Dropped({:?})", id),
        }
    }
}

#[derive(Debug)]
pub struct SegmentRequest {
    // The ID of the requested segment
    id: SegmentId,
    // The one-shot channel to send the segment back to the caller
    callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl SegmentRequest {
    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// Resolve the segment request with the given buffer result.
    pub fn resolve(self, buffer: VortexResult<ByteBuffer>) {
        self.callback
            .send(buffer)
            .map_err(|_| vortex_err!("send failed"))
            .vortex_expect("send failed");
    }
}

impl SegmentEvents {
    /// Get or create a segment future for the given segment ID.
    fn segment_future(
        self: Arc<Self>,
        id: SegmentId,
        for_whom: Arc<str>,
    ) -> Shared<SegmentEventsFuture> {
        loop {
            // Loop in case the pending future has no strong references, in which case we clear it
            // out of the map and create a new one on the next iteration.
            match self.pending.entry(id) {
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
                    let fut = SegmentEventsFuture {
                        future: recv
                            .map_err(|e| vortex_err!("pending segment receiver dropped: {}", e))
                            .map(|r| r.unnest())
                            .map_err(Arc::new)
                            .boxed(),
                        id,
                        source: self.clone(),
                        polled: AtomicBool::new(false),
                    }
                    .shared();

                    // Create a new pending segment.
                    let pending = PendingSegment {
                        id,
                        for_whom,
                        created_at: Instant::now(),
                        fut: fut
                            .downgrade()
                            .vortex_expect("cannot fail, only just created"),
                    };
                    e.insert(pending);

                    // Set up a SegmentRequest tied to the send end of the channel.
                    self.submit_event(SegmentEvent::Requested(SegmentRequest {
                        id,
                        callback: send,
                    }));

                    break fut;
                }
            }
        }
    }

    /// Submit a segment event.
    fn submit_event(&self, event: SegmentEvent) {
        if self.events.unbounded_send(event).is_err() {
            log::trace!("Segment queue shutting down, no problem if we lose events")
        }
    }
}

struct EventsSegmentSource {
    events: Arc<SegmentEvents>,
    request_counter: Arc<Counter>,
}

impl SegmentSource for EventsSegmentSource {
    fn request(
        &self,
        id: SegmentId,
        for_whom: &Arc<str>,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        self.request_counter.inc();
        self.events
            .clone()
            .segment_future(id, for_whom.clone())
            .map_err(VortexError::from)
            .boxed()
    }
}

/// A pending segment returned by the [`SegmentSource`].
pub struct PendingSegment {
    id: SegmentId,
    /// A debug string identifying which layout requested the segment.
    for_whom: Arc<str>,
    /// The time at which the segment was requested.
    created_at: Instant,
    /// A weak shared future that we hand out to all requesters. Once all requesters have been
    /// dropped, typically because their row split has completed (or been pruned), then the weak
    /// feature is no longer upgradable, and the segment can be dropped.
    fut: WeakShared<SegmentEventsFuture>,
}

impl Debug for PendingSegment {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSegment")
            .field("id", &self.id)
            .field("for_whom", &self.for_whom)
            .field("created_at", &self.created_at)
            .finish()
    }
}

impl PendingSegment {
    /// Create a new future resolving this segment, provided the segment is still alive.
    fn future(&self) -> Option<Shared<SegmentEventsFuture>> {
        self.fut.upgrade()
    }
}

/// A future that notifies the segment queue when it is first polled, as well as logging
/// when it is dropped.
struct SegmentEventsFuture {
    future: BoxFuture<'static, SharedVortexResult<ByteBuffer>>,
    id: SegmentId,
    source: Arc<SegmentEvents>,
    polled: AtomicBool,
}

impl Future for SegmentEventsFuture {
    type Output = SharedVortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.polled.fetch_or(true, atomic::Ordering::Relaxed) {
            self.source.submit_event(SegmentEvent::Polled(self.id));
        }
        self.future.poll_unpin(cx)
    }
}

impl Drop for SegmentEventsFuture {
    fn drop(&mut self) {
        self.source.submit_event(SegmentEvent::Dropped(self.id));
    }
}
