use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock, Weak};

use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_layout::segments::{AsyncSegmentReader, PendingSegment, PendingSegmentLease, SegmentId};

type Segments = Arc<RwLock<VecDeque<Weak<PendingSegment>>>>;

/// Pre-fetch queue for segments for the generic file reader.
pub struct SegmentQueue {
    segments: Segments,
    send: mpsc::UnboundedSender<()>,
    recv: mpsc::UnboundedReceiver<()>,
}

impl SegmentQueue {
    /// Create a new segment queue, returning the queue and a segment reader that can be used to
    /// populate it.
    pub fn new() -> (Self, Arc<dyn AsyncSegmentReader>) {
        let segments = Arc::new(RwLock::new(VecDeque::default()));

        let (send, recv) = mpsc::unbounded();
        let this = Self {
            segments: segments.clone(),
            send: send.clone(),
            recv,
        };

        // We return a segment reader (instead of holding a strong reference to the send channel)
        // such that when all segment readers are dropped, the "send" end of the queue is closed,
        // and we can return `None` from the next function.
        let segment_reader = Arc::new(SegmentQueueSegmentReader {
            segments,
            notifier: Mutex::new(send),
        });

        (this, segment_reader)
    }

    /// Inspect all pending segments, in order of segment ID.
    /// TODO(ngates): we want this in order of request, not segment ID.
    pub fn with_pending_segments<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut dyn Iterator<Item = PendingSegmentLease>) -> T,
    {
        let mut segments = self.segments.write().vortex_expect("poisoned lock");

        let result = f(&mut segments
            .iter()
            .filter_map(|p| p.upgrade())
            .filter_map(|p| p.lease()));

        segments.retain(|p| p.upgrade().is_some());

        result
    }

    /// Returns a future that resolves when a new segment has been requested, or all segment
    /// readers have been dropped.
    pub async fn next(&mut self) -> Option<()> {
        self.recv.next().await
    }
}

struct SegmentQueueSegmentReader {
    segments: Segments,
    notifier: Mutex<mpsc::UnboundedSender<()>>,
}

impl AsyncSegmentReader for SegmentQueueSegmentReader {
    fn get(&self, id: SegmentId) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let mut segments = self.segments.write().vortex_expect("poisoned lock");

        let future = match segments
            .iter()
            .filter_map(|p| p.upgrade())
            .find(|p| p.id() == id)
        {
            None => {
                let pending = PendingSegment::new(id);
                // Insert the pending segment into the map, return the strong shared future to the caller.
                segments.push_back(Arc::downgrade(&pending));
                pending.clone().new_future().boxed()
            }
            Some(pending) => pending.new_future().boxed(),
        };

        // Send a notification that there may be more work to do in the queue.
        self.notifier
            .lock()
            .vortex_expect("poisoned lock")
            .unbounded_send(())
            .map_err(|e| vortex_err!("Failed to notify segment queue {}", e))
            .vortex_expect("Failed to notify segment queue");

        future
    }
}
