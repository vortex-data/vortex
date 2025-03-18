use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

use futures::channel::{mpsc, oneshot};
use futures::future::{BoxFuture, Shared, WeakShared};
use futures::{FutureExt, StreamExt, TryFutureExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
    vortex_panic,
};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

type Segments = Arc<RwLock<VecDeque<PendingSegment>>>;

/// Pre-fetch queue for segments for the generic file reader.
pub struct SegmentQueue {
    segments: Segments,
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

    pub fn has_pending(&self) -> bool {
        self.with_pending_segments(|pending| pending.next().is_some())
    }

    /// Inspect all pending segments, in order of segment ID.
    /// TODO(ngates): we want this in order of request, not segment ID.
    pub fn with_pending_segments<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut dyn Iterator<Item = &mut PendingSegment>) -> T,
    {
        f(&mut self
            .segments
            .write()
            .vortex_expect("poisoned lock")
            .iter_mut()
            .filter(|p| p.send.is_some()))
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

        match segments.iter().find(|p| p.id == id) {
            None => {
                let (pending, fut) = PendingSegment::new(id, self.segments.clone());
                // Insert the pending segment into the map, return the strong shared future to the caller.
                segments.push_back(pending);
                self.notifier
                    .lock()
                    .vortex_expect("poisoned lock")
                    .unbounded_send(())
                    .map_err(|e| vortex_err!("Failed to notify segment queue {}", e))
                    .vortex_expect("Failed to notify segment queue");
                Box::pin(fut.map_err(VortexError::from))
            }
            Some(pending) => {
                // If the entry is occupied, return a clone of the future
                if let Some(fut) = pending.recv.upgrade() {
                    Box::pin(fut.map_err(VortexError::from))
                } else {
                    vortex_panic!("Segment lost all strong refs without cleaning itself up");
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct PendingSegment {
    id: SegmentId,
    /// The sender half of the channel to resolve the buffer once it has been read.
    /// If the option is empty, it means it is in-flight by a current request.
    send: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,
    recv: WeakShared<BoxFuture<'static, SharedVortexResult<ByteBuffer>>>,
    segments: Segments,
}

impl PendingSegment {
    fn new(
        segment_id: SegmentId,
        segments: Segments,
    ) -> (
        Self,
        Shared<BoxFuture<'static, SharedVortexResult<ByteBuffer>>>,
    ) {
        let (send, recv) = oneshot::channel();
        let shared_recv = recv
            .map_err(|e| vortex_err!("Failed to receive segment {}", e))
            .map(|r| r.unnest().map_err(Arc::new))
            .boxed()
            .shared();

        let pending = Self {
            id: segment_id,
            send: Some(send),
            recv: shared_recv.downgrade().vortex_expect("Just created"),
            segments,
        };

        (pending, shared_recv)
    }

    pub fn id(&self) -> SegmentId {
        self.id
    }

    pub fn take_callback(&mut self) -> Option<oneshot::Sender<VortexResult<ByteBuffer>>> {
        self.send.take()
    }
}

impl Drop for PendingSegment {
    fn drop(&mut self) {
        // When a pending segment is dropped, we clean it up and remove it from the map.
        if self.send.is_some() {
            log::info!(
                "DROP DROP DROP segment request {:?} prior to launch",
                self.id
            );
        }
        let mut segments = self.segments.write().vortex_expect("poisoned lock");
        let idx = segments
            .iter()
            .position(|p| p.id == self.id)
            .vortex_expect("Segment not found");
        segments.remove(idx);
    }
}
