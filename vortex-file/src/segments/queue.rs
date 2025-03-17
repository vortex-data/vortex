use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
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

type SegmentsMap = Arc<RwLock<BTreeMap<SegmentId, PendingSegment>>>;

/// Pre-fetch queue for segments for the generic file reader.
pub struct SegmentQueue {
    segments: SegmentsMap,
    send: mpsc::UnboundedSender<()>,
    recv: mpsc::UnboundedReceiver<()>,
}

impl SegmentQueue {
    pub fn new() -> Self {
        let (send, recv) = mpsc::unbounded();
        Self {
            segments: Arc::new(RwLock::new(BTreeMap::new())),
            send,
            recv,
        }
    }

    pub fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        Arc::new(SegmentQueueSegmentReader {
            segments: self.segments.clone(),
            notifier: Mutex::new(self.send.clone()),
        })
    }

    pub async fn io_driver(mut self) -> VortexResult<()> {
        while let Some(_) = self.recv.next().await {
            // We've been notified that there is I/O work to do
            println!("I/O work to do");
        }
        Ok(())
    }
}

struct SegmentQueueSegmentReader {
    segments: SegmentsMap,
    notifier: Mutex<mpsc::UnboundedSender<()>>,
}

impl AsyncSegmentReader for SegmentQueueSegmentReader {
    fn get(&self, id: SegmentId) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        match self
            .segments
            .write()
            .vortex_expect("poisoned lock")
            .entry(id)
        {
            Entry::Vacant(e) => {
                let (pending, fut) = PendingSegment::new(id, self.segments.clone());
                // Insert the pending segment into the map, return the strong shared future to the caller.
                e.insert(pending);
                self.notifier
                    .lock()
                    .vortex_expect("poisoned lock")
                    .unbounded_send(())
                    .map_err(|e| vortex_err!("Failed to notify segment queue {}", e))
                    .vortex_expect("Failed to notify segment queue");
                Box::pin(fut.map_err(VortexError::from))
            }
            Entry::Occupied(e) => {
                // If the entry is occupied, and the pending segment
                if let Some(fut) = e.get().recv.upgrade() {
                    Box::pin(fut.map_err(VortexError::from))
                } else {
                    vortex_panic!("Segment lost all strong refs without cleaning itself up");
                }
            }
        }
    }
}

pub struct PendingSegment {
    id: SegmentId,
    send: oneshot::Sender<VortexResult<ByteBuffer>>,
    recv: WeakShared<BoxFuture<'static, SharedVortexResult<ByteBuffer>>>,
    segments: SegmentsMap,
}

impl PendingSegment {
    fn new(
        segment_id: SegmentId,
        segments: SegmentsMap,
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
            send,
            recv: shared_recv.downgrade().vortex_expect("Just created"),
            segments,
        };

        (pending, shared_recv)
    }
}

impl Drop for PendingSegment {
    fn drop(&mut self) {
        // When a pending segment is dropped, we clean it up and remove it from the map.
        log::debug!("Dropping segment {:?}", self.id);
        self.segments
            .write()
            .vortex_expect("poisoned lock")
            .remove(&self.id);
    }
}
