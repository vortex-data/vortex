use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::pin::Pin;
use std::sync::{Arc, RwLock, RwLockReadGuard, Weak};
use std::task::{Context, Poll};

use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
    vortex_panic,
};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

type SegmentsMap = Arc<RwLock<BTreeMap<SegmentId, Weak<PendingSegment>>>>;

/// Pre-fetch queue for segments for the generic file reader.
pub struct SegmentQueue {
    segments: SegmentsMap,
}

impl SegmentQueue {
    pub fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        Arc::new(Self {
            segments: self.segments.clone(),
        })
    }

    pub fn segments(&self) -> RwLockReadGuard<BTreeMap<SegmentId, Weak<PendingSegment>>> {
        self.segments.read().vortex_expect("poisoned lock")
    }
}

impl AsyncSegmentReader for SegmentQueue {
    fn get(&self, id: SegmentId) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        match self
            .segments
            .write()
            .vortex_expect("poisoned lock")
            .entry(id)
        {
            Entry::Vacant(e) => {
                let pending = Arc::new(PendingSegment::new(id, self.segments.clone()));
                // Insert a weak reference into the map, and return the strong one to the caller.
                e.insert(Arc::downgrade(&pending));
                PendingSegmentFuture(pending).boxed()
            }
            Entry::Occupied(e) => {
                // If the entry is occupied, and the pending segment
                if let Some(pending) = e.get().upgrade() {
                    PendingSegmentFuture(pending).boxed()
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
    recv: Shared<BoxFuture<'static, SharedVortexResult<ByteBuffer>>>,
    segments: SegmentsMap,
}

impl PendingSegment {
    fn new(segment_id: SegmentId, segments: SegmentsMap) -> Self {
        let (send, recv) = oneshot::channel();
        Self {
            id: segment_id,
            send,
            recv: recv
                .map_err(|e| vortex_err!("Failed to recieve segment"))
                .map(|r| r.unnest().map_err(Arc::new))
                .boxed()
                .shared(),
            segments,
        }
    }
}

impl Drop for PendingSegment {
    fn drop(&mut self) {
        /// When a pending segment is dropped, we clean it up and remove it from the map.
        log::debug!("Dropping segment {} {:?}", self.id);
        self.segments
            .write()
            .vortex_expect("poisoned lock")
            .remove(&self.id);
    }
}

/// Wrap up a pending segment in a future that the caller can poll.
struct PendingSegmentFuture(Arc<PendingSegment>);

impl Future for PendingSegmentFuture {
    type Output = VortexResult<ByteBuffer>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.recv.poll_unpin(cx).map_err(VortexError::from)
    }
}
