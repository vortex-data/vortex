use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, ready};

use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{
    ResultExt, SharedVortexResult, VortexError, VortexExpect, VortexResult, vortex_err,
};

use crate::segments::SegmentId;

/// A pending segment returned by the [`AsyncSegmentReader`].
pub struct PendingSegment {
    id: SegmentId,
    /// The sender end of the one-shot channel can be taken by leasing the segment.
    /// If the lease is dropped before resolving, the sender is put back into this field to allow
    /// another lease.
    send: Mutex<Option<oneshot::Sender<VortexResult<ByteBuffer>>>>,
    state: Mutex<PendingSegmentState>,
}

impl Debug for PendingSegment {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSegment")
            .field("id", &self.id)
            .field("state", &self.state)
            .finish()
    }
}

impl PendingSegment {
    /// Create a new [`PendingSegment`] that can be resolved later.
    pub fn new(id: SegmentId) -> Arc<PendingSegment> {
        log::debug!("Pending segment {}: REGISTERED", id);
        let (send, recv) = oneshot::channel();

        Arc::new(Self {
            id,
            send: Mutex::new(Some(send)),
            state: Mutex::new(PendingSegmentState::Prefetched(
                recv.map_err(|e| vortex_err!("Failed to receive segment: {}", e))
                    .map(|r| r.unnest())
                    .boxed(),
            )),
        })
    }

    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// Create a new shared future that resolved to the segment buffer.
    pub fn new_future(self: Arc<Self>) -> impl Future<Output = VortexResult<ByteBuffer>> {
        PendingSegmentFuture { pending: self }
    }

    /// Take a unique lease on the pending segment to resolve it some time later.
    pub fn lease(self: Arc<Self>) -> Option<PendingSegmentLease> {
        self.send
            .lock()
            .vortex_expect("poisoned lock")
            .take()
            .map(|send| PendingSegmentLease {
                id: self.id,
                pending: Arc::downgrade(&self),
                send: Some(send),
            })
    }
}

impl Drop for PendingSegment {
    fn drop(&mut self) {
        match *self.state.lock().vortex_expect("poisoned lock") {
            PendingSegmentState::Prefetched(_) => {
                log::debug!("Pending segment {}: DROPPED", self.id);
            }
            PendingSegmentState::Resolved(_) => {
                // Not interesting if a segment is dropped after being resolved
            }
        }
    }
}

/// The state of a pending segment.
enum PendingSegmentState {
    /// The segment has been prefetched but not yet explicitly requested (polled).
    Prefetched(BoxFuture<'static, VortexResult<ByteBuffer>>),
    /// The segment has been resolved and the buffer is available.
    Resolved(SharedVortexResult<ByteBuffer>),
}

impl Debug for PendingSegmentState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSegmentState")
            .field(
                "state",
                &match self {
                    PendingSegmentState::Prefetched(_) => "Prefetched",
                    PendingSegmentState::Resolved(_) => "Resolved",
                },
            )
            .finish()
    }
}

/// A future that resolves when a pending segment is resolved.
///
/// It supports being polled multiple times, and will return the same result.
pub struct PendingSegmentFuture {
    pending: Arc<PendingSegment>,
}

impl Future for PendingSegmentFuture {
    type Output = VortexResult<ByteBuffer>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.pending.state.lock().vortex_expect("poisoned lock");
        match &mut *state {
            PendingSegmentState::Prefetched(fut) => {
                let result = ready!(fut.poll_unpin(cx)).map_err(Arc::new);
                *state = PendingSegmentState::Resolved(result.clone());
                Poll::Ready(result.map_err(VortexError::from))
            }
            PendingSegmentState::Resolved(result) => {
                Poll::Ready(result.clone().map_err(VortexError::from))
            }
        }
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
            log::debug!("Pending segment {}: DROPPED WHILE LEASED", self.id);
        }
    }
}

impl Debug for PendingSegmentLease {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingSegmentLease")
            .field("id", &self.id)
            .finish()
    }
}

impl Drop for PendingSegmentLease {
    fn drop(&mut self) {
        // If the lease is dropped without resolving, we put the send channel back into
        // the pending segment.
        if let Some(send) = self.send.take() {
            if let Some(pending) = self.pending.upgrade() {
                *pending.send.lock().vortex_expect("poisoned lock") = Some(send);
            }
        }
    }
}
