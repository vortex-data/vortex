use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, ready};

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
    inner: Mutex<PendingSegmentInner>,
}

struct PendingSegmentInner {
    /// The sender end of the one-shot channel can be taken by leasing the segment.
    /// If the lease is dropped before resolving, the sender is put back into this field to allow
    /// another lease.
    send: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,
    recv: BoxFuture<'static, VortexResult<ByteBuffer>>,
    result: Option<SharedVortexResult<ByteBuffer>>,
    on_poll: Option<Box<dyn FnOnce() + Send + 'static>>,
    on_drop: Option<Box<dyn FnOnce() + Send + 'static>>,
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
    pub fn new<P, D>(id: SegmentId, on_poll: P, on_drop: D) -> Arc<PendingSegment>
    where
        P: FnOnce() + Send + 'static,
        D: FnOnce() + Send + 'static,
    {
        log::debug!("Pending segment {}: REGISTERED", id);
        let (send, recv) = oneshot::channel();

        Arc::new(Self {
            id,
            inner: Mutex::new(PendingSegmentInner {
                send: Some(send),
                recv: recv
                    .map_err(|e| vortex_err!("pending segment sender dropped: {}", e))
                    .map(|r| r.unnest())
                    .boxed(),
                result: None,
                on_poll: Some(Box::new(on_poll)),
                on_drop: Some(Box::new(on_drop)),
            }),
        })
    }

    pub fn id(&self) -> SegmentId {
        self.id
    }

    pub fn new_future(self: Arc<Self>) -> PendingSegmentFuture {
        PendingSegmentFuture { pending: self }
    }

    /// Take a unique lease on the pending segment to resolve it some time later.
    pub fn lease(self: Arc<Self>) -> Option<PendingSegmentLease> {
        self.inner
            .lock()
            .vortex_expect("poisoned lock")
            .send
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
        let mut inner = self.inner.lock().vortex_expect("poisoned lock");
        if inner.result.is_none() && inner.send.is_some() {
            log::debug!("Pending segment {}: DROPPED BEFORE LAUNCH", self.id);
        }
        if inner.result.is_none() && inner.send.is_none() {
            log::debug!("Pending segment {}: DROPPED IN FLIGHT", self.id);
        }
        if let Some(f) = inner.on_drop.take() {
            f();
        }
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
        let mut inner = self.pending.inner.lock().vortex_expect("poisoned lock");

        // Continue to return the same result if it is already resolved.
        if let Some(result) = &inner.result {
            return Poll::Ready(result.clone().map_err(VortexError::from));
        }

        // Trigger the on-poll callback if it exists.
        if let Some(on_poll) = inner.on_poll.take() {
            on_poll();
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
                pending
                    .inner
                    .lock()
                    .vortex_expect("poisoned lock")
                    .send
                    .replace(send);
            }
        }
    }
}
