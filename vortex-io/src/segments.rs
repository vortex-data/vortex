// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads for scans that drive their own IO.
//!
//! A request names what to read by [`IoKey`], a registered segment source and a
//! [`SourceRequest`] of it (one of its segments, or its size), and says how much it needs it with
//! an [`IoIntent`]. One service serves many sources, so one session can read from all of them.
//! A [`SegmentIoSession`] registers a whole batch of them before any can be read, and returns one
//! [`IoHandle`] per request. The handle is the request: its result arrives in it. The session holds fetches and prefetches until they finish; an announcement is
//! withdrawn by dropping every handle of it.
//!
//! A [`SegmentIoService`] serves every registered source. Its sessions share and coalesce reads,
//! so one session's announcement can widen another session's read.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::FuturesUnordered;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

/// A segment source registered with a [`SegmentIoService`], such as one opened file.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentSourceId(pub u32);

/// What to read from a segment source.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceRequest {
    /// The source's total length in bytes, answered with [`SourceResult::Size`].
    Size,
    /// One segment, by its id within the source (the value of a
    /// `vortex_layout::segments::SegmentId`), answered with [`SourceResult::Bytes`].
    Segment(u32),
}

/// What one request reads: a source, and what to read from it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IoKey {
    /// The source to read.
    pub source: SegmentSourceId,
    /// What to read from it.
    pub request: SourceRequest,
}

impl IoKey {
    /// The key of `request` of `source`.
    pub fn new(source: SegmentSourceId, request: SourceRequest) -> Self {
        Self { source, request }
    }

    /// The key of `segment` of `source`.
    pub fn segment(source: SegmentSourceId, segment: u32) -> Self {
        Self::new(source, SourceRequest::Segment(segment))
    }

    /// The key of the size of `source`.
    pub fn size(source: SegmentSourceId) -> Self {
        Self::new(source, SourceRequest::Size)
    }
}

/// The answer to one request.
#[derive(Clone, Debug)]
pub enum SourceResult {
    /// The source's total length in bytes.
    Size(u64),
    /// A segment's bytes.
    Bytes(BufferHandle),
}

impl SourceResult {
    /// The bytes of a segment read, failing for any other answer.
    pub fn into_bytes(self) -> VortexResult<BufferHandle> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::Size(_) => vortex_bail!("expected segment bytes, got a size"),
        }
    }

    /// The size a size request was answered with, failing for any other answer.
    pub fn size(&self) -> VortexResult<u64> {
        match self {
            Self::Size(size) => Ok(*size),
            Self::Bytes(_) => vortex_bail!("expected a size, got segment bytes"),
        }
    }
}

/// How much a request needs what it reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IoIntent {
    /// Needed now. Its owner is reported by [`SegmentIoSession::poll_ready`] when it finishes.
    /// A service may read fetches ahead of prefetches.
    Fetch,
    /// Needed soon: read, but its owner is not reported when it finishes.
    Prefetch,
    /// May be needed: only widens nearby reads, and is never read on its own. Dropping its
    /// handles withdraws it.
    Announce,
}

/// One read a planner or morsel asks for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IoRequest {
    /// What to read.
    pub key: IoKey,
    /// How much it is needed.
    pub intent: IoIntent,
}

impl IoRequest {
    /// A request for `key` with `intent`.
    pub fn new(key: IoKey, intent: IoIntent) -> Self {
        Self { key, intent }
    }
}

/// Identifies the object that submitted a request, chosen by the caller.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IoOwnerId(pub u64);

/// The result of one read, shared by every handle of it.
pub type SharedResult = Shared<BoxFuture<'static, Result<SourceResult, Arc<VortexError>>>>;

/// One registered read.
///
/// Clones share the read. The read stays registered while any handle of it, from any session, is
/// alive. A session also holds its fetches and prefetches until they finish, so dropping every
/// handle withdraws a read only when it was announced alone.
#[derive(Clone)]
pub struct IoHandle {
    result: SharedResult,
    /// Keeps the read registered with its service while the handle lives.
    _registration: Option<Arc<dyn Any + Send + Sync>>,
}

impl IoHandle {
    /// A handle whose result arrives through `result`, keeping `registration` alive while it
    /// lives.
    pub fn new(result: SharedResult, registration: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            result,
            _registration: Some(registration),
        }
    }

    /// A handle that already holds its result, such as a cache hit or a source's size.
    pub fn ready(result: VortexResult<SourceResult>) -> Self {
        let result = futures::future::ready(result.map_err(Arc::new))
            .boxed()
            .shared();
        // `Shared::peek` only sees a result once a poll has completed the future, so poll it once
        // here for `try_get` to see it without a waker.
        drop(result.clone().now_or_never());
        Self {
            result,
            _registration: None,
        }
    }

    /// The read's result once it has arrived. Never blocks and never reads.
    pub fn try_get(&self) -> Option<VortexResult<SourceResult>> {
        self.result
            .peek()
            .map(|result| result.clone().map_err(VortexError::from))
    }

    /// The shared future the result arrives through, for sessions that wake on it.
    pub fn result(&self) -> SharedResult {
        self.result.clone()
    }
}

/// One scope's reads, such as the work descended from one root planner.
///
/// Registration does no IO and never blocks. Every read is driven by the service, so nothing has
/// to be polled for reads to progress; polling only learns which reads finished.
pub trait SegmentIoSession: Send + Sync {
    /// Registers the whole batch before any of it can be read, and returns one handle per
    /// request, in order. Every fetch is reported by [`poll_ready`](Self::poll_ready) once it
    /// finishes, including one already finished at submit, such as a cache hit or an error.
    fn submit(&self, owner: IoOwnerId, batch: &[IoRequest]) -> VortexResult<Vec<IoHandle>>;

    /// Moves the owners of fetches that finished since the last call into `ready`, once per
    /// fetch, and drives the session's prefetches. Returns `Pending`, registering `cx` to be
    /// woken by the next one, if none have, and fails if no fetch is in flight, since nothing
    /// would wake the caller. An owner is reported even if it dropped the fetch's handles.
    fn poll_ready(
        &self,
        cx: &mut Context<'_>,
        ready: &mut Vec<IoOwnerId>,
    ) -> Poll<VortexResult<()>>;
}

/// Serves the reads of every registered segment source.
pub trait SegmentIoService: Send + Sync {
    /// A new session for one scope.
    fn session(&self) -> Arc<dyn SegmentIoSession>;
}

/// A source registered with a [`SegmentIoService`]: the service, the source's id in it, and the
/// registration. The service can read the source while any clone of this is alive.
#[derive(Clone)]
pub struct RegisteredSource {
    service: Arc<dyn SegmentIoService>,
    id: SegmentSourceId,
    _registration: Arc<dyn Any + Send + Sync>,
}

impl RegisteredSource {
    /// The registration of `id` with `service`, kept alive by `registration`.
    pub fn new(
        service: Arc<dyn SegmentIoService>,
        id: SegmentSourceId,
        registration: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            service,
            id,
            _registration: registration,
        }
    }

    /// The service that reads the source.
    pub fn service(&self) -> &Arc<dyn SegmentIoService> {
        &self.service
    }

    /// The source's id in its service.
    pub fn id(&self) -> SegmentSourceId {
        self.id
    }
}

/// The fetches and prefetches of one session, waking the session when a fetch finishes.
///
/// Sessions keep one of these to implement [`SegmentIoSession::poll_ready`]. Prefetches are
/// polled until they finish, so their reads start and their results land in their handles even
/// though nobody awaits them.
#[derive(Default)]
pub struct FetchTracker {
    fetches: FuturesUnordered<Pin<Box<dyn Future<Output = IoOwnerId> + Send>>>,
    prefetches: FuturesUnordered<SharedResult>,
}

impl FetchTracker {
    /// Tracks a prefetch until it finishes, without reporting it.
    pub fn track_prefetch(&mut self, handle: &IoHandle) {
        self.prefetches.push(handle.result());
    }

    /// Tracks a fetch submitted by `owner`.
    pub fn track(&mut self, owner: IoOwnerId, handle: &IoHandle) {
        let bytes = handle.result();
        self.fetches.push(Box::pin(async move {
            // Errors are taken from the handle by its owner.
            drop(bytes.await);
            owner
        }));
    }

    /// Moves the owners of finished fetches into `ready`, as [`SegmentIoSession::poll_ready`].
    pub fn poll_ready(
        &mut self,
        cx: &mut Context<'_>,
        ready: &mut Vec<IoOwnerId>,
    ) -> Poll<VortexResult<()>> {
        // Finished prefetches are dropped; their handles hold the results.
        while let Poll::Ready(Some(_)) = self.prefetches.poll_next_unpin(cx) {}
        let before = ready.len();
        loop {
            match self.fetches.poll_next_unpin(cx) {
                Poll::Ready(Some(owner)) => ready.push(owner),
                Poll::Ready(None) if ready.len() > before => return Poll::Ready(Ok(())),
                Poll::Ready(None) => {
                    return Poll::Ready(Err(vortex_err!(
                        "waiting for segment reads with no fetch in flight"
                    )));
                }
                Poll::Pending if ready.len() > before => return Poll::Ready(Ok(())),
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    /// Drops every tracked fetch and prefetch.
    pub fn clear(&mut self) {
        self.fetches.clear();
        self.prefetches.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::task::Waker;

    use futures::channel::oneshot;
    use vortex_buffer::ByteBuffer;

    use super::*;

    fn pending() -> (oneshot::Sender<BufferHandle>, IoHandle) {
        let (send, recv) = oneshot::channel::<BufferHandle>();
        let result = recv
            .map(|result| {
                result
                    .map(SourceResult::Bytes)
                    .map_err(|_| Arc::new(vortex_err!("cancelled")))
            })
            .boxed()
            .shared();
        (send, IoHandle::new(result, Arc::new(())))
    }

    #[test]
    fn ready_handle_needs_no_poll() -> VortexResult<()> {
        let handle = IoHandle::ready(Ok(SourceResult::Bytes(BufferHandle::new_host(
            ByteBuffer::copy_from(b"abc"),
        ))));
        let result = handle.try_get().ok_or_else(|| vortex_err!("not ready"))??;
        assert_eq!(result.into_bytes()?.len(), 3);
        Ok(())
    }

    #[test]
    fn results_are_checked_against_what_was_asked() -> VortexResult<()> {
        assert_eq!(SourceResult::Size(7).size()?, 7);
        assert!(SourceResult::Size(7).into_bytes().is_err());
        let bytes = SourceResult::Bytes(BufferHandle::new_host(ByteBuffer::copy_from(b"a")));
        assert!(bytes.size().is_err());
        Ok(())
    }

    #[test]
    fn tracker_reports_owners_as_fetches_finish() -> VortexResult<()> {
        let mut tracker = FetchTracker::default();
        let (send_a, a) = pending();
        let (_send_b, b) = pending();
        tracker.track(IoOwnerId(1), &a);
        tracker.track(IoOwnerId(2), &b);
        let mut cx = Context::from_waker(Waker::noop());
        let mut ready = Vec::new();
        assert!(tracker.poll_ready(&mut cx, &mut ready).is_pending());
        assert!(a.try_get().is_none());

        send_a
            .send(BufferHandle::new_host(ByteBuffer::copy_from(b"a")))
            .map_err(|_| vortex_err!("receiver dropped"))?;
        assert!(matches!(
            tracker.poll_ready(&mut cx, &mut ready),
            Poll::Ready(Ok(()))
        ));
        assert_eq!(ready, vec![IoOwnerId(1)]);
        assert!(a.try_get().is_some());
        assert!(b.try_get().is_none());
        Ok(())
    }

    #[test]
    fn tracker_fails_with_nothing_in_flight() {
        let mut tracker = FetchTracker::default();
        let mut cx = Context::from_waker(Waker::noop());
        assert!(matches!(
            tracker.poll_ready(&mut cx, &mut Vec::new()),
            Poll::Ready(Err(_))
        ));
    }
}
