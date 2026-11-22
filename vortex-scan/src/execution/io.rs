// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
use futures::future::AbortHandle;
use futures::future::Aborted;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::abortable;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_utils::aliases::hash_map::HashMap;

use crate::execution::SegmentLocation;
use crate::source::SegmentFuture;
use crate::source::SegmentId;
use crate::source::SegmentSource;

/// Byte ranges of a file's segments, keyed by `(offset, length)`, back to their segment ids.
pub(crate) type SegmentRanges = Arc<HashMap<(u64, usize), SegmentId>>;

pub(crate) fn segment_ranges(locations: &[SegmentLocation]) -> SegmentRanges {
    let ranges = locations.iter().enumerate().map(|(id, location)| {
        let id = SegmentId::from(u32::try_from(id).unwrap_or(u32::MAX));
        ((location.offset, location.length as usize), id)
    });
    Arc::new(ranges.collect())
}

/// Serves splits from any [`SegmentSource`], by segment id.
pub(crate) struct SegmentScanIo {
    segments: Arc<dyn SegmentSource>,
    ranges: SegmentRanges,
}

impl SegmentScanIo {
    pub(crate) fn new(segments: Arc<dyn SegmentSource>, ranges: SegmentRanges) -> Self {
        Self { segments, ranges }
    }
}

impl IoService for SegmentScanIo {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(SegmentIoSource::new(
            Arc::clone(&self.segments),
            Arc::clone(&self.ranges),
        ))
    }
}

/// Serves a split's range requests from a segment source, without blocking.
///
/// Each requested range names one segment. [`submit`](IoSource::submit) starts its read at once,
/// [`poll`](IoSource::poll) takes a finished one if there is any, and the split's future awaits
/// the next with [`poll_completion`](IoSource::poll_completion) when the driver has nothing else
/// to run. [`wait`](IoSource::wait) is never called.
///
/// An announcement requests the segment and holds the request, never polled, until a prefetch or
/// fetch consumes it, so a source that coalesces requests can fold it into a nearby read.
pub(super) struct SegmentIoSource {
    segments: Arc<dyn SegmentSource>,
    ranges: SegmentRanges,
    reads: Mutex<FuturesUnordered<PendingRead>>,
    ready: Mutex<VecDeque<Completion>>,
    /// Active in-flight `Fetch` subscriptions per owner so `release(owner)` can cancel them.
    owners: Mutex<HashMap<IoOwnerId, HashMap<u64, AbortHandle>>>,
    next_subscription: AtomicU64,
    /// Every read the session has started, by segment, shared by later fetches of the segment
    /// until the session is cleared.
    prefetched: Mutex<HashMap<SegmentId, Prefetched>>,
    /// Announced segments, requested and never polled.
    announced: Mutex<HashMap<SegmentId, VecDeque<SegmentFuture>>>,
}

/// An in-flight fetch subscription that can be cancelled when its owner retires.
type PendingRead = BoxFuture<'static, Result<(u64, Completion), Aborted>>;

/// A read started ahead of the fetch that needs it.
type Prefetched = Shared<BoxFuture<'static, Result<BufferHandle, Arc<VortexError>>>>;

impl SegmentIoSource {
    pub(super) fn new(segments: Arc<dyn SegmentSource>, ranges: SegmentRanges) -> Self {
        Self {
            segments,
            ranges,
            reads: Mutex::new(FuturesUnordered::new()),
            ready: Mutex::default(),
            owners: Mutex::default(),
            next_subscription: AtomicU64::new(0),
            prefetched: Mutex::default(),
            announced: Mutex::default(),
        }
    }

    fn finish_subscription(&self, owner: IoOwnerId, subscription: u64) {
        let mut owners = self.owners.lock();
        if let Some(subs) = owners.get_mut(&owner) {
            subs.remove(&subscription);
            if subs.is_empty() {
                owners.remove(&owner);
            }
        }
    }
}

impl IoSource for SegmentIoSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let reads = self.reads.lock();
        let mut ready = self.ready.lock();
        let mut owners = self.owners.lock();
        let mut prefetched = self.prefetched.lock();
        for request in batch {
            let IoTarget::Range { offset, len, .. } = request.target else {
                vortex_bail!(
                    "SegmentIoSource only serves byte ranges, not {:?}",
                    request.target
                );
            };
            let segment = *self
                .ranges
                .get(&(offset, len))
                .ok_or_else(|| vortex_err!("No segment at bytes {offset}+{len}"))?;
            if request.intent == IoIntent::Announce {
                self.announced
                    .lock()
                    .entry(segment)
                    .or_default()
                    .push_back(self.segments.request(segment));
                continue;
            }
            if request.intent == IoIntent::Prefetch {
                prefetched.entry(segment).or_insert_with(|| {
                    let read = self
                        .announced
                        .lock()
                        .get_mut(&segment)
                        .and_then(VecDeque::pop_front)
                        .unwrap_or_else(|| self.segments.request(segment))
                        .map_err(Arc::new)
                        .boxed()
                        .shared();
                    // Polling registers the read as wanted, so the source starts it now.
                    drop(
                        read.clone()
                            .poll_unpin(&mut Context::from_waker(Waker::noop())),
                    );
                    read
                });
                continue;
            }
            // A segment stays with the session once read, so a later stage or morsel of the
            // split that reads it again shares the read instead of starting another once the
            // first has completed.
            let mut bytes = prefetched
                .entry(segment)
                .or_insert_with(|| {
                    self.announced
                        .lock()
                        .get_mut(&segment)
                        .and_then(VecDeque::pop_front)
                        .unwrap_or_else(|| self.segments.request(segment))
                        .map_err(Arc::new)
                        .boxed()
                        .shared()
                })
                .clone()
                .map_err(move |err| vortex_err!("read of segment {segment} failed: {err}"))
                .boxed();
            let id = request.request;
            // In-memory and cached segments need no allocated asynchronous task. A pending
            // read is polled again by FuturesUnordered before the driver can go to sleep.
            if let Poll::Ready(result) = bytes.poll_unpin(&mut Context::from_waker(Waker::noop())) {
                ready.push_back(Completion {
                    owner,
                    request: id,
                    result: result.map(IoResult::Bytes),
                });
                continue;
            }
            let subscription = self.next_subscription.fetch_add(1, Ordering::Relaxed);
            let (fut, abort_handle) = abortable(async move {
                (
                    subscription,
                    Completion {
                        owner,
                        request: id,
                        result: bytes.await.map(IoResult::Bytes),
                    },
                )
            });
            owners
                .entry(owner)
                .or_default()
                .insert(subscription, abort_handle);
            reads.push(fut.boxed());
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        if let Some(completion) = self.ready.lock().pop_front() {
            return Ok(Some(completion));
        }
        // The split's future polls again with its own waker before it waits, so a read that
        // finishes after this check is not missed.
        let mut cx = Context::from_waker(Waker::noop());
        let mut reads = self.reads.lock();
        loop {
            match reads.poll_next_unpin(&mut cx) {
                Poll::Ready(Some(Ok((subscription, completion)))) => {
                    self.finish_subscription(completion.owner, subscription);
                    return Ok(Some(completion));
                }
                Poll::Ready(Some(Err(_aborted))) => continue,
                Poll::Ready(None) | Poll::Pending => return Ok(None),
            }
        }
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        if let Some(completion) = self.ready.lock().pop_front() {
            return Poll::Ready(Ok(completion));
        }
        let mut reads = self.reads.lock();
        loop {
            match reads.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok((subscription, completion)))) => {
                    self.finish_subscription(completion.owner, subscription);
                    return Poll::Ready(Ok(completion));
                }
                Poll::Ready(Some(Err(_aborted))) => continue,
                Poll::Ready(None) => {
                    return Poll::Ready(Err(vortex_err!(
                        "The split's driver is waiting with no reads in flight"
                    )));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("SegmentIoSource is awaited through poll_completion, not waited on")
    }

    fn release(&self, owner: IoOwnerId) {
        self.ready
            .lock()
            .retain(|completion| completion.owner != owner);
        if let Some(handles) = self.owners.lock().remove(&owner) {
            for handle in handles.into_values() {
                handle.abort();
            }
        }
    }

    fn clear(&self) {
        self.reads.lock().clear();
        self.ready.lock().clear();
        self.owners.lock().clear();
        self.prefetched.lock().clear();
        self.announced.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use futures::channel::oneshot;
    use futures::future;
    use parking_lot::Mutex;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_io::request::IoIntent;
    use vortex_io::request::IoOwnerId;
    use vortex_io::request::IoRequest;
    use vortex_io::request::IoRequestId;
    use vortex_io::request::IoSource;
    use vortex_io::request::IoTarget;

    use super::SegmentIoSource;
    use super::segment_ranges;
    use crate::execution::SegmentLocation;
    use crate::source::SegmentFuture;
    use crate::source::SegmentId;
    use crate::source::SegmentSource;

    /// Serves four bytes per segment and counts the reads it is asked for.
    #[derive(Default)]
    struct CountingSegments(AtomicUsize);

    impl SegmentSource for CountingSegments {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.0.fetch_add(1, Ordering::Relaxed);
            let bytes = ByteBuffer::from(vec![u8::try_from(*id).unwrap_or(u8::MAX); 4]);
            future::ready(Ok(BufferHandle::new_host(bytes))).boxed()
        }
    }

    fn request(intent: IoIntent, id: u32) -> IoRequest {
        IoRequest {
            intent,
            request: IoRequestId(id),
            target: IoTarget::range(0, 4),
        }
    }

    struct MixedSegments(Mutex<Option<oneshot::Receiver<ByteBuffer>>>);

    impl SegmentSource for MixedSegments {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            if *id == 0 {
                return future::ready(Ok(BufferHandle::new_host(ByteBuffer::from(vec![0; 4]))))
                    .boxed();
            }
            let receiver = self.0.lock().take().expect("one pending read");
            async move {
                receiver
                    .await
                    .map(BufferHandle::new_host)
                    .map_err(|err| vortex_err!("{err}"))
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn pending_read_wakes_after_an_immediate_completion() -> VortexResult<()> {
        let (sender, receiver) = oneshot::channel();
        let segments = Arc::new(MixedSegments(Mutex::new(Some(receiver))));
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(segments, segment_ranges(&locations));
        let mut pending = request(IoIntent::Fetch, 1);
        pending.target = IoTarget::range(4, 4);
        io.submit(IoOwnerId(0), vec![pending, request(IoIntent::Fetch, 0)])?;
        let immediate = io.poll()?.expect("the in-memory read is already ready");
        assert_eq!(immediate.request, IoRequestId(0));
        assert!(immediate.result.is_ok());
        assert!(io.poll()?.is_none());

        let delivery = tokio::spawn(async move {
            tokio::task::yield_now().await;
            sender
                .send(ByteBuffer::from(vec![1; 4]))
                .expect("read is live");
        });
        let completed = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completed.request, IoRequestId(1));
        assert!(completed.result.is_ok());
        delivery.await.expect("delivery task succeeds");
        assert!(io.poll()?.is_none());
        Ok(())
    }

    /// A fetch of a prefetched segment waits on the prefetch's read rather than reading again,
    /// only the fetch is delivered, and a later fetch of the segment shares the same read.
    #[tokio::test]
    async fn fetch_reuses_a_prefetched_read() -> VortexResult<()> {
        let segments = Arc::new(CountingSegments::default());
        let location = SegmentLocation {
            offset: 0,
            length: 4,
            alignment: Alignment::none(),
        };
        let io = SegmentIoSource::new(Arc::clone(&segments) as _, segment_ranges(&[location]));

        io.submit(IoOwnerId(0), vec![request(IoIntent::Prefetch, 0)])?;
        assert_eq!(segments.0.load(Ordering::Relaxed), 1);
        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 1)])?;
        assert_eq!(segments.0.load(Ordering::Relaxed), 1);

        let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completion.request, IoRequestId(1));
        assert!(completion.result.is_ok());
        assert!(io.poll()?.is_none());

        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 2)])?;
        let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completion.request, IoRequestId(2));
        assert!(completion.result.is_ok());
        assert_eq!(segments.0.load(Ordering::Relaxed), 1);

        io.clear();
        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 3)])?;
        assert!(
            std::future::poll_fn(|cx| io.poll_completion(cx))
                .await?
                .result
                .is_ok()
        );
        assert_eq!(segments.0.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[tokio::test]
    async fn announcements_keep_independent_consumers() -> VortexResult<()> {
        let segments = Arc::new(CountingSegments::default());
        let location = SegmentLocation {
            offset: 0,
            length: 4,
            alignment: Alignment::none(),
        };
        let io = SegmentIoSource::new(Arc::clone(&segments) as _, segment_ranges(&[location]));
        io.submit(
            IoOwnerId(0),
            vec![
                request(IoIntent::Announce, 0),
                request(IoIntent::Announce, 1),
            ],
        )?;
        assert_eq!(segments.0.load(Ordering::Relaxed), 2);
        for id in 2..5 {
            io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, id)])?;
            assert!(
                std::future::poll_fn(|cx| io.poll_completion(cx))
                    .await?
                    .result
                    .is_ok()
            );
            // The first fetch takes one announcement and the session keeps its read, so the
            // later fetches share it and the other announcement is never consumed.
            assert_eq!(segments.0.load(Ordering::Relaxed), 2);
        }
        Ok(())
    }

    #[tokio::test]
    async fn release_discards_ready_and_pending_fetches() -> VortexResult<()> {
        let (sender, receiver) = oneshot::channel();
        let segments = Arc::new(MixedSegments(Mutex::new(Some(receiver))));
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(segments, segment_ranges(&locations));

        // Owner 0 gets an immediate completion (queued in `ready`) and a pending read (in `reads`).
        let mut pending = request(IoIntent::Fetch, 1);
        pending.target = IoTarget::range(4, 4);
        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0), pending])?;
        // Owner 1 subscribes to the same pending segment.
        let mut live_pending = request(IoIntent::Fetch, 2);
        live_pending.target = IoTarget::range(4, 4);
        io.submit(IoOwnerId(1), vec![live_pending])?;

        io.release(IoOwnerId(0));

        // Owner 0's queued ready completion must not be delivered.
        assert!(io.poll()?.is_none());

        sender
            .send(ByteBuffer::from(vec![9; 4]))
            .expect("shared pending read is still live for owner 1");

        let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completion.owner, IoOwnerId(1));
        assert_eq!(completion.request, IoRequestId(2));
        assert!(completion.result.is_ok());
        assert!(io.poll()?.is_none());
        assert!(io.owners.lock().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn release_only_owner_leaves_no_reads_in_flight() -> VortexResult<()> {
        let (sender, receiver) = oneshot::channel();
        let segments = Arc::new(MixedSegments(Mutex::new(Some(receiver))));
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(segments, segment_ranges(&locations));

        let mut pending = request(IoIntent::Fetch, 0);
        pending.target = IoTarget::range(4, 4);
        io.submit(IoOwnerId(0), vec![pending])?;
        assert_eq!(io.owners.lock().len(), 1);

        io.release(IoOwnerId(0));
        assert!(io.owners.lock().is_empty());

        // Even if the underlying shared read completes, the released subscriber is discarded.
        drop(sender.send(ByteBuffer::from(vec![7; 4])));
        assert!(io.poll()?.is_none());
        let err = std::future::poll_fn(|cx| io.poll_completion(cx))
            .await
            .err()
            .expect("no live reads should remain");
        assert!(err.to_string().contains("no reads in flight"));
        Ok(())
    }

    #[tokio::test]
    async fn release_preserves_optional_announcements_and_prefetches() -> VortexResult<()> {
        let segments = Arc::new(CountingSegments::default());
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(Arc::clone(&segments) as _, segment_ranges(&locations));

        let mut prefetch = request(IoIntent::Prefetch, 1);
        prefetch.target = IoTarget::range(4, 4);
        io.submit(IoOwnerId(0), vec![request(IoIntent::Announce, 0), prefetch])?;
        assert_eq!(segments.0.load(Ordering::Relaxed), 2);

        io.release(IoOwnerId(0));

        let mut fetch_prefetched = request(IoIntent::Fetch, 3);
        fetch_prefetched.target = IoTarget::range(4, 4);
        io.submit(
            IoOwnerId(1),
            vec![request(IoIntent::Fetch, 2), fetch_prefetched],
        )?;
        // Both the announcement and the prefetch survived owner 0's release.
        assert_eq!(segments.0.load(Ordering::Relaxed), 2);

        let first = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        let second = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(first.owner, IoOwnerId(1));
        assert_eq!(second.owner, IoOwnerId(1));
        assert!(first.result.is_ok());
        assert!(second.result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn release_preserves_live_failures() -> VortexResult<()> {
        let (sender, receiver) = oneshot::channel::<ByteBuffer>();
        let segments = Arc::new(MixedSegments(Mutex::new(Some(receiver))));
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(segments, segment_ranges(&locations));

        let mut fetch_a = request(IoIntent::Fetch, 0);
        fetch_a.target = IoTarget::range(4, 4);
        let mut fetch_b = request(IoIntent::Fetch, 1);
        fetch_b.target = IoTarget::range(4, 4);
        io.submit(IoOwnerId(0), vec![fetch_a])?;
        io.submit(IoOwnerId(1), vec![fetch_b])?;

        io.release(IoOwnerId(0));
        // Dropping the sender causes the underlying read to fail.
        drop(sender);

        let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
        assert_eq!(completion.owner, IoOwnerId(1));
        assert_eq!(completion.request, IoRequestId(1));
        assert!(completion.result.is_err());
        assert!(io.poll()?.is_none());
        assert!(io.owners.lock().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn clear_drops_pending_owner_subscriptions() -> VortexResult<()> {
        let (_sender, receiver) = oneshot::channel::<ByteBuffer>();
        let segments = Arc::new(MixedSegments(Mutex::new(Some(receiver))));
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(segments, segment_ranges(&locations));

        let mut fetch = request(IoIntent::Fetch, 0);
        fetch.target = IoTarget::range(4, 4);
        io.submit(IoOwnerId(0), vec![fetch])?;
        assert_eq!(io.owners.lock().len(), 1);

        io.clear();
        assert!(io.owners.lock().is_empty());
        assert!(io.poll()?.is_none());
        let err = match std::future::poll_fn(|cx| io.poll_completion(cx)).await {
            Ok(_) => {
                return Err(vortex_err!(
                    "clearing pending subscriptions should leave no reads in flight"
                ));
            }
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("no reads in flight"),
            "unexpected error: {err}"
        );
        Ok(())
    }
}
