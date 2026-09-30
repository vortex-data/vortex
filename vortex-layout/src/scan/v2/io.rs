// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
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
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_utils::aliases::hash_map::HashMap;

use crate::scan::planning::SegmentLocation;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Byte ranges of a file's segments, keyed by `(offset, length)`, back to their segment ids.
pub(super) type SegmentRanges = Arc<HashMap<(u64, usize), SegmentId>>;

pub(super) fn segment_ranges(locations: &[SegmentLocation]) -> SegmentRanges {
    let ranges = locations.iter().enumerate().map(|(id, location)| {
        let id = SegmentId::from(u32::try_from(id).unwrap_or(u32::MAX));
        ((location.offset, location.length as usize), id)
    });
    Arc::new(ranges.collect())
}

/// The IO a scan's splits read through: one [`SplitIo`] per split's driver run.
///
/// A file opened for reading provides one that drives the file's coalescing reads directly. Any
/// other segment source is served by [`SegmentScanIo`].
pub trait ScanIo: Send + Sync {
    /// A fresh source for one split's driver run.
    fn split_io(&self) -> Arc<dyn SplitIo>;
}

/// One split's IO: the protocol's [`IoSource`], which the split can also wait on without blocking.
pub trait SplitIo: IoSource {
    /// The next completed fetch, registering `cx` to be woken when one completes.
    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>>;
}

/// Serves splits from any [`SegmentSource`], by segment id.
pub(super) struct SegmentScanIo {
    segments: Arc<dyn SegmentSource>,
    ranges: SegmentRanges,
}

impl SegmentScanIo {
    pub(super) fn new(segments: Arc<dyn SegmentSource>, ranges: SegmentRanges) -> Self {
        Self { segments, ranges }
    }
}

impl ScanIo for SegmentScanIo {
    fn split_io(&self) -> Arc<dyn SplitIo> {
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
/// the next with [`poll_completion`](SplitIo::poll_completion) when the driver has nothing else to
/// run. [`wait`](IoSource::wait) is never called. Optional intents are declined.
///
/// An announcement requests the segment and holds the request, never polled, until the source is
/// dropped, so a source that coalesces registered requests can fold it into a nearby read.
/// A read started ahead of the fetch that needs it.
type Prefetched = Shared<BoxFuture<'static, Result<BufferHandle, Arc<VortexError>>>>;

pub(super) struct SegmentIoSource {
    segments: Arc<dyn SegmentSource>,
    ranges: SegmentRanges,
    reads: Mutex<FuturesUnordered<BoxFuture<'static, Completion>>>,
    /// Reads started by prefetches, which later fetches of the same segment wait on.
    prefetched: Mutex<HashMap<SegmentId, Prefetched>>,
    /// Announced segments, requested and never polled.
    announced: Mutex<Vec<SegmentFuture>>,
}

impl SegmentIoSource {
    pub(super) fn new(segments: Arc<dyn SegmentSource>, ranges: SegmentRanges) -> Self {
        Self {
            segments,
            ranges,
            reads: Mutex::new(FuturesUnordered::new()),
            prefetched: Mutex::default(),
            announced: Mutex::default(),
        }
    }
}

impl SplitIo for SegmentIoSource {
    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        match self.reads.lock().poll_next_unpin(cx) {
            Poll::Ready(Some(completion)) => Poll::Ready(Ok(completion)),
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!(
                "The split's driver is waiting with no reads in flight"
            ))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl IoSource for SegmentIoSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let reads = self.reads.lock();
        let mut prefetched = self.prefetched.lock();
        for request in batch {
            let IoTarget::Range { offset, len } = request.target else {
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
                if !prefetched.contains_key(&segment) {
                    self.announced.lock().push(self.segments.request(segment));
                }
                continue;
            }
            if request.intent == IoIntent::Prefetch {
                prefetched.entry(segment).or_insert_with(|| {
                    let read = self
                        .segments
                        .request(segment)
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
            let bytes = match prefetched.get(&segment) {
                Some(read) => read
                    .clone()
                    .map_err(move |err| {
                        vortex_err!("prefetched read of segment {segment} failed: {err}")
                    })
                    .boxed(),
                None => self.segments.request(segment),
            };
            let id = request.request;
            reads.push(
                async move {
                    Completion {
                        owner,
                        request: id,
                        result: bytes.await.map(IoResult::Bytes),
                    }
                }
                .boxed(),
            );
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        // The split's future polls again with its own waker before it waits, so a read that
        // finishes after this check is not missed.
        let mut cx = Context::from_waker(Waker::noop());
        match self.reads.lock().poll_next_unpin(&mut cx) {
            Poll::Ready(completion) => Ok(completion),
            Poll::Pending => Ok(None),
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("SegmentIoSource is awaited through poll_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        self.reads.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use futures::future;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_io::request::IoIntent;
    use vortex_io::request::IoOwnerId;
    use vortex_io::request::IoRequest;
    use vortex_io::request::IoRequestId;
    use vortex_io::request::IoSource;
    use vortex_io::request::IoTarget;

    use super::SegmentIoSource;
    use super::SplitIo;
    use super::segment_ranges;
    use crate::scan::planning::SegmentLocation;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;

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
            target: IoTarget::Range { offset: 0, len: 4 },
        }
    }

    /// A fetch of a prefetched segment waits on the prefetch's read rather than reading again,
    /// and only the fetch is delivered.
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
        Ok(())
    }
}
