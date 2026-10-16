// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
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
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_utils::aliases::hash_map::HashMap;

use crate::scan::execution::SegmentLocation;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// File segments are normally ordered, so range lookup can reuse the location table directly.
/// Custom segment sources can supply arbitrary order and retain indexed lookup.
pub(super) enum SegmentRanges {
    Sorted(Arc<[SegmentLocation]>),
    Indexed(HashMap<(u64, usize), SegmentId>),
}

pub(super) fn segment_ranges(locations: Arc<[SegmentLocation]>) -> Arc<SegmentRanges> {
    if locations.is_sorted_by_key(|location| (location.offset, location.length)) {
        return Arc::new(SegmentRanges::Sorted(locations));
    }
    Arc::new(SegmentRanges::Indexed(
        locations
            .iter()
            .enumerate()
            .map(|(id, location)| {
                let id = SegmentId::from(u32::try_from(id).unwrap_or(u32::MAX));
                ((location.offset, location.length as usize), id)
            })
            .collect(),
    ))
}

impl SegmentRanges {
    fn get(&self, offset: u64, len: usize) -> Option<SegmentId> {
        match self {
            Self::Sorted(locations) => {
                let end = locations.partition_point(|location| {
                    (location.offset, location.length as usize) <= (offset, len)
                });
                let index = end.checked_sub(1)?;
                let location = locations[index];
                // Match the indexed path's last-entry rule for duplicate ranges.
                (location.offset == offset && location.length as usize == len)
                    .then(|| SegmentId::from(u32::try_from(index).unwrap_or(u32::MAX)))
            }
            Self::Indexed(ranges) => ranges.get(&(offset, len)).copied(),
        }
    }
}

/// Serves splits from any [`SegmentSource`], by segment id.
pub(super) struct SegmentScanIo {
    segments: Arc<dyn SegmentSource>,
    ranges: Arc<SegmentRanges>,
}

impl SegmentScanIo {
    pub(super) fn new(segments: Arc<dyn SegmentSource>, ranges: Arc<SegmentRanges>) -> Self {
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
    ranges: Arc<SegmentRanges>,
    reads: Mutex<FuturesUnordered<BoxFuture<'static, Completion>>>,
    ready: Mutex<VecDeque<Completion>>,
    /// Reads started by prefetches, consumed by the next fetch of their segment.
    prefetched: Mutex<HashMap<SegmentId, Prefetched>>,
    /// Announced segments, requested and never polled.
    announced: Mutex<HashMap<SegmentId, VecDeque<SegmentFuture>>>,
}

/// A read started ahead of the fetch that needs it.
type Prefetched = Shared<BoxFuture<'static, Result<BufferHandle, Arc<VortexError>>>>;

impl SegmentIoSource {
    pub(super) fn new(segments: Arc<dyn SegmentSource>, ranges: Arc<SegmentRanges>) -> Self {
        Self {
            segments,
            ranges,
            reads: Mutex::new(FuturesUnordered::new()),
            ready: Mutex::default(),
            prefetched: Mutex::default(),
            announced: Mutex::default(),
        }
    }
}

impl IoSource for SegmentIoSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let reads = self.reads.lock();
        let mut ready = self.ready.lock();
        let mut prefetched = self.prefetched.lock();
        for request in batch {
            let IoTarget::Range { offset, len, .. } = request.target else {
                vortex_bail!(
                    "SegmentIoSource only serves byte ranges, not {:?}",
                    request.target
                );
            };
            let segment = self
                .ranges
                .get(offset, len)
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
            let mut bytes = match prefetched.remove(&segment) {
                Some(read) => read
                    .map_err(move |err| {
                        vortex_err!("prefetched read of segment {segment} failed: {err}")
                    })
                    .boxed(),
                None => self
                    .announced
                    .lock()
                    .get_mut(&segment)
                    .and_then(VecDeque::pop_front)
                    .unwrap_or_else(|| self.segments.request(segment)),
            };
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
        if let Some(completion) = self.ready.lock().pop_front() {
            return Ok(Some(completion));
        }
        // The split's future polls again with its own waker before it waits, so a read that
        // finishes after this check is not missed.
        let mut cx = Context::from_waker(Waker::noop());
        match self.reads.lock().poll_next_unpin(&mut cx) {
            Poll::Ready(completion) => Ok(completion),
            Poll::Pending => Ok(None),
        }
    }

    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        if let Some(completion) = self.ready.lock().pop_front() {
            return Poll::Ready(Ok(completion));
        }
        match self.reads.lock().poll_next_unpin(cx) {
            Poll::Ready(Some(completion)) => Poll::Ready(Ok(completion)),
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!(
                "The split's driver is waiting with no reads in flight"
            ))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("SegmentIoSource is awaited through poll_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        self.reads.lock().clear();
        self.ready.lock().clear();
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
    use rstest::rstest;
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
    use super::SegmentRanges;
    use super::segment_ranges;
    use crate::scan::execution::SegmentLocation;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;

    #[rstest]
    #[case::sorted(vec![(0, 0), (0, 4), (8, 4), (u64::MAX, 0)], true, vec![Some(0), Some(1), Some(2), Some(3)])]
    #[case::unordered(vec![(8, 4), (0, 4), (0, 0), (u64::MAX, 0)], false, vec![Some(2), Some(1), Some(0), Some(3)])]
    #[case::duplicates(vec![(0, 0), (0, 0), (0, 4), (8, 4), (8, 4)], true, vec![Some(1), Some(2), Some(4), None])]
    #[case::empty(vec![], true, vec![None, None, None, None])]
    fn range_lookup_preserves_segment_ids(
        #[case] entries: Vec<(u64, u32)>,
        #[case] sorted: bool,
        #[case] expected: Vec<Option<u32>>,
    ) {
        let locations: Arc<[_]> = entries
            .into_iter()
            .map(|(offset, length)| SegmentLocation {
                offset,
                length,
                alignment: Alignment::none(),
            })
            .collect();
        let ranges = segment_ranges(Arc::clone(&locations));
        assert_eq!(matches!(ranges.as_ref(), SegmentRanges::Sorted(_)), sorted);
        if let SegmentRanges::Sorted(reused) = ranges.as_ref() {
            assert!(Arc::ptr_eq(reused, &locations));
        }
        for ((offset, len), expected) in [(0, 0), (0, 4), (8, 4), (u64::MAX, 0)]
            .into_iter()
            .zip(expected)
        {
            assert_eq!(ranges.get(offset, len), expected.map(SegmentId::from));
        }
        for (offset, len) in [(0, 1), (1, 4), (8, 0), (8, 5)] {
            assert_eq!(ranges.get(offset, len), None);
        }
    }

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
        let io = SegmentIoSource::new(segments, segment_ranges(Arc::from(locations)));
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
    /// and only the fetch is delivered.
    #[tokio::test]
    async fn fetch_reuses_a_prefetched_read() -> VortexResult<()> {
        let segments = Arc::new(CountingSegments::default());
        let location = SegmentLocation {
            offset: 0,
            length: 4,
            alignment: Alignment::none(),
        };
        let io = SegmentIoSource::new(
            Arc::clone(&segments) as _,
            segment_ranges(Arc::from([location])),
        );

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
        let io = SegmentIoSource::new(
            Arc::clone(&segments) as _,
            segment_ranges(Arc::from([location])),
        );
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
            assert_eq!(
                segments.0.load(Ordering::Relaxed),
                if id < 4 { 2 } else { 3 }
            );
        }
        Ok(())
    }
}
