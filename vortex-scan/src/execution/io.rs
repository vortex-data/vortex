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
use vortex_io::request::IoRequestId;
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
    reads: Mutex<FuturesUnordered<BoxFuture<'static, Result<Completion, Aborted>>>>,
    ready: Mutex<VecDeque<Completion>>,
    /// Per-owner abort handles for in-flight subscriptions; aborted on [`release`](IoSource::release)
    /// so retired subscriptions drop immediately while leaving the underlying shared read intact.
    owners: Mutex<HashMap<IoOwnerId, HashMap<IoRequestId, AbortHandle>>>,
    /// Every read the session has started, by segment, shared by later fetches of the segment
    /// until the session is cleared.
    prefetched: Mutex<HashMap<SegmentId, Prefetched>>,
    /// Announced segments, requested and never polled.
    announced: Mutex<HashMap<SegmentId, VecDeque<SegmentFuture>>>,
}

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
            prefetched: Mutex::default(),
            announced: Mutex::default(),
        }
    }

    fn remove_owner_handle(&self, owner: IoOwnerId, request: IoRequestId) {
        let mut owners = self.owners.lock();
        if let Some(handles) = owners.get_mut(&owner) {
            handles.remove(&request);
            if handles.is_empty() {
                owners.remove(&owner);
            }
        }
    }
}

impl IoSource for SegmentIoSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let reads = self.reads.lock();
        let mut ready = self.ready.lock();
        let mut prefetched = self.prefetched.lock();
        let mut owners = self.owners.lock();
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
            let (read, abort) = abortable(async move {
                Completion {
                    owner,
                    request: id,
                    result: bytes.await.map(IoResult::Bytes),
                }
            });
            owners.entry(owner).or_default().insert(id, abort);
            reads.push(read.boxed());
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        if let Some(completion) = self.ready.lock().pop_front() {
            return Ok(Some(completion));
        }
        // The split's future polls again with its own waker before it waits, so a read that
        // finishes after this check is not missed.
        let mut reads = self.reads.lock();
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            match reads.poll_next_unpin(&mut cx) {
                Poll::Ready(Some(Ok(completion))) => {
                    drop(reads);
                    self.remove_owner_handle(completion.owner, completion.request);
                    return Ok(Some(completion));
                }
                Poll::Ready(Some(Err(_))) => continue,
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
                Poll::Ready(Some(Ok(completion))) => {
                    drop(reads);
                    self.remove_owner_handle(completion.owner, completion.request);
                    return Poll::Ready(Ok(completion));
                }
                Poll::Ready(Some(Err(_))) => continue,
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
        let handles = self.owners.lock().remove(&owner);
        if let Some(handles) = handles {
            for handle in handles.into_values() {
                handle.abort();
            }
        }
        self.ready
            .lock()
            .retain(|completion| completion.owner != owner);
    }

    fn clear(&self) {
        self.owners.lock().clear();
        self.reads.lock().clear();
        self.ready.lock().clear();
        self.prefetched.lock().clear();
        self.announced.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;
    use std::task::Waker;

    use futures::FutureExt;
    use futures::channel::oneshot;
    use futures::future;
    use parking_lot::Mutex;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;
    use vortex_io::request::IoConsumer;
    use vortex_io::request::IoIntent;
    use vortex_io::request::IoOwnerId;
    use vortex_io::request::IoRequest;
    use vortex_io::request::IoRequestId;
    use vortex_io::request::IoResult;
    use vortex_io::request::IoSource;
    use vortex_io::request::IoTarget;
    use vortex_mask::Mask;

    use super::SegmentIoSource;
    use super::segment_ranges;
    use crate::execution::PreparedScan;
    use crate::execution::SegmentLocation;
    use crate::execution::SelectedRows;
    use crate::execution::cache::ScanCache;
    use crate::execution::operator::Cx;
    use crate::execution::operator::Input;
    use crate::execution::operator::Operator;
    use crate::execution::operator::Source;
    use crate::execution::operator::SourceBlocked;
    use crate::execution::operator::SourceReady;
    use crate::execution::operator::Step;
    use crate::execution::pipeline::Chain;
    use crate::execution::pipeline::synthetic::SyntheticPlan;
    use crate::execution::projection::ProjectionPlanner;
    use crate::planning::driver::Progress;
    use crate::planning::driver::Run;
    use crate::planning::planner::Planner;
    use crate::planning::planner::PlannerOutput;
    use crate::planning::planner::State;
    use crate::planning::planner::WorkScope;
    use crate::selection::Selection;
    use crate::source::SegmentFuture;
    use crate::source::SegmentId;
    use crate::source::SegmentSource;
    use crate::test::SESSION;

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
        assert!(io.owners.lock().is_empty());
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

    #[test]
    fn release_drops_queued_ready_completions() -> VortexResult<()> {
        let segments = Arc::new(CountingSegments::default());
        let location = SegmentLocation {
            offset: 0,
            length: 4,
            alignment: Alignment::none(),
        };
        let io = SegmentIoSource::new(segments, segment_ranges(&[location]));
        io.submit(IoOwnerId(0), vec![request(IoIntent::Fetch, 0)])?;
        io.submit(IoOwnerId(1), vec![request(IoIntent::Fetch, 1)])?;

        io.release(IoOwnerId(0));

        let remaining = io
            .poll()?
            .expect("other owner's ready completion must remain");
        assert_eq!(remaining.owner, IoOwnerId(1));
        assert_eq!(remaining.request, IoRequestId(1));
        assert!(io.poll()?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn retired_projection_suppresses_late_inflight_completion() -> VortexResult<()> {
        struct FirstOf {
            wanted: Vec<SegmentId>,
            done: bool,
        }

        impl Source for FirstOf {
            fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
                if self.done {
                    return Ok(SourceReady::Ready);
                }
                if !self.wanted.is_empty() {
                    while let Some(segment) = self.wanted.pop() {
                        cx.request_read(segment);
                    }
                    return Ok(SourceReady::Blocked(SourceBlocked::Io));
                }
                Ok(if cx.has_bytes() {
                    SourceReady::Ready
                } else {
                    SourceReady::Blocked(SourceBlocked::Io)
                })
            }
        }

        impl Operator for FirstOf {
            fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
                if self.done {
                    return Ok(Step::End);
                }
                let (segment, _) = cx
                    .take_segment()
                    .ok_or_else(|| vortex_err!("FirstOf computed before a read arrived"))?;
                self.done = true;
                let value = PrimitiveArray::from_buffer_handle(
                    BufferHandle::new_host(
                        Buffer::from(vec![u64::from(*segment)]).into_byte_buffer(),
                    ),
                    PType::U64,
                    Validity::NonNullable,
                )
                .into_array();
                Ok(Step::Final(value))
            }
        }

        struct GatedSegments(Mutex<[Option<oneshot::Receiver<ByteBuffer>>; 3]>);

        impl SegmentSource for GatedSegments {
            fn request(&self, id: SegmentId) -> SegmentFuture {
                let receiver = self.0.lock()[usize::try_from(*id).expect("segment fits")]
                    .take()
                    .expect("each segment requested once");
                async move {
                    receiver
                        .await
                        .map(BufferHandle::new_host)
                        .map_err(|err| vortex_err!("{err}"))
                }
                .boxed()
            }
        }

        let (tx0, rx0) = oneshot::channel();
        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();
        let segments: Arc<dyn SegmentSource> =
            Arc::new(GatedSegments(Mutex::new([Some(rx0), Some(rx1), Some(rx2)])));
        let locations: Arc<[SegmentLocation]> =
            Arc::from([0, 4, 8].map(|offset| SegmentLocation {
                offset,
                length: 4,
                alignment: Alignment::none(),
            }));
        let ranges = segment_ranges(&locations);
        let cache = ScanCache::with_retained_bytes(0);
        let values = cache.values();
        let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let projection = SyntheticPlan::new(dtype.clone(), 2, Vec::new(), |rows, _, _| {
            let wanted = if rows.start == 0 {
                vec![SegmentId::from(1), SegmentId::from(0)]
            } else {
                vec![SegmentId::from(2)]
            };
            Ok(Some(Chain::new(FirstOf {
                wanted,
                done: false,
            })))
        })
        .into_plan();
        let scan = Arc::new(PreparedScan {
            session: SESSION.clone(),
            rows: 0..2,
            selection: Selection::All,
            segments,
            ranges,
            cache,
            locations,
            projection,
            filter: None,
            row_offset: 0,
            values,
            chunk_rows: 1,
        });

        struct ParentPlanner(VecDeque<(WorkScope, Box<dyn Planner>)>);

        impl IoConsumer for ParentPlanner {
            fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
        }

        impl Planner for ParentPlanner {
            fn state(&self) -> State {
                if self.0.is_empty() {
                    State::Done
                } else {
                    State::NeedsCompute
                }
            }

            fn compute(&mut self) -> VortexResult<PlannerOutput> {
                Ok(match self.0.pop_front() {
                    Some((scope, planner)) => PlannerOutput::Planner(scope, planner),
                    None => PlannerOutput::Done,
                })
            }
        }

        let io = scan.io().session();
        let mut run = Run::new();
        let children = (0..2u64)
            .map(|row| {
                let scope = WorkScope {
                    file_ordinal: 0,
                    rows: row..row + 1,
                };
                let planner: Box<dyn Planner> = Box::new(ProjectionPlanner::new(
                    Arc::clone(&scan),
                    SelectedRows {
                        scope: scope.clone(),
                        mask: Mask::new_true(1),
                    },
                ));
                (scope, planner)
            })
            .collect();
        run.admit(
            Box::new(ParentPlanner(children)),
            WorkScope {
                file_ordinal: 0,
                rows: 0..2,
            },
            Arc::clone(&io),
        );

        let mut tx0 = Some(tx0);
        let mut tx1 = Some(tx1);
        let mut tx2 = Some(tx2);
        let mut batches = Vec::new();
        loop {
            match run.advance()? {
                Progress::Batch(batch) => batches.push(batch.array),
                Progress::Waiting => {
                    if let Some(tx) = tx0.take() {
                        tx.send(ByteBuffer::from(vec![0u8; 4]))
                            .expect("segment 0 receiver is live");
                    } else if let Some(tx) = tx1.take() {
                        // Deliver the retired projection's in-flight segment 1 first; it must be
                        // suppressed rather than delivered to `Run`, while segment 2 reaches the
                        // live sibling.
                        tx.send(ByteBuffer::from(vec![1u8; 4]))
                            .expect("segment 1 read still finishes");
                        assert!(io.poll()?.is_none());
                        tx2.take()
                            .expect("segment 2 sender")
                            .send(ByteBuffer::from(vec![2u8; 4]))
                            .expect("segment 2 receiver is live");
                    }
                    let completion = std::future::poll_fn(|cx| io.poll_completion(cx)).await?;
                    run.complete(completion)?;
                }
                Progress::RootDone(_) => {}
                Progress::Idle => break,
                Progress::Spawn(scope, planner) => {
                    run.admit(planner, scope, Arc::clone(&io));
                }
            }
        }
        assert_arrays_eq!(
            ChunkedArray::try_new(batches, dtype)?.into_array(),
            PrimitiveArray::from_iter([0u64, 2]).into_array(),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn release_drops_inflight_subscription_without_waiting_for_read() -> VortexResult<()> {
        let (_sender, receiver) = oneshot::channel();
        let segments = Arc::new(MixedSegments(Mutex::new(Some(receiver))));
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(segments, segment_ranges(&locations));
        let mut pending = request(IoIntent::Fetch, 1);
        pending.target = IoTarget::range(4, 4);

        io.submit(IoOwnerId(0), vec![pending])?;
        io.release(IoOwnerId(0));

        let mut cx = Context::from_waker(Waker::noop());
        let Poll::Ready(Err(err)) = io.poll_completion(&mut cx) else {
            vortex_bail!("releasing the only subscriber leaves no reads in flight");
        };
        assert!(
            err.to_string().contains("no reads in flight"),
            "unexpected error: {err}"
        );
        assert!(io.reads.lock().is_empty());
        Ok(())
    }

    #[test]
    fn clear_drops_pending_owner_subscriptions() -> VortexResult<()> {
        let (_sender, receiver) = oneshot::channel();
        let segments = Arc::new(MixedSegments(Mutex::new(Some(receiver))));
        let locations = [0, 4].map(|offset| SegmentLocation {
            offset,
            length: 4,
            alignment: Alignment::none(),
        });
        let io = SegmentIoSource::new(segments, segment_ranges(&locations));
        let mut pending = request(IoIntent::Fetch, 1);
        pending.target = IoTarget::range(4, 4);

        io.submit(IoOwnerId(0), vec![pending])?;
        assert!(!io.owners.lock().is_empty());

        io.clear();
        assert!(io.owners.lock().is_empty());
        assert!(io.reads.lock().is_empty());
        Ok(())
    }
}
