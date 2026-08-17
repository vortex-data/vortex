// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::VecDeque;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::SelectAll;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::ReadAtRequest;
use vortex_io::VortexReadAt;
use vortex_io::runtime::Handle;
use vortex_io::runtime::JoinOutcome;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::Counter;
use vortex_metrics::Histogram;
use vortex_metrics::Label;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

use crate::SegmentSpec;
use crate::read::IoRequest;
use crate::read::IoRequestStream;
use crate::read::ReadRequest;
use crate::read::RequestId;

#[derive(Debug)]
/// Events sent from segment futures to the coalescing read driver.
pub enum ReadEvent {
    /// A segment read has been registered.
    Request(ReadRequest),
    /// A registered read future has been polled.
    Polled(RequestId),
    /// A registered read future was dropped before completion.
    Dropped(RequestId),
}

/// A [`SegmentSource`] for file-like IO.
/// ## Coalescing and Pre-fetching
///
/// It is important to understand the semantics of the read futures returned by a [`FileSegmentSource`].
/// Under the hood, each instance is backed by a stream that services read requests by
/// applying coalescing and concurrency constraints.
///
/// Each read future has four states:
/// * `registered` - the read future has been created, but not yet polled.
/// * `requested` - the read future has been polled.
/// * `in-flight` - the read request has been sent to the underlying storage system.
/// * `resolved` - the read future has completed and resolved a result.
///
/// When a read request is `registered`, it will not itself trigger any I/O, but is eligible to
/// be coalesced with other requests.
///
/// If a read future is dropped, it will be canceled if possible. This depends on the current
/// state of the request, as well as whether the underlying storage system supports cancellation.
///
/// I/O requests will be processed in the order they are `registered`, however coalescing may mean
/// other registered requests are lumped together into a single I/O operation.
/// A cloneable handle to the background read driver, shared by every in-flight [`ReadFuture`].
///
/// [`Shared`] fans a single completion out to all readers with correct waker bookkeeping, so a
/// reader is always woken when the driver finishes — even if another reader polled the driver more
/// recently and was then dropped. Its output is `()`; a driver panic is carried out of band in
/// [`DriverPanic`] so it can be re-raised on the reader side.
type SharedDriver = Shared<BoxFuture<'static, ()>>;

/// Slot holding the driver's panic payload, if it panicked while driving reads. The first reader to
/// observe completion takes the payload and re-raises it; later readers report a graceful error.
type DriverPanic = Arc<Mutex<Option<Box<dyn Any + Send>>>>;

const MAX_PARTIAL_SUBMISSION_REQUESTS: usize = 512;
const MAX_PARTIAL_SUBMISSION_BYTES: usize = 16 << 20;

fn partial_submission_len(requests: &VecDeque<IoRequest>) -> usize {
    let mut count = 0usize;
    let mut bytes = 0usize;
    for request in requests.iter().take(MAX_PARTIAL_SUBMISSION_REQUESTS) {
        if !request.is_partial() {
            break;
        }
        let next_bytes = bytes.saturating_add(request.len());
        if count > 0 && next_bytes > MAX_PARTIAL_SUBMISSION_BYTES {
            break;
        }
        count += 1;
        bytes = next_bytes;
    }
    count
}

fn validate_read_result(
    request: &IoRequest,
    result: VortexResult<BufferHandle>,
) -> VortexResult<BufferHandle> {
    result.and_then(|buffer| {
        if request.len() != buffer.len() {
            return Err(vortex_err!(
                "FileSegmentSource: expected buffer of length {} but received {}. {:?}",
                request.len(),
                buffer.len(),
                request
            ));
        }
        Ok(buffer)
    })
}

pub struct FileSegmentSource {
    segments: Arc<[SegmentSpec]>,
    /// A queue for sending read request events to the I/O stream.
    events: mpsc::UnboundedSender<ReadEvent>,
    /// Background request driver, joined by readers to surface a driver panic.
    driver: SharedDriver,
    /// Panic payload captured if the driver panicked while driving reads.
    driver_panic: DriverPanic,
    /// The next read request ID.
    next_id: Arc<AtomicUsize>,
    /// Preferred size of canonical byte ranges for the underlying source.
    preferred_read_size: Option<u64>,
}

impl FileSegmentSource {
    /// Open a file-backed segment source over `reader`.
    ///
    /// The returned source spawns a background driver on `handle` that coalesces and executes
    /// random-access read requests.
    pub fn open<R: VortexReadAt + Clone>(
        segments: Arc<[SegmentSpec]>,
        reader: R,
        handle: Handle,
        metrics: RequestMetrics,
    ) -> Self {
        let (send, recv) = mpsc::unbounded();
        let preferred_read_size = reader.preferred_read_size();

        let max_alignment = segments
            .iter()
            .map(|segment| segment.alignment)
            .max()
            .unwrap_or_else(Alignment::none);
        let coalesce_config = reader.coalesce_config().map(|mut config| {
            // Aligning the coalesced start down can add up to (alignment - 1) bytes.
            // Increase max_size to keep the effective payload window consistent.
            let extra = (*max_alignment as u64).saturating_sub(1);
            config.max_size = config.max_size.saturating_add(extra);
            config
        });
        let concurrency = reader.concurrency();
        if concurrency == 0 {
            vortex_panic!(
                "VortexReadAt::concurrency returned 0 (uri={:?}); this would stall I/O",
                reader.uri()
            );
        }

        let stream = IoRequestStream::new(
            StreamExt::boxed(recv),
            coalesce_config,
            max_alignment,
            MAX_PARTIAL_SUBMISSION_REQUESTS,
            metrics.clone(),
        )
        .boxed();

        let drive_fut = async move {
            let mut batches = stream.fuse();
            let mut pending = VecDeque::<IoRequest>::new();
            let mut reads = SelectAll::new();
            let mut num_active = 0usize;
            let mut batches_done = false;

            loop {
                if !batches_done {
                    loop {
                        match batches.next().now_or_never() {
                            Some(Some(batch)) => pending.extend(batch),
                            Some(None) => {
                                batches_done = true;
                                break;
                            }
                            None => break,
                        }
                    }
                }

                while num_active < concurrency && !pending.is_empty() {
                    // A partial batch is submitted through one `read_ranges` stream. Do not
                    // refill individual slots from another partial batch as each range finishes:
                    // that turns a queued group into one syscall submission per completion. Let
                    // the current group drain, then submit all ready partial ranges together.
                    if num_active != 0 && pending.front().is_some_and(IoRequest::is_partial) {
                        break;
                    }
                    let batch_len =
                        if num_active == 0 && pending.front().is_some_and(IoRequest::is_partial) {
                            partial_submission_len(&pending)
                        } else {
                            (concurrency - num_active).min(pending.len())
                        };
                    let reqs = pending.drain(..batch_len).collect::<Vec<_>>();
                    num_active += batch_len;

                    metrics.read_ranges_calls.add(1);
                    metrics.read_ranges_num_ranges.update(batch_len as f64);
                    if batch_len > 1 {
                        metrics.read_ranges_multi.add(1);
                    }
                    tracing::trace!(
                        target: "vortex_file::read_ranges",
                        num_ranges = batch_len,
                        num_active,
                        "submitting positional read batch"
                    );

                    let requests = reqs
                        .iter()
                        .map(|req| ReadAtRequest::new(req.offset(), req.len(), req.alignment()))
                        .collect::<Vec<_>>()
                        .into();
                    let mut remaining = reqs.into_iter().map(Some).collect::<Vec<_>>();
                    let mut results = reader.read_ranges(requests);
                    reads.push(
                        async_stream::stream! {
                            while let Some((request, result)) = results.next().await {
                                let Some(position) = remaining.iter().position(|req| {
                                    req.as_ref().is_some_and(|req| {
                                        req.offset() == request.offset
                                            && req.len() == request.length
                                            && req.alignment() == request.alignment
                                    })
                                }) else {
                                    tracing::warn!(?request, "reader returned an unknown range");
                                    continue;
                                };
                                let req = remaining[position]
                                    .take()
                                    .vortex_expect("matched request is present");
                                yield (req, result);
                            }
                            for req in remaining.into_iter().flatten() {
                                let error = vortex_err!(
                                    "FileSegmentSource: read_ranges ended before resolving request. {:?}",
                                    req
                                );
                                yield (req, Err(error));
                            }
                        }
                        .boxed(),
                    );
                }

                if batches_done && num_active == 0 {
                    break;
                }
                if num_active == 0 {
                    match batches.next().await {
                        Some(batch) => pending.extend(batch),
                        None => batches_done = true,
                    }
                    continue;
                }

                let next_read = reads.next();
                let next = if batches_done {
                    future::Either::Left((next_read.await, batches.next()))
                } else {
                    future::select(next_read, batches.next()).await
                };
                match next {
                    future::Either::Left((result, _)) => {
                        if let Some((req, result)) = result {
                            num_active -= 1;
                            let result = validate_read_result(&req, result);
                            req.resolve(result);
                        }
                    }
                    future::Either::Right((batch, _)) => match batch {
                        Some(batch) => pending.extend(batch),
                        None => batches_done = true,
                    },
                }
            }
        };

        // Spawn the driver so the runtime makes I/O progress independently of any reader. Readers
        // join it (below) only to surface a panic raised while driving reads.
        let mut task = handle.spawn(drive_fut);
        let driver_panic: DriverPanic = Arc::new(Mutex::new(None));
        let driver = {
            let driver_panic = Arc::clone(&driver_panic);
            async move {
                // Poll for the terminal outcome without re-raising: a benign abort (runtime
                // teardown) resolves to `()` so readers report a graceful error, while a panic is
                // stashed for the first reader to re-raise.
                if let JoinOutcome::Panicked(panic) = future::poll_fn(|cx| task.poll_join(cx)).await
                {
                    *driver_panic.lock() = Some(panic);
                }
            }
            .boxed()
            .shared()
        };

        Self {
            segments,
            events: send,
            driver,
            driver_panic,
            next_id: Arc::new(AtomicUsize::new(0)),
            preferred_read_size,
        }
    }
}

impl SegmentSource for FileSegmentSource {
    fn preferred_read_size(&self) -> Option<u64> {
        self.preferred_read_size
    }

    fn segment_len(&self, id: SegmentId) -> Option<u64> {
        self.segments
            .get(*id as usize)
            .map(|spec| u64::from(spec.length))
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        let Some(length) = self.segment_len(id) else {
            return future::ready(Err(vortex_err!("Missing segment: {}", id))).boxed();
        };
        self.request_range_with_coalesce_distance(id, 0..length, None)
    }

    fn request_range(&self, segment_id: SegmentId, range: Range<u64>) -> SegmentFuture {
        self.request_range_with_coalesce_distance(
            segment_id,
            range,
            self.preferred_read_size.map(|size| size / 4),
        )
    }

    fn request_ranges(&self, segment_id: SegmentId, ranges: Vec<Range<u64>>) -> Vec<SegmentFuture> {
        let coalesce_distance = self.preferred_read_size.map(|size| size / 4);
        let mut registered = ranges
            .into_iter()
            .map(|range| self.register_range(segment_id, range, coalesce_distance))
            .collect::<Vec<_>>();
        let poll_ids: Arc<[usize]> = registered
            .iter()
            .filter_map(|registration| registration.as_ref().ok().map(|read| read.id))
            .collect();
        let poll_once = Arc::new(AtomicBool::new(false));

        registered
            .drain(..)
            .map(|registration| match registration {
                Ok(read) => self.read_future(read, Arc::clone(&poll_ids), Arc::clone(&poll_once)),
                Err(error) => future::ready(Err(error)).boxed(),
            })
            .collect()
    }
}

impl FileSegmentSource {
    fn request_range_with_coalesce_distance(
        &self,
        segment_id: SegmentId,
        range: Range<u64>,
        coalesce_distance: Option<u64>,
    ) -> SegmentFuture {
        match self.register_range(segment_id, range, coalesce_distance) {
            Ok(read) => {
                let poll_ids = Arc::from([read.id]);
                self.read_future(read, poll_ids, Arc::new(AtomicBool::new(false)))
            }
            Err(error) => future::ready(Err(error)).boxed(),
        }
    }

    fn register_range(
        &self,
        segment_id: SegmentId,
        range: Range<u64>,
        coalesce_distance: Option<u64>,
    ) -> VortexResult<RegisteredRead> {
        // We eagerly register the read request here assuming the behaviour of
        // [`FileSegmentSource`], where coalescing becomes effective prior to polling.
        let spec = *match self.segments.get(*segment_id as usize) {
            Some(spec) => spec,
            None => return Err(vortex_err!("Missing segment: {}", segment_id)),
        };

        if range.start > range.end || range.end > u64::from(spec.length) {
            return Err(vortex_err!(
                "Segment {} range {}..{} is out of bounds for a {}-byte segment",
                segment_id,
                range.start,
                range.end,
                spec.length
            ));
        }

        let SegmentSpec {
            offset, alignment, ..
        } = spec;

        let Some(offset) = offset.checked_add(range.start) else {
            return Err(vortex_err!("Segment range offset overflow"));
        };
        let Ok(length) = usize::try_from(range.end - range.start) else {
            return Err(vortex_err!("Segment range length does not fit usize"));
        };

        let (send, recv) = oneshot::channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let event = ReadEvent::Request(ReadRequest {
            id,
            offset,
            length,
            alignment,
            coalesce_distance,
            callback: send,
        });

        if let Err(error) = self.events.unbounded_send(event) {
            return Err(vortex_err!("Failed to submit read request: {error}"));
        }

        Ok(RegisteredRead {
            id,
            recv: recv.into_future(),
        })
    }

    fn read_future(
        &self,
        read: RegisteredRead,
        poll_ids: Arc<[usize]>,
        poll_once: Arc<AtomicBool>,
    ) -> SegmentFuture {
        ReadFuture {
            id: read.id,
            recv: read.recv,
            polled: false,
            finished: false,
            poll_ids,
            poll_once,
            events: self.events.clone(),
            driver: self.driver.clone(),
            driver_panic: Arc::clone(&self.driver_panic),
        }
        .boxed()
    }
}

struct RegisteredRead {
    id: usize,
    recv: oneshot::AsyncReceiver<VortexResult<BufferHandle>>,
}

/// A future that resolves a read request from a [`FileSegmentSource`].
///
/// See the documentation for [`FileSegmentSource`] for details on coalescing and pre-fetching.
/// If dropped, the read request will be canceled where possible.
struct ReadFuture {
    id: usize,
    recv: oneshot::AsyncReceiver<VortexResult<BufferHandle>>,
    polled: bool,
    finished: bool,
    poll_ids: Arc<[usize]>,
    poll_once: Arc<AtomicBool>,
    events: mpsc::UnboundedSender<ReadEvent>,
    driver: SharedDriver,
    driver_panic: DriverPanic,
}

impl Future for ReadFuture {
    type Output = VortexResult<BufferHandle>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.recv.poll_unpin(cx) {
            // note: we are skipping polled and dropped events for this if the future is ready on
            //       the first poll, that means this request was completed before it was polled,
            //       as part of a coalesced request.
            Poll::Ready(Ok(result)) => {
                self.finished = true;
                Poll::Ready(result)
            }
            // The request's sender was dropped, so the driver has finished. Join it so a panic
            // raised while driving reads is re-raised here rather than surfacing as a generic
            // error. Only report the dropped error once the driver has finished.
            Poll::Ready(Err(e)) => match self.driver.poll_unpin(cx) {
                Poll::Ready(()) => {
                    self.finished = true;
                    // Re-raise the driver panic on the first reader to observe it; later readers
                    // fall through to the graceful dropped error.
                    if let Some(panic) = self.driver_panic.lock().take() {
                        std::panic::resume_unwind(panic);
                    }
                    Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}")))
                }
                Poll::Pending => Poll::Pending,
            },
            Poll::Pending if !self.polled => {
                self.polled = true;
                if !self.poll_once.swap(true, Ordering::AcqRel) {
                    for &id in self.poll_ids.iter() {
                        if let Err(error) = self.events.unbounded_send(ReadEvent::Polled(id)) {
                            return Poll::Ready(Err(vortex_err!(
                                "ReadRequest dropped by runtime: {error}"
                            )));
                        }
                    }
                }
                Poll::Pending
            }
            _ => Poll::Pending,
        }
    }
}

impl Drop for ReadFuture {
    fn drop(&mut self) {
        // Completed requests have already left driver state.
        if self.finished {
            return;
        }

        // Best-effort cancellation signal to the I/O stream.
        drop(self.events.unbounded_send(ReadEvent::Dropped(self.id)));
    }
}

/// Metrics emitted by the file segment request driver.
#[derive(Clone)]
pub struct RequestMetrics {
    /// Number of individual segment requests observed by the driver.
    pub individual_requests: Counter,
    /// Number of physical reads after coalescing.
    pub coalesced_requests: Counter,
    /// Distribution of how many segment requests were merged into each physical read.
    pub num_requests_coalesced: Histogram,
    /// Number of calls made to [`VortexReadAt::read_ranges`](vortex_io::VortexReadAt::read_ranges).
    pub read_ranges_calls: Counter,
    /// Number of `read_ranges` calls containing more than one physical range.
    pub read_ranges_multi: Counter,
    /// Distribution of physical range counts submitted per `read_ranges` call.
    pub read_ranges_num_ranges: Histogram,
}

impl RequestMetrics {
    /// Create request metrics in `metrics_registry` with shared labels.
    pub fn new(metrics_registry: &dyn MetricsRegistry, labels: Vec<Label>) -> Self {
        Self {
            individual_requests: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.requests.individual"),
            coalesced_requests: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.requests.coalesced"),
            num_requests_coalesced: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .histogram("io.requests.coalesced.num_coalesced"),
            read_ranges_calls: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.read_ranges.calls"),
            read_ranges_multi: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.read_ranges.multi_range_calls"),
            read_ranges_num_ranges: MetricBuilder::new(metrics_registry)
                .add_labels(labels)
                .histogram("io.read_ranges.num_ranges"),
        }
    }
}

/// A [`SegmentSource`] that resolves segments synchronously from an
/// in-memory [`ByteBuffer`].
///
/// Resolves segments synchronously, bypassing the async I/O pipeline.
pub(crate) struct BufferSegmentSource {
    buffer: ByteBuffer,
    segments: Arc<[SegmentSpec]>,
}

impl BufferSegmentSource {
    /// Create a new `BufferSegmentSource` from a buffer and its segment map.
    pub fn new(buffer: ByteBuffer, segments: Arc<[SegmentSpec]>) -> Self {
        Self { buffer, segments }
    }
}

impl SegmentSource for BufferSegmentSource {
    fn segment_len(&self, id: SegmentId) -> Option<u64> {
        self.segments
            .get(*id as usize)
            .map(|spec| u64::from(spec.length))
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        let Some(length) = self.segment_len(id) else {
            return future::ready(Err(vortex_err!("Missing segment: {}", id))).boxed();
        };
        self.request_range(id, 0..length)
    }

    fn request_range(&self, id: SegmentId, range: Range<u64>) -> SegmentFuture {
        let spec = match self.segments.get(*id as usize) {
            Some(spec) => spec,
            None => {
                return future::ready(Err(vortex_err!("Missing segment: {}", id))).boxed();
            }
        };

        if range.start > range.end || range.end > u64::from(spec.length) {
            return future::ready(Err(vortex_err!(
                "Segment {} range {}..{} out of bounds for segment length {}",
                *id,
                range.start,
                range.end,
                spec.length
            )))
            .boxed();
        }

        let start = spec.offset as usize + range.start as usize;
        let end = spec.offset as usize + range.end as usize;
        if end > self.buffer.len() {
            return future::ready(Err(vortex_err!(
                "Segment {} range {}..{} out of bounds for buffer of length {}",
                *id,
                start,
                end,
                self.buffer.len()
            )))
            .boxed();
        }

        let slice = if range.start == 0 {
            self.buffer.slice(start..end).aligned(spec.alignment)
        } else {
            self.buffer.slice(start..end)
        };
        future::ready(Ok(BufferHandle::new_host(slice))).boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use futures::future::BoxFuture;
    use vortex_error::vortex_bail;
    use vortex_io::runtime::tokio::TokioRuntime;
    use vortex_layout::segments::SegmentSource;
    use vortex_metrics::DefaultMetricsRegistry;

    use super::*;

    #[derive(Clone)]
    struct MissingAndUnknownReadRanges;

    impl VortexReadAt for MissingAndUnknownReadRanges {
        fn concurrency(&self) -> usize {
            2
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(8) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async { panic!("read_at should not be called") }.boxed()
        }

        fn read_ranges(&self, _requests: Arc<[ReadAtRequest]>) -> vortex_io::ReadAtStream {
            let unknown = ReadAtRequest::new(8, 4, Alignment::none());
            let returned = ReadAtRequest::new(4, 4, Alignment::none());
            let buffer = BufferHandle::new_host(ByteBuffer::from(vec![0; 4]));
            futures::stream::iter([(unknown, Ok(buffer.clone())), (returned, Ok(buffer))]).boxed()
        }
    }

    #[tokio::test]
    async fn read_driver_ignores_unknown_results_and_reports_missing_requests() {
        let segments: Arc<[SegmentSpec]> = Arc::from([
            SegmentSpec {
                offset: 0,
                length: 4,
                alignment: Alignment::none(),
            },
            SegmentSpec {
                offset: 4,
                length: 4,
                alignment: Alignment::none(),
            },
        ]);
        let metrics = DefaultMetricsRegistry::default();
        let source = FileSegmentSource::open(
            segments,
            MissingAndUnknownReadRanges,
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        );

        let results = future::join_all([
            source.request(SegmentId::from(0)),
            source.request(SegmentId::from(1)),
        ])
        .await;

        assert!(results[0].is_err());
        match &results[1] {
            Ok(buffer) => assert_eq!(buffer.len(), 4),
            Err(error) => vortex_panic!("second request must resolve: {error}"),
        }
    }

    #[derive(Clone)]
    struct PanickingReadAt;

    impl VortexReadAt for PanickingReadAt {
        fn concurrency(&self) -> usize {
            1
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(4) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async {
                panic!("read-at panic");
            }
            .boxed()
        }
    }

    fn panicking_source() -> FileSegmentSource {
        let segments: Arc<[SegmentSpec]> = Arc::from([SegmentSpec {
            offset: 0,
            length: 4,
            alignment: Alignment::none(),
        }]);
        let metrics = DefaultMetricsRegistry::default();
        FileSegmentSource::open(
            segments,
            PanickingReadAt,
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        )
    }

    fn panic_message(payload: &(dyn Any + Send)) -> &str {
        payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic>")
    }

    #[tokio::test]
    #[should_panic(expected = "read-at panic")]
    async fn file_segment_source_propagates_read_driver_panic() {
        let source = panicking_source();
        let _result = source.request(SegmentId::from(0)).await;
    }

    // A read-driver panic must propagate on *every* run rather than sometimes surfacing as a
    // generic "dropped by runtime" error, which is nondeterministic.
    #[tokio::test]
    async fn file_segment_source_read_driver_panic_propagates_deterministically() {
        for i in 0..100 {
            let source = panicking_source();
            let outcome = AssertUnwindSafe(source.request(SegmentId::from(0)))
                .catch_unwind()
                .await;
            assert!(
                outcome.is_err(),
                "read-driver panic was not propagated on iteration {i}; \
                 the request resolved instead of panicking"
            );
        }
    }

    // A driver panic must surface to every concurrent reader sharing one driver: exactly one
    // reader re-raises the original panic, the rest report a graceful error, and none hang. This
    // also covers the fan-out invariant — `Shared` wakes every reader on completion, so a reader
    // dropped mid-flight can never swallow another reader's wake-up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn file_segment_source_panic_propagates_to_all_concurrent_readers() {
        let source = Arc::new(panicking_source());
        let handle = TokioRuntime::current();
        let reader_count = 8;

        let readers: Vec<_> = (0..reader_count)
            .map(|_| {
                let source = Arc::clone(&source);
                handle.spawn(async move {
                    AssertUnwindSafe(source.request(SegmentId::from(0)))
                        .catch_unwind()
                        .await
                })
            })
            .collect();

        let joined =
            tokio::time::timeout(std::time::Duration::from_secs(5), future::join_all(readers))
                .await
                .expect("a reader hung instead of observing the driver panic");

        let mut original_panics = 0;
        for reader in joined {
            match reader {
                // The first reader to observe completion re-raises the original panic.
                Err(payload) => {
                    assert!(
                        panic_message(&*payload).contains("read-at panic"),
                        "got: {:?}",
                        panic_message(&*payload)
                    );
                    original_panics += 1;
                }
                // Every other reader reports a graceful dropped-by-runtime error.
                Ok(result) => assert!(result.is_err(), "expected a dropped-by-runtime error"),
            }
        }

        assert_eq!(
            original_panics, 1,
            "exactly one reader should re-raise the original driver panic"
        );
    }

    #[derive(Clone)]
    struct ReadRangesOnly {
        calls: Arc<AtomicUsize>,
        max_batch: Arc<AtomicUsize>,
    }

    impl VortexReadAt for ReadRangesOnly {
        fn concurrency(&self) -> usize {
            4
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(16) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async { panic!("read_at should not be called") }.boxed()
        }

        fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> vortex_io::ReadAtStream {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.max_batch.fetch_max(requests.len(), Ordering::Relaxed);
            let results = requests
                .iter()
                .copied()
                .map(|request| {
                    let buffer = BufferHandle::new_host(
                        ByteBuffer::from(vec![0; request.length]).aligned(request.alignment),
                    );
                    (request, Ok(buffer))
                })
                .collect::<Vec<_>>();
            futures::stream::iter(results).boxed()
        }
    }

    #[tokio::test]
    async fn read_driver_batches_ready_requests() -> VortexResult<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let max_batch = Arc::new(AtomicUsize::new(0));
        let segments: Arc<[SegmentSpec]> = (0..4)
            .map(|i| SegmentSpec {
                offset: i * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let request_metrics = RequestMetrics::new(&metrics, vec![]);
        let source = FileSegmentSource::open(
            segments,
            ReadRangesOnly {
                calls: Arc::clone(&calls),
                max_batch: Arc::clone(&max_batch),
            },
            TokioRuntime::current(),
            request_metrics.clone(),
        );

        let results = future::join_all((0..4).map(|i| source.request(SegmentId::from(i)))).await;

        for result in results {
            assert_eq!(result?.len(), 4);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(max_batch.load(Ordering::Relaxed), 4);
        assert_eq!(request_metrics.read_ranges_calls.value(), 1);
        assert_eq!(request_metrics.read_ranges_multi.value(), 1);
        assert_eq!(request_metrics.read_ranges_num_ranges.count(), 1);
        assert_eq!(request_metrics.read_ranges_num_ranges.total(), 4.0);
        Ok(())
    }

    #[tokio::test]
    async fn read_driver_submits_partial_ranges_together() -> VortexResult<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let max_batch = Arc::new(AtomicUsize::new(0));
        let segments: Arc<[SegmentSpec]> = (0..6)
            .map(|i| SegmentSpec {
                offset: i * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let source = FileSegmentSource::open(
            segments,
            ReadRangesOnly {
                calls: Arc::clone(&calls),
                max_batch: Arc::clone(&max_batch),
            },
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        );

        let results = source.request_ranges(SegmentId::from(0), vec![0..1, 1..2, 2..3, 3..4]);
        for result in future::join_all(results).await {
            assert_eq!(result?.len(), 1);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(max_batch.load(Ordering::Relaxed), 4);
        Ok(())
    }

    #[derive(Clone)]
    struct ControlledReadRanges {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        batch_sizes: Arc<Mutex<Vec<usize>>>,
        permits: Arc<tokio::sync::Semaphore>,
    }

    impl VortexReadAt for ControlledReadRanges {
        fn concurrency(&self) -> usize {
            4
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(24) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async { panic!("read_at should not be called") }.boxed()
        }

        fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> vortex_io::ReadAtStream {
            self.batch_sizes.lock().push(requests.len());
            let active = self.active.fetch_add(requests.len(), Ordering::SeqCst) + requests.len();
            self.max_active.fetch_max(active, Ordering::SeqCst);

            let reads = requests
                .iter()
                .copied()
                .map(|request| {
                    let active = Arc::clone(&self.active);
                    let permits = Arc::clone(&self.permits);
                    async move {
                        let Ok(permit) = permits.acquire_owned().await else {
                            vortex_panic!("test semaphore unexpectedly closed");
                        };
                        permit.forget();
                        active.fetch_sub(1, Ordering::SeqCst);
                        let buffer = BufferHandle::new_host(
                            ByteBuffer::from(vec![0; request.length]).aligned(request.alignment),
                        );
                        (request, Ok(buffer))
                    }
                })
                .collect::<Vec<_>>();
            futures::stream::iter(reads).buffer_unordered(4).boxed()
        }
    }

    #[tokio::test]
    async fn read_driver_refills_global_concurrency_across_batches() -> VortexResult<()> {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let batch_sizes = Arc::new(Mutex::new(Vec::new()));
        let permits = Arc::new(tokio::sync::Semaphore::new(0));
        let segments: Arc<[SegmentSpec]> = (0..6)
            .map(|i| SegmentSpec {
                offset: i * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let source = FileSegmentSource::open(
            segments,
            ControlledReadRanges {
                active: Arc::clone(&active),
                max_active: Arc::clone(&max_active),
                batch_sizes: Arc::clone(&batch_sizes),
                permits: Arc::clone(&permits),
            },
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        );
        let reads = TokioRuntime::current().spawn(async move {
            future::join_all((0..6).map(|i| source.request(SegmentId::from(i)))).await
        });

        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while active.load(Ordering::SeqCst) != 4 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok()
        );

        permits.add_permits(1);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while batch_sizes.lock().len() < 2 || active.load(Ordering::SeqCst) != 4 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok()
        );
        assert_eq!(batch_sizes.lock().as_slice(), [4, 1]);
        assert_eq!(max_active.load(Ordering::SeqCst), 4);

        permits.add_permits(5);
        for result in reads.await {
            assert_eq!(result?.len(), 4);
        }
        assert_eq!(max_active.load(Ordering::SeqCst), 4);
        Ok(())
    }

    #[tokio::test]
    async fn read_driver_keeps_slots_full_while_a_straggler_is_in_flight() -> VortexResult<()> {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let batch_sizes = Arc::new(Mutex::new(Vec::new()));
        let permits = Arc::new(tokio::sync::Semaphore::new(0));
        let segments: Arc<[SegmentSpec]> = (0..8)
            .map(|i| SegmentSpec {
                offset: i * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let source = FileSegmentSource::open(
            segments,
            ControlledReadRanges {
                active: Arc::clone(&active),
                max_active: Arc::clone(&max_active),
                batch_sizes: Arc::clone(&batch_sizes),
                permits: Arc::clone(&permits),
            },
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        );
        let reads = TokioRuntime::current().spawn(async move {
            future::join_all((0..8).map(|i| source.request(SegmentId::from(i)))).await
        });

        wait_for_active_reads(&active, 4).await;

        // Complete three reads while leaving one original read blocked as a straggler. Each freed
        // slot must be refilled before the next completion; a batch-barrier implementation would
        // instead fall from four active reads to one and submit no replacement work.
        for expected_calls in 2..=4 {
            permits.add_permits(1);
            assert!(
                tokio::time::timeout(std::time::Duration::from_secs(1), async {
                    while batch_sizes.lock().len() < expected_calls
                        || active.load(Ordering::SeqCst) != 4
                    {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .is_ok()
            );
        }

        assert_eq!(batch_sizes.lock().as_slice(), [4, 1, 1, 1]);
        assert_eq!(active.load(Ordering::SeqCst), 4);
        assert_eq!(max_active.load(Ordering::SeqCst), 4);

        permits.add_permits(5);
        for result in reads.await {
            assert_eq!(result?.len(), 4);
        }
        Ok(())
    }

    async fn wait_for_active_reads(active: &AtomicUsize, expected: usize) {
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while active.load(Ordering::SeqCst) != expected {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok()
        );
    }

    #[derive(Clone)]
    struct SlowErrReadAt;

    impl VortexReadAt for SlowErrReadAt {
        fn concurrency(&self) -> usize {
            4
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(1024) }.boxed()
        }

        fn read_at(
            &self,
            offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async move {
                // Stagger completions so some reads finish while others are still in flight.
                for _ in 0..(offset as usize % 5 + 1) {
                    tokio::task::yield_now().await;
                }
                vortex_bail!("slow read done")
            }
            .boxed()
        }
    }

    // Many segment reads driven on separate tasks must all make progress and complete.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_driver_concurrent_reads_make_progress() {
        let n = 16u32;
        let segments: Arc<[SegmentSpec]> = (0..n)
            .map(|i| SegmentSpec {
                offset: u64::from(i) * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let source = Arc::new(FileSegmentSource::open(
            segments,
            SlowErrReadAt,
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        ));

        let handle = TokioRuntime::current();
        let tasks: Vec<_> = (0..n)
            .map(|i| {
                let source = Arc::clone(&source);
                handle.spawn(async move { source.request(SegmentId::from(i)).await })
            })
            .collect();

        let joined =
            tokio::time::timeout(std::time::Duration::from_secs(5), future::join_all(tasks)).await;

        assert!(joined.is_ok(), "concurrent reads stalled before completing");
    }
}
