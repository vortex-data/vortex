// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::BoxStream;
use futures::stream::Fuse;
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
use vortex_io::ReadAtStream;
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

type IoBatchStream = Fuse<BoxStream<'static, Vec<IoRequest>>>;

enum ReadRangeResultsState {
    Reading(ReadAtStream),
    Missing,
}

/// Matches streamed range results back to their logical requests.
struct ReadRangeResults {
    state: ReadRangeResultsState,
    remaining: Vec<Option<IoRequest>>,
}

impl ReadRangeResults {
    fn new(results: ReadAtStream, requests: Vec<IoRequest>) -> Self {
        Self {
            state: ReadRangeResultsState::Reading(results),
            remaining: requests.into_iter().map(Some).collect(),
        }
    }
}

impl Stream for ReadRangeResults {
    type Item = (IoRequest, VortexResult<BufferHandle>);

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                ReadRangeResultsState::Reading(results) => match results.poll_next_unpin(cx) {
                    Poll::Ready(Some((request, result))) => {
                        let Some(position) = self.remaining.iter().position(|req| {
                            req.as_ref().is_some_and(|req| {
                                req.offset() == request.offset
                                    && req.len() == request.length
                                    && req.alignment() == request.alignment
                            })
                        }) else {
                            tracing::warn!(?request, "reader returned an unknown range");
                            continue;
                        };
                        let req = self.remaining[position]
                            .take()
                            .vortex_expect("matched request is present");
                        return Poll::Ready(Some((req, result)));
                    }
                    Poll::Ready(None) => self.state = ReadRangeResultsState::Missing,
                    Poll::Pending => return Poll::Pending,
                },
                ReadRangeResultsState::Missing => {
                    let Some(req) = self.remaining.iter_mut().find_map(Option::take) else {
                        return Poll::Ready(None);
                    };
                    let error = vortex_err!(
                        "FileSegmentSource: read_ranges ended before resolving request. {:?}",
                        req
                    );
                    return Poll::Ready(Some((req, Err(error))));
                }
            }
        }
    }
}

/// Drives request batches while keeping the reader's concurrency slots occupied.
struct ReadDriver<R> {
    reader: Arc<R>,
    batches: IoBatchStream,
    pending: VecDeque<IoRequest>,
    reads: SelectAll<ReadRangeResults>,
    num_active: usize,
    batches_done: bool,
    concurrency: usize,
    metrics: RequestMetrics,
}

impl<R: VortexReadAt> ReadDriver<R> {
    fn new(
        reader: R,
        batches: BoxStream<'static, Vec<IoRequest>>,
        concurrency: usize,
        metrics: RequestMetrics,
    ) -> Self {
        Self {
            reader: Arc::new(reader),
            batches: batches.fuse(),
            pending: VecDeque::new(),
            reads: SelectAll::new(),
            num_active: 0,
            batches_done: false,
            concurrency,
            metrics,
        }
    }

    fn submit_pending(&mut self) {
        while self.num_active < self.concurrency && !self.pending.is_empty() {
            let batch_len = (self.concurrency - self.num_active).min(self.pending.len());
            let reqs = self.pending.drain(..batch_len).collect::<Vec<_>>();
            self.num_active += batch_len;

            self.metrics.read_ranges_calls.add(1);
            self.metrics.read_ranges_num_ranges.update(batch_len as f64);
            if batch_len > 1 {
                self.metrics.read_ranges_multi.add(1);
            }
            tracing::trace!(
                target: "vortex_file::read_ranges",
                num_ranges = batch_len,
                num_active = self.num_active,
                "submitting positional read batch"
            );

            let requests = reqs
                .iter()
                .map(|req| ReadAtRequest::new(req.offset(), req.len(), req.alignment()))
                .collect::<Vec<_>>()
                .into();
            let results = self.reader.read_ranges(requests);
            self.reads.push(ReadRangeResults::new(results, reqs));
        }
    }
}

impl<R: VortexReadAt> Stream for ReadDriver<R> {
    type Item = ();

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();

        // Observe every batch already available so submission can fill all free slots at once.
        if !this.batches_done {
            loop {
                match this.batches.poll_next_unpin(cx) {
                    Poll::Ready(Some(batch)) => this.pending.extend(batch),
                    Poll::Ready(None) => {
                        this.batches_done = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }
        }

        this.submit_pending();

        if this.batches_done && this.num_active == 0 {
            return Poll::Ready(None);
        }

        match this.reads.poll_next_unpin(cx) {
            Poll::Ready(Some((req, result))) => {
                this.num_active -= 1;
                let result = validate_read_result(&req, result);
                req.resolve(result);
                Poll::Ready(Some(()))
            }
            Poll::Ready(None) if this.num_active == 0 => Poll::Pending,
            Poll::Ready(None) => {
                vortex_panic!("read result streams ended with active requests")
            }
            Poll::Pending => Poll::Pending,
        }
    }
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
            concurrency,
            metrics.clone(),
        )
        .boxed();

        let drive_fut = ReadDriver::new(reader, stream, concurrency, metrics).collect::<()>();

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
        }
    }
}

impl SegmentSource for FileSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        // We eagerly register the read request here assuming the behaviour of [`FileSegmentSource`], where
        // coalescing becomes effective prior to the future being polled.
        let spec = *match self.segments.get(*id as usize) {
            Some(spec) => spec,
            None => {
                return future::ready(Err(vortex_err!("Missing segment: {}", id))).boxed();
            }
        };

        let SegmentSpec {
            offset,
            length,
            alignment,
        } = spec;

        let (send, recv) = oneshot::channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let event = ReadEvent::Request(ReadRequest {
            id,
            offset,
            length: length as usize,
            alignment,
            callback: send,
        });

        // If we fail to submit the event, we create a future that has failed.
        if let Err(e) = self.events.unbounded_send(event) {
            return future::ready(Err(vortex_err!("Failed to submit read request: {e}"))).boxed();
        }

        let fut = ReadFuture {
            id,
            recv: recv.into_future(),
            polled: false,
            finished: false,
            events: self.events.clone(),
            driver: self.driver.clone(),
            driver_panic: Arc::clone(&self.driver_panic),
        };

        // One allocation: we only box the returned SegmentFuture, not the inner ReadFuture.
        fut.boxed()
    }
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
                // Notify the I/O stream that this request has been polled.
                match self.events.unbounded_send(ReadEvent::Polled(self.id)) {
                    Ok(()) => Poll::Pending,
                    Err(e) => Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}"))),
                }
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
    /// Number of calls made to [`VortexReadAt::read_ranges`].
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
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let spec = match self.segments.get(*id as usize) {
            Some(spec) => spec,
            None => {
                return future::ready(Err(vortex_err!("Missing segment: {}", id))).boxed();
            }
        };

        let start = spec.offset as usize;
        let end = start + spec.length as usize;
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

        let slice = self.buffer.slice(start..end).aligned(spec.alignment);
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

    fn io_request(id: RequestId, offset: u64, length: usize) -> IoRequest {
        let (callback, _receiver) = oneshot::channel();
        IoRequest::new_single(ReadRequest {
            id,
            offset,
            length,
            alignment: Alignment::none(),
            callback,
        })
    }

    #[tokio::test]
    async fn read_range_results_matches_results_and_reports_missing_requests() {
        let requests = vec![io_request(0, 0, 4), io_request(1, 4, 4)];
        let unknown = ReadAtRequest::new(8, 4, Alignment::none());
        let returned = ReadAtRequest::new(4, 4, Alignment::none());
        let buffer = BufferHandle::new_host(ByteBuffer::from(vec![0; 4]));
        let results =
            futures::stream::iter([(unknown, Ok(buffer.clone())), (returned, Ok(buffer))]).boxed();

        let resolved = ReadRangeResults::new(results, requests)
            .collect::<Vec<_>>()
            .await;

        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].0.offset(), 4);
        assert!(resolved[0].1.is_ok());
        assert_eq!(resolved[1].0.offset(), 0);
        assert!(resolved[1].1.is_err());
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

        fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
            self.calls.fetch_add(1, Ordering::Relaxed);
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
            },
            TokioRuntime::current(),
            request_metrics.clone(),
        );

        let results = future::join_all((0..4).map(|i| source.request(SegmentId::from(i)))).await;

        for result in results {
            assert_eq!(result?.len(), 4);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(request_metrics.read_ranges_calls.value(), 1);
        assert_eq!(request_metrics.read_ranges_multi.value(), 1);
        assert_eq!(request_metrics.read_ranges_num_ranges.count(), 1);
        assert_eq!(request_metrics.read_ranges_num_ranges.total(), 4.0);
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

        fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
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
