// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
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
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
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
            metrics,
        )
        .boxed();

        let drive_fut = async move {
            stream
                .map(move |req| {
                    let reader = reader.clone();
                    async move {
                        let result = reader
                            .read_at(req.offset(), req.len(), req.alignment())
                            .await;
                        let result = result.and_then(|buffer| {
                            if req.len() != buffer.len() {
                                vortex_bail!(
                                    "FileSegmentSource: expected buffer of length {} but received {}. {:?}",
                                    req.len(),
                                    buffer.len(),
                                    req
                                )
                            }
                            Ok(buffer)
                        });

                        req.resolve(result);
                    }
                })
                .buffer_unordered(concurrency)
                .collect::<()>()
                .await
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
pub struct RequestMetrics {
    /// Number of individual segment requests observed by the driver.
    pub individual_requests: Counter,
    /// Number of physical reads after coalescing.
    pub coalesced_requests: Counter,
    /// Distribution of how many segment requests were merged into each physical read.
    pub num_requests_coalesced: Histogram,
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
                .add_labels(labels)
                .histogram("io.requests.coalesced.num_coalesced"),
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
    use vortex_io::runtime::tokio::TokioRuntime;
    use vortex_layout::segments::SegmentSource;
    use vortex_metrics::DefaultMetricsRegistry;

    use super::*;

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
