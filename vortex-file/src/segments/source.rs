// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::BufferAllocator;
use vortex_io::VortexReadAt;
use vortex_io::runtime::Handle;
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
use crate::read::ReadRequestState;
use crate::read::RequestId;

#[derive(Debug)]
pub enum ReadEvent {
    Request(ReadRequest),
    Polled(RequestId),
    Dropped(RequestId),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IoRequestStats {
    pub registered: u64,
    pub polled: u64,
    pub dropped: u64,
    pub dispatched: u64,
    pub completed: u64,
    pub in_flight: u64,
    pub max_in_flight: u64,
}

static IO_REGISTERED: AtomicU64 = AtomicU64::new(0);
static IO_POLLED: AtomicU64 = AtomicU64::new(0);
static IO_DROPPED: AtomicU64 = AtomicU64::new(0);
static IO_DISPATCHED: AtomicU64 = AtomicU64::new(0);
static IO_COMPLETED: AtomicU64 = AtomicU64::new(0);
static IO_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
static IO_MAX_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

pub fn reset_io_request_stats() {
    IO_REGISTERED.store(0, Ordering::Relaxed);
    IO_POLLED.store(0, Ordering::Relaxed);
    IO_DROPPED.store(0, Ordering::Relaxed);
    IO_DISPATCHED.store(0, Ordering::Relaxed);
    IO_COMPLETED.store(0, Ordering::Relaxed);
    IO_IN_FLIGHT.store(0, Ordering::Relaxed);
    IO_MAX_IN_FLIGHT.store(0, Ordering::Relaxed);
}

pub fn io_request_stats() -> IoRequestStats {
    IoRequestStats {
        registered: IO_REGISTERED.load(Ordering::Relaxed),
        polled: IO_POLLED.load(Ordering::Relaxed),
        dropped: IO_DROPPED.load(Ordering::Relaxed),
        dispatched: IO_DISPATCHED.load(Ordering::Relaxed),
        completed: IO_COMPLETED.load(Ordering::Relaxed),
        in_flight: IO_IN_FLIGHT.load(Ordering::Relaxed),
        max_in_flight: IO_MAX_IN_FLIGHT.load(Ordering::Relaxed),
    }
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
pub struct FileSegmentSource {
    segments: Arc<[SegmentSpec]>,
    /// A queue for sending read request events to the I/O stream.
    events: mpsc::UnboundedSender<ReadEvent>,
    /// The next read request ID.
    next_id: Arc<AtomicUsize>,
}

impl FileSegmentSource {
    pub fn open<R: VortexReadAt + Clone>(
        segments: Arc<[SegmentSpec]>,
        reader: R,
        handle: Handle,
        metrics: RequestMetrics,
    ) -> Self {
        Self::open_with_allocator(segments, reader, handle, metrics, None)
    }

    pub fn open_with_allocator<R: VortexReadAt + Clone>(
        segments: Arc<[SegmentSpec]>,
        reader: R,
        handle: Handle,
        metrics: RequestMetrics,
        allocator: Option<Arc<dyn BufferAllocator>>,
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
        let allocator = allocator.clone();

        let drive_fut = async move {
            stream
                .map(move |req| {
                    let reader = reader.clone();
                    let allocator = allocator.clone();
                    async move {
                        IO_DISPATCHED.fetch_add(1, Ordering::Relaxed);
                        let in_flight = IO_IN_FLIGHT.fetch_add(1, Ordering::Relaxed) + 1;
                        let mut prev_max = IO_MAX_IN_FLIGHT.load(Ordering::Relaxed);
                        while in_flight > prev_max {
                            match IO_MAX_IN_FLIGHT.compare_exchange(
                                prev_max,
                                in_flight,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(next) => prev_max = next,
                            }
                        }

                        let result = if let Some(allocator) = allocator {
                            // Pipeline: start the I/O read and allocate the target
                            // buffer concurrently. try_join! polls read_fut first
                            // (initiating the HTTP request for object store sources),
                            // then polls alloc_fut (which may block on cuMemAllocHost
                            // for pool misses). This overlaps I/O first-byte latency
                            // with buffer allocation.
                            let read_fut = reader.read_at(req.offset(), req.len(), req.alignment());
                            let len = req.len();
                            let alignment = req.alignment();
                            let alloc_fut = async move { allocator.allocate(len, alignment) };

                            match futures::try_join!(read_fut, alloc_fut) {
                                Ok((data, mut target)) => {
                                    target
                                        .as_mut_slice()
                                        .copy_from_slice(data.as_host().as_ref());
                                    target.into_handle()
                                }
                                Err(e) => Err(e),
                            }
                        } else {
                            reader
                                .read_at(req.offset(), req.len(), req.alignment())
                                .await
                        };
                        req.resolve(result);
                        IO_COMPLETED.fetch_add(1, Ordering::Relaxed);
                        IO_IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
                    }
                })
                .buffer_unordered(concurrency)
                .collect::<()>()
                .await
        };

        handle.spawn(drive_fut).detach();

        Self {
            segments,
            events: send,
            next_id: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl SegmentSource for FileSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        // We eagerly register the read request here assuming the behaviour of [`FileRead`], where
        // coalescing becomes effective prior to the future being polled.
        let spec = match self.segments.get(*id as usize).cloned() {
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

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (request, state) = ReadRequest::new(id, offset, length as usize, alignment);
        let event = ReadEvent::Request(request);

        // If we fail to submit the event, we create a future that has failed.
        if let Err(e) = self.events.unbounded_send(event) {
            return future::ready(Err(vortex_err!("Failed to submit read request: {e}"))).boxed();
        }
        IO_REGISTERED.fetch_add(1, Ordering::Relaxed);

        let fut = ReadFuture {
            id,
            state,
            polled: false,
            events: self.events.clone(),
        };

        // One allocation: we only box the returned SegmentFuture, not the inner ReadFuture.
        fut.boxed()
    }
}

/// A future that resolves a read request from a [`FileRead`].
///
/// See the documentation for [`FileRead`] for details on coalescing and pre-fetching.
/// If dropped, the read request will be canceled where possible.
struct ReadFuture {
    id: usize,
    state: Arc<ReadRequestState>,
    polled: bool,
    events: mpsc::UnboundedSender<ReadEvent>,
}

impl Future for ReadFuture {
    type Output = VortexResult<BufferHandle>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.polled {
            self.polled = true;
            // Notify the I/O stream that this request has been polled.
            if let Err(e) = self.events.unbounded_send(ReadEvent::Polled(self.id)) {
                return Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}")));
            }
            IO_POLLED.fetch_add(1, Ordering::Relaxed);
        }

        self.state.poll_result(cx)
    }
}

impl Drop for ReadFuture {
    fn drop(&mut self) {
        self.state.close();
        // When the FileHandle is dropped, we can send a shutdown event to the I/O stream.
        // If the I/O stream has already been dropped, this will fail silently.
        if self
            .events
            .unbounded_send(ReadEvent::Dropped(self.id))
            .is_ok()
        {
            IO_DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub struct RequestMetrics {
    pub individual_requests: Counter,
    pub coalesced_requests: Counter,
    pub num_requests_coalesced: Histogram,
}

impl RequestMetrics {
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
