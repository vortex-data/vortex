// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
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
use crate::read::RequestId;

#[derive(Debug)]
pub enum ReadEvent {
    Request(ReadRequest),
    Polled(RequestId),
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
    recv: oneshot::AsyncReceiver<VortexResult<BufferHandle>>,
    polled: bool,
    finished: bool,
    events: mpsc::UnboundedSender<ReadEvent>,
}

impl Future for ReadFuture {
    type Output = VortexResult<BufferHandle>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.recv.poll_unpin(cx) {
            Poll::Ready(result) => {
                self.finished = true;
                // note: we are skipping polled and dropped events for this if the future
                //       is ready on the first poll, that means this request was completed
                //       before it was polled, as part of a coalesced request.
                Poll::Ready(
                    result.unwrap_or_else(|e| {
                        Err(vortex_err!("ReadRequest dropped by runtime: {e}"))
                    }),
                )
            }
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
