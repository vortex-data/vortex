use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{SinkExt, Stream, TryStreamExt};
use pin_project_lite::pin_project;
use vortex_array::ArrayData;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

pub trait Driver<T> {
    /// Returns a segment reader for the driver.
    fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static>;

    /// Drive the given stream of evaluation tasks.
    fn drive(&self, stream: BoxStream<BoxFuture<VortexResult<T>>>) -> BoxStream<VortexResult<T>>;
}

/// An evaluation driver that polls multiple evaluation tasks concurrently (using
/// [`futures::StreamExt::buffered`].
///
/// In order to provide the I/O driver with as much visibility as possible into the enqueued
/// segment requests, the driver passes all segment requests into a single stream, and polls it
/// alongside the evaluation stream.
pub struct ConcurrentDriver {
    execution_concurrency: usize,
    segment_concurrency: usize,

    request_send: mpsc::UnboundedSender<SegmentRequest>,
    request_recv: mpsc::UnboundedReceiver<SegmentRequest>,
}

impl ConcurrentDriver {
    pub fn new() -> Self {
        let (send, recv) = mpsc::unbounded();
        Self {
            execution_concurrency: 1,
            segment_concurrency: 1,
            request_send: send,
            request_recv: recv,
        }
    }

    /// Returns a reader for the unified driver that enqueues requests.
    pub fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static> {
        Arc::new(ConcurrentSegmentReader(self.request_send.clone()))
    }
}

impl Driver<ArrayData> for ConcurrentDriver {
    fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static> {
        Arc::new(ConcurrentSegmentReader(self.request_send.clone()))
    }

    fn drive(
        &self,
        stream: BoxStream<BoxFuture<VortexResult<ArrayData>>>,
    ) -> BoxStream<VortexResult<ArrayData>> {
        ConcurrentSegmentDriverStream {
            // TODO(ngates): we should buffer the evaluation stream, otherwise what's the point!
            evalation_driver: stream,
            segment_driver: self.request_recv.clone().map(|req| async move {
                let SegmentRequest { id, callback } = req;
                let bytes = ByteBuffer::new();
                callback
                    .send(bytes)
                    .map_err(|e| vortex_err!("Failed to send segment: {:?}", e))
            }),
        }
    }
}

struct SegmentRequest {
    // The ID of the requested segment
    id: SegmentId,
    // The one-shot channel to send the segment back to the caller
    callback: oneshot::Sender<ByteBuffer>,
}

/// An adapter struct that wraps the send-side of the segment request channel.
struct ConcurrentSegmentReader(mpsc::UnboundedSender<SegmentRequest>);

#[async_trait]
impl AsyncSegmentReader for ConcurrentSegmentReader {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        // Set up a channel to send the segment back to the caller.
        let (send, recv) = oneshot::channel();

        // TODO(ngates): attempt to resolve the segments from the cache before joining the
        //  request queue?

        // Send a request to the segment cache.
        self.0
            .clone()
            .send(SegmentRequest { id, callback: send })
            .await
            .map_err(|e| vortex_err!("Failed to request segment {:?}", e))?;

        // Await the callback
        recv.await
            .map_err(|cancelled| vortex_err!("segment read cancelled: {:?}", cancelled))
    }
}

pin_project! {
    /// A [`Stream`] that drives the unified segment source alongside polling the stream of
    /// evaluation tasks for completion.
    ///
    /// This is sort of like a `select!` implementation.
    struct ConcurrentSegmentDriverStream<R, S> {
        #[pin]
        evalation_driver: R,
        #[pin]
        segment_driver: S,
    }
}

impl<R, S> Stream for ConcurrentSegmentDriverStream<R, S>
where
    R: Stream<Item = VortexResult<ArrayData>>,
    S: Stream<Item = VortexResult<()>>,
{
    type Item = VortexResult<ArrayData>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            // If the row group driver is ready, then we can return the result.
            if let Poll::Ready(r) = this.evalation_driver.try_poll_next_unpin(cx) {
                return Poll::Ready(r);
            }
            // Otherwise, we try to poll the I/O driver.
            // If the I/O driver is not ready, then we return Pending and wait for I/
            // to wake up the driver.
            if matches!(this.segment_driver.as_mut().poll_next(cx), Poll::Pending) {
                return Poll::Pending;
            }
        }
    }
}
