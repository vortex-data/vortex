use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::{FutureExt, Stream};
use oneshot;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::segments::SegmentRequest;

/// The [`SegmentChannel`] is responsible for funnelling segment requests from each of the
/// evaluation threads into a single stream of segment requests.
///
/// Consumers of the stream can then choose how to buffer, debounce, coalesce, or otherwise manage
/// the requests, ultimately resolving them by sending the requested segment back to the caller
/// via the provided one-shot channel.
pub(crate) struct SegmentChannel {
    request_send: mpsc::UnboundedSender<SegmentRequest>,
    request_recv: mpsc::UnboundedReceiver<SegmentRequest>,
}

impl SegmentChannel {
    pub fn new() -> Self {
        let (send, recv) = mpsc::unbounded();
        Self {
            request_send: send,
            request_recv: recv,
        }
    }

    /// Returns a reader for the segment cache.
    pub fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static> {
        Arc::new(SegmentChannelReader(self.request_send.clone()))
    }

    /// Returns the stream of segment requests.
    pub fn into_stream(self) -> impl Stream<Item = SegmentRequest> {
        self.request_recv
    }
}

struct SegmentChannelReader(mpsc::UnboundedSender<SegmentRequest>);

impl AsyncSegmentReader for SegmentChannelReader {
    fn get(&self, id: SegmentId) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        // Set up a channel to send the segment back to the caller.
        let (send, recv) = oneshot::channel();

        // TODO(ngates): attempt to resolve the segments from the cache before joining the
        //  request queue.
        // Send a request to the segment channel.
        let channel = self.0.clone();

        SegmentFuture {
            future: async move {
                channel
                    .unbounded_send(SegmentRequest { id, callback: send })
                    .map_err(|e| vortex_err!("Failed to request segment {} {:?}", id, e))?;

                // Await the callback
                match recv.await {
                    Ok(result) => result,
                    Err(_canceled) => {
                        // The sender was dropped before returning a result to us
                        Err(vortex_err!("Segment request handler was dropped {}", id,))
                    }
                }
            }
            .boxed(),
            id,
            complete: false,
        }
        .boxed()
    }
}

pub struct SegmentFuture<F> {
    future: F,
    id: SegmentId,
    complete: bool,
}

impl<F> Future for SegmentFuture<F>
where
    F: Future<Output = VortexResult<ByteBuffer>> + Unpin,
{
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.future.poll_unpin(cx) {
            Poll::Ready(r) => {
                self.complete = true;
                Poll::Ready(r)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<F> Drop for SegmentFuture<F> {
    fn drop(&mut self) {}
}
