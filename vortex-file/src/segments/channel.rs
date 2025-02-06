use std::sync::Arc;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::Stream;
use futures_util::SinkExt;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
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

#[async_trait]
impl AsyncSegmentReader for SegmentChannelReader {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        // Set up a channel to send the segment back to the caller.
        let (send, recv) = oneshot::channel();

        // TODO(ngates): attempt to resolve the segments from the cache before joining the
        //  request queue.

        // Send a request to the segment cache.
        self.0
            .clone()
            .send(SegmentRequest { id, callback: send })
            .await
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
}
