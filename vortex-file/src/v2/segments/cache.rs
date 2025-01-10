//! The segment reader provides an async interface to layouts for resolving individual segments.

use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::Stream;
use futures_util::{stream, SinkExt, StreamExt, TryFutureExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::v2::footer::Segment;

pub(crate) struct SegmentCache<R> {
    read: R,
    segments: Arc<[Segment]>,
    request_send: mpsc::UnboundedSender<SegmentRequest>,
    request_recv: mpsc::UnboundedReceiver<SegmentRequest>,
}

struct SegmentRequest {
    // The ID of the requested segment
    id: SegmentId,
    // The one-shot channel to send the segment back to the caller
    callback: oneshot::Sender<ByteBuffer>,
}

impl<R> SegmentCache<R> {
    pub fn new(read: R, segments: Arc<[Segment]>) -> Self {
        let (send, recv) = mpsc::unbounded();
        Self {
            read,
            segments,
            request_send: send,
            request_recv: recv,
        }
    }

    pub fn set(&mut self, _segment_id: SegmentId, _bytes: ByteBuffer) -> VortexResult<()> {
        // Do nothing for now
        Ok(())
    }

    /// Returns a reader for the segment cache.
    pub fn reader(&self) -> Arc<dyn AsyncSegmentReader + 'static> {
        Arc::new(SegmentCacheReader(self.request_send.clone()))
    }
}

impl<R: VortexReadAt + Unpin> SegmentCache<R> {
    /// Drives the segment cache.
    pub(crate) fn driver(
        self,
    ) -> impl Stream<Item = impl Future<Output = VortexResult<()>>> + 'static {
        self.request_recv
            // The more chunks we grab, the better visibility we have to perform coalescing.
            // Since we know this stream is finite (number of segments in the file), then we
            // can just shove in a very high capacity. Rest assured the internal Vec is not
            // pre-allocated with this capacity.
            .ready_chunks(100_000)
            // TODO(ngates): now we should flat_map the requests to split them into coalesced
            //  read operations.
            .flat_map(stream::iter)
            .map(move |request| {
                let read = self.read.clone();
                let segments = self.segments.clone();
                async move {
                    let segment = &segments[*request.id as usize];
                    let bytes = read
                        .read_byte_range(segment.offset, segment.length as u64)
                        .map_ok(|bytes| ByteBuffer::from(bytes).aligned(segment.alignment))
                        .await?;
                    request
                        .callback
                        .send(bytes)
                        .map_err(|_| vortex_err!("receiver dropped"))?;
                    Ok(())
                }
            })
    }
}

struct SegmentCacheReader(mpsc::UnboundedSender<SegmentRequest>);

#[async_trait]
impl AsyncSegmentReader for SegmentCacheReader {
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
            .map_err(|e| vortex_err!("Failed to request segment {:?}", e))?;

        // Await the callback
        recv.await
            .map_err(|cancelled| vortex_err!("segment read cancelled: {:?}", cancelled))
    }
}
