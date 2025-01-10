//! The segment reader provides an async interface to layouts for resolving individual segments.

use std::future::Future;
use std::sync::{Arc, RwLock};
use std::task::Poll;

use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::Stream;
use futures_util::{stream, SinkExt, StreamExt, TryFutureExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::v2::footer::Segment;

#[derive(Clone)]
pub(crate) struct SegmentCache<R> {
    read: R,
    segments: Arc<[Segment]>,
    request_send: mpsc::UnboundedSender<SegmentRequest>,
    request_recv: Arc<RwLock<mpsc::UnboundedReceiver<SegmentRequest>>>,
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
            request_recv: Arc::new(RwLock::new(recv)),
        }
    }

    pub fn set(&mut self, _segment_id: SegmentId, _bytes: ByteBuffer) -> VortexResult<()> {
        // Do nothing for now
        Ok(())
    }
}

impl<R: VortexReadAt + Unpin> SegmentCache<R> {
    /// Drives the segment cache.
    pub(crate) fn driver(
        self,
    ) -> impl Stream<Item = impl Future<Output = VortexResult<()>>> + 'static {
        stream::poll_fn(move |cx| {
            // First we drain the request channel to see if there are any new segments to read.
            let mut recv = self
                .request_recv
                .write()
                .map_err(|_| vortex_err!("failed to acquire read lock"))
                .vortex_expect("lock poisoned");
            let mut requests = Vec::with_capacity(recv.size_hint().0);
            loop {
                match recv.poll_next_unpin(cx) {
                    Poll::Ready(Some(req)) => requests.push(req),
                    Poll::Ready(None) => vortex_panic!("Unexpected end of request channel"),
                    Poll::Pending => {
                        break;
                    }
                }
            }
            Poll::Ready(Some(requests))
        })
        // TODO(ngates): now we should flat_map the requests to split them into coalesced
        //  read operations.
        .flat_map(|requests| stream::iter(requests))
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

#[async_trait]
impl<R: VortexReadAt> AsyncSegmentReader for SegmentCache<R> {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        // Set up a channel to send the segment back to the caller.
        let (send, recv) = oneshot::channel();

        // TODO(ngates): attempt to resolve the segments from the cache before joining the
        //  request queue.

        // Send a request to the segment cache.
        self.request_send
            .clone()
            .send(SegmentRequest { id, callback: send })
            .await
            .map_err(|e| vortex_err!("Failed to request segment {:?}", e))?;

        // Await the callback
        recv.await
            .map_err(|cancelled| vortex_err!("segment read cancelled: {:?}", cancelled))
    }
}
