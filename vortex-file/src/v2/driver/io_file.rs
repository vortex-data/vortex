use std::ops::Range;
use std::sync::Arc;

use futures::channel::oneshot;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::{stream, FutureExt, StreamExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_io::VortexReadAt;

use crate::v2::driver::IoDriver;
use crate::v2::footer::Segment;
use crate::v2::segments::{SegmentCache, SegmentRequest};

pub struct FileIoDriver<R: VortexReadAt> {
    /// The file to read from.
    pub(crate) read: R,
    /// The map of segment locations within a file.
    /// TODO(ngates): wrap this up in its own data structure
    pub(crate) segment_map: Arc<[Segment]>,
    /// The segment cache.
    pub(crate) segment_cache: Arc<dyn SegmentCache>,
    /// The number of concurrent I/O requests to submit.
    pub(crate) concurrency: usize,
}

struct FileSegmentRequest {
    /// The segment location.
    location: Segment,
    /// The callback channel
    callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

struct CoalescedSegmentRequest {
    /// The range of the file to read.
    byte_range: Range<u64>,
    /// The original segment requests.
    requests: Vec<FileSegmentRequest>,
}

impl<R: VortexReadAt> IoDriver for FileIoDriver<R> {
    fn drive(
        &self,
        stream: BoxStream<'static, SegmentRequest>,
    ) -> BoxStream<'static, VortexResult<()>> {
        // First, we need to map the segment requests to their respective locations within the file.
        let stream = stream.filter_map(move |request| async move {
            let Some(location) = self.segment_map.get(*request.id as usize) else {
                request
                    .callback
                    .send(Err(vortex_err!("segment not found")))
                    .map_err(|_| vortex_err!("send failed"))
                    .vortex_expect("send failed");
                return None;
            };
            Some(FileSegmentRequest {
                location: location.clone(),
                callback: request.callback,
            })
        });

        // Grab all available segment requests from the I/O queue so we get maximal visibility into
        // the requests for coalescing.
        // Note that we can provide a somewhat arbitrarily high capacity here since we're going to
        // deduplicate and coalesce. Meaning the resulting stream will at-most cover the entire
        // file and therefore be reasonably bounded.
        let stream = stream.ready_chunks(1024);

        // Coalesce the segment requests to minimize the number of I/O operations.
        let stream = stream
            .map(|requests| coalesce(requests))
            .flat_map(|requests| stream::iter(requests.into_iter()));

        // Submit the coalesced requests to the I/O.
        let stream = stream.map(|request| self.evaluate(request).boxed());

        // Buffer some number of concurrent I/O operations.
        stream.buffer_unordered(self.concurrency).boxed()
    }
}

impl<R: VortexReadAt + Send> FileIoDriver<R> {
    fn evaluate(&self, request: CoalescedSegmentRequest) -> BoxFuture<'static, VortexResult<()>> {
        let read = self.read.clone();
        async move {
            // TODO(ngates): change VortexReadAt to support reading into a pre-allocated set of aligned
            //  buffers.
            let bytes = read
                .read_byte_range(request.byte_range.start, request.byte_range.end)
                .await?;

            // TODO(ngates): traverse the segment map to find un-requested segments that happen to
            //  fall within the range of the request. Then we can populate those in the cache.

            for req in request.requests {
                let offset = usize::try_from(req.location.offset - request.byte_range.start)?;
                let buffer = bytes.slice(offset..offset + req.location.length as usize);
                let buffer = ByteBuffer::from(buffer).aligned(req.location.alignment);
                req.callback
                    .send(Ok(buffer))
                    .map_err(|_| vortex_err!("send failed"))?;
            }

            Ok(())
        }
        .boxed()
    }
}

fn coalesce(requests: Vec<FileSegmentRequest>) -> Vec<CoalescedSegmentRequest> {
    requests
        .into_iter()
        .map(|req| CoalescedSegmentRequest {
            byte_range: req.location.offset..req.location.offset + req.location.length as u64,
            requests: vec![req],
        })
        .collect()
}
