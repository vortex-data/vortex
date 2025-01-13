use std::future::Future;
use std::ops::Range;

use futures::channel::oneshot;
use futures::Stream;
use futures_util::{stream, StreamExt};
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_io::VortexReadAt;

use crate::v2::footer::{FileLayout, Segment};
use crate::v2::io::IoDriver;
use crate::v2::segments::SegmentRequest;

// TODO(ngates): use this sort of trait for I/O?
#[allow(dead_code)]
pub trait RangeReader {
    fn read_range(
        &self,
        range: Range<u64>,
        buffer: &mut ByteBufferMut,
    ) -> impl Future<Output = VortexResult<()>> + 'static;
}

/// An I/O driver for reading segments from a file.
///
/// This driver includes functionality for coalescing requests to minimize the number of I/O
/// operations, as well as executing multiple I/O operations concurrently.
pub struct FileIoDriver<R: VortexReadAt> {
    /// The file to read from.
    pub read: R,
    /// The file layout
    pub file_layout: FileLayout,
    /// The number of concurrent I/O requests to submit.
    pub concurrency: usize,
}

#[derive(Debug)]
struct FileSegmentRequest {
    /// The segment location.
    pub(crate) location: Segment,
    /// The callback channel
    pub(crate) callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

#[derive(Debug)]
struct CoalescedSegmentRequest {
    /// The range of the file to read.
    pub(crate) byte_range: Range<u64>,
    /// The original segment requests.
    pub(crate) requests: Vec<FileSegmentRequest>,
}

impl CoalescedSegmentRequest {
    /// Resolve the coalesced segment request.
    fn resolve(self, buffer: ByteBuffer) -> VortexResult<()> {
        for req in self.requests {
            let offset = usize::try_from(req.location.offset - self.byte_range.start)?;
            req.callback
                .send(Ok(buffer
                    .slice(offset..offset + req.location.length as usize)
                    .aligned(req.location.alignment)))
                .map_err(|_| vortex_err!("send failed"))?;
        }
        Ok(())
    }
}

impl<R: VortexReadAt> IoDriver for FileIoDriver<R> {
    fn drive(
        &self,
        stream: impl Stream<Item = SegmentRequest> + 'static,
    ) -> impl Stream<Item = VortexResult<()>> + 'static {
        let segment_map = self.file_layout.segments.clone();
        let read = self.read.clone();

        // First, we need to map the segment requests to their respective locations within the file.
        stream
            .filter_map(move |request| {
                let segment_map = segment_map.clone();
                async move {
                    let Some(location) = segment_map.get(*request.id as usize) else {
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
                }
            })
            // Grab all available segment requests from the I/O queue so we get maximal visibility into
            // the requests for coalescing.
            // Note that we can provide a somewhat arbitrarily high capacity here since we're going to
            // deduplicate and coalesce. Meaning the resulting stream will at-most cover the entire
            // file and therefore be reasonably bounded.
            .ready_chunks(1024)
            // Coalesce the segment requests to minimize the number of I/O operations.
            .map(coalesce)
            .flat_map(stream::iter)
            // Submit the coalesced requests to the I/O.
            .map(move |request| {
                let read = read.clone();
                evaluate(read, request)
            })
            // Buffer some number of concurrent I/O operations.
            .buffer_unordered(self.concurrency)
    }
}

async fn evaluate<R: VortexReadAt>(read: R, request: CoalescedSegmentRequest) -> VortexResult<()> {
    log::trace!(
        "Reading byte range: {:?} {}",
        request.byte_range,
        request.byte_range.end - request.byte_range.start
    );
    let bytes = read
        .read_byte_range(
            request.byte_range.start,
            request.byte_range.end - request.byte_range.start,
        )
        .await
        .map_err(|e| vortex_err!("Failed to read coalesced segment: {:?} {:?}", request, e))?;

    // TODO(ngates): traverse the segment map to find un-requested segments that happen to
    //  fall within the range of the request. Then we can populate those in the cache.
    request.resolve(ByteBuffer::from(bytes))
}

/// TODO(ngates): outsource coalescing to a trait
fn coalesce(requests: Vec<FileSegmentRequest>) -> Vec<CoalescedSegmentRequest> {
    requests
        .into_iter()
        .map(|req| CoalescedSegmentRequest {
            byte_range: req.location.offset..req.location.offset + req.location.length as u64,
            requests: vec![req],
        })
        .collect()
}
