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
            // We support zero-length segments (so layouts don't have to store this information)
            // If we encounter a zero-length segment, we can just resolve it now.
            .filter_map(|request| async move {
                if request.location.length == 0 {
                    request
                        .callback
                        .send(Ok(ByteBuffer::empty_aligned(request.location.alignment)))
                        .map_err(|_| vortex_err!("send failed"))
                        .vortex_expect("send failed");
                    return None;
                }
                Some(request)
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
    log::warn!(
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
    const COALESCE: u64 = 1024 * 1024; // 1MB
    let fetch_ranges = merge_ranges(
        requests
            .iter()
            .map(|r| r.location.offset..r.location.offset + r.location.length as u64),
        COALESCE,
    );
    let mut coalesced = fetch_ranges
        .iter()
        .map(|range| CoalescedSegmentRequest {
            byte_range: range.clone(),
            requests: vec![],
        })
        .collect::<Vec<_>>();

    for req in requests {
        let idx = fetch_ranges.partition_point(|v| v.start <= req.location.offset) - 1;
        coalesced.as_mut_slice()[idx].requests.push(req);
    }

    coalesced
}

/// Returns a sorted list of ranges that cover `ranges`
///
/// From arrow-rs.
fn merge_ranges<R>(ranges: R, coalesce: u64) -> Vec<Range<u64>>
where
    R: IntoIterator<Item = Range<u64>>,
{
    let mut ranges: Vec<Range<u64>> = ranges.into_iter().collect();
    ranges.sort_unstable_by_key(|range| range.start);

    let mut ret = Vec::with_capacity(ranges.len());
    let mut start_idx = 0;
    let mut end_idx = 1;

    while start_idx != ranges.len() {
        let mut range_end = ranges[start_idx].end;

        while end_idx != ranges.len()
            && ranges[end_idx]
                .start
                .checked_sub(range_end)
                .map(|delta| delta <= coalesce)
                .unwrap_or(true)
        {
            range_end = range_end.max(ranges[end_idx].end);
            end_idx += 1;
        }

        let start = ranges[start_idx].start;
        let end = range_end;
        ret.push(start..end);

        start_idx = end_idx;
        end_idx += 1;
    }

    ret
}
