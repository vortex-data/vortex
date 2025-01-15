use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use futures::channel::oneshot;
use futures::Stream;
use futures_util::future::try_join_all;
use futures_util::{stream, StreamExt};
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::segments::SegmentId;

use crate::v2::footer::{FileLayout, Segment};
use crate::v2::io::IoDriver;
use crate::v2::segments::{SegmentCache, SegmentRequest};

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
    /// A segment cache to store segments in memory.
    pub segment_cache: Arc<dyn SegmentCache>,
}

#[derive(Debug)]
struct FileSegmentRequest {
    /// The segment ID.
    pub(crate) id: SegmentId,
    /// The segment location.
    pub(crate) location: Segment,
    /// The callback channel
    callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl FileSegmentRequest {
    fn resolve(self, buffer: VortexResult<ByteBuffer>) {
        self.callback
            .send(buffer)
            .map_err(|_| vortex_err!("send failed"))
            .vortex_expect("send failed");
    }
}

#[derive(Debug)]
struct CoalescedSegmentRequest {
    /// The range of the file to read.
    pub(crate) byte_range: Range<u64>,
    /// The original segment requests.
    pub(crate) requests: Vec<FileSegmentRequest>,
}

impl<R: VortexReadAt> IoDriver for FileIoDriver<R> {
    fn drive(
        &self,
        stream: impl Stream<Item = SegmentRequest> + 'static,
    ) -> impl Stream<Item = VortexResult<()>> + 'static {
        let segment_map = self.file_layout.segments.clone();
        let read = self.read.clone();
        let segment_cache1 = self.segment_cache.clone();
        let segment_cache2 = self.segment_cache.clone();

        stream
            // We map the segment requests to their respective locations within the file.
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
                        id: request.id,
                        location: location.clone(),
                        callback: request.callback,
                    })
                }
            })
            // We support zero-length segments (so layouts don't have to store this information)
            // If we encounter a zero-length segment, we can just resolve it now.
            .filter_map(move |request| async move {
                if request.location.length == 0 {
                    let alignment = request.location.alignment;
                    request.resolve(Ok(ByteBuffer::empty_aligned(alignment)));
                    None
                } else {
                    Some(request)
                }
            })
            // Check if the segment exists in the cache
            .filter_map(move |request| {
                let segment_cache = segment_cache1.clone();
                async move {
                    match segment_cache
                        .get(request.id, request.location.alignment)
                        .await
                    {
                        Ok(None) => Some(request),
                        Ok(Some(buffer)) => {
                            request.resolve(Ok(buffer));
                            None
                        }
                        Err(e) => {
                            request.resolve(Err(e));
                            None
                        }
                    }
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
            .map(move |request| evaluate(read.clone(), request, segment_cache2.clone()))
            // Buffer some number of concurrent I/O operations.
            .buffer_unordered(self.concurrency)
    }
}

async fn evaluate<R: VortexReadAt>(
    read: R,
    request: CoalescedSegmentRequest,
    segment_cache: Arc<dyn SegmentCache>,
) -> VortexResult<()> {
    log::debug!(
        "Reading byte range: {:?} {}",
        request.byte_range,
        request.byte_range.end - request.byte_range.start
    );
    let buffer: ByteBuffer = read
        .read_byte_range(
            request.byte_range.start,
            request.byte_range.end - request.byte_range.start,
        )
        .await
        .map_err(|e| vortex_err!("Failed to read coalesced segment: {:?} {:?}", request, e))?
        .into();

    // TODO(ngates): traverse the segment map to find un-requested segments that happen to
    //  fall within the range of the request. Then we can populate those in the cache.
    let mut cache_futures = Vec::with_capacity(request.requests.len());
    for req in request.requests {
        let offset = usize::try_from(req.location.offset - request.byte_range.start)?;
        let buf = buffer
            .slice(offset..offset + req.location.length as usize)
            .aligned(req.location.alignment);

        // Send the callback
        req.callback
            .send(Ok(buf.clone()))
            .map_err(|_| vortex_err!("send failed"))?;

        cache_futures.push(segment_cache.put(req.id, buf));
    }

    // Populate the cache
    try_join_all(cache_futures).await?;

    Ok(())
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
