use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::stream::FuturesUnordered;
use futures_util::{stream, StreamExt, TryStreamExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::scan::unified::UnifiedDriverStream;
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::exec::ExecutionMode;
use crate::footer::{FileLayout, Segment};
use crate::segments::channel::SegmentChannel;
use crate::segments::SegmentCache;
use crate::{FileType, VortexOpenOptions};

/// A type of Vortex file that supports any [`VortexReadAt`] implementation.
///
/// This is a reasonable choice for files backed by a network since it performs I/O coalescing.
pub struct GenericVortexFile<R>(PhantomData<R>);

impl<R: VortexReadAt> FileType for GenericVortexFile<R> {
    type Options = GenericScanOptions;
    type Read = R;
    type ScanDriver = GenericScanDriver<R>;

    fn scan_driver(
        read: Self::Read,
        options: Self::Options,
        file_layout: FileLayout,
        segment_cache: Arc<dyn SegmentCache>,
    ) -> Self::ScanDriver {
        GenericScanDriver {
            read,
            options,
            file_layout,
            segment_cache,
            segment_channel: SegmentChannel::new(),
        }
    }
}

impl<R: VortexReadAt> VortexOpenOptions<GenericVortexFile<R>> {
    pub fn with_execution_mode(mut self, execution_mode: ExecutionMode) -> Self {
        self.options.execution_mode = execution_mode;
        self
    }

    pub fn with_execution_concurrency(mut self, execution_concurrency: usize) -> Self {
        self.options.execution_concurrency = execution_concurrency;
        self
    }

    pub fn with_io_concurrency(mut self, io_concurrency: usize) -> Self {
        self.options.io_concurrency = io_concurrency;
        self
    }
}

#[derive(Debug, Clone)]
pub struct GenericScanOptions {
    /// The number of concurrent splits to process.
    execution_concurrency: usize,
    execution_mode: ExecutionMode,
    /// The number of concurrent I/O requests to spawn.
    io_concurrency: usize,
}

impl Default for GenericScanOptions {
    fn default() -> Self {
        Self {
            execution_concurrency: 10,
            execution_mode: ExecutionMode::Inline,
            io_concurrency: 10,
        }
    }
}

pub struct GenericScanDriver<R> {
    read: R,
    options: GenericScanOptions,
    file_layout: FileLayout,
    segment_cache: Arc<dyn SegmentCache>,
    segment_channel: SegmentChannel,
}

impl<R: VortexReadAt> ScanDriver for GenericScanDriver<R> {
    type Options = GenericScanOptions;

    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        // This reader simply enqueues segment requests into the channel.
        // Our driver polls the other end of this channel to drive the I/O requests.
        self.segment_channel.reader()
    }

    fn drive_stream(
        self,
        stream: impl Stream<Item = BoxFuture<'static, VortexResult<()>>> + Send + 'static,
    ) -> impl Stream<Item = VortexResult<()>> + 'static {
        let exec_driver = self
            .options
            .execution_mode
            .into_driver(self.options.execution_concurrency);
        let exec_stream = exec_driver.drive(stream.boxed());

        // Now we set up the I/O stream to poll the other end of the segment channel.
        let io_stream = self.segment_channel.into_stream();

        // We map the segment requests to their respective locations within the file.
        let coalescing_window = self.read.performance_hint().coalescing_window();
        let segment_map = self.file_layout.segment_map().clone();
        let io_stream = io_stream.filter_map(move |request| {
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
        });

        // We support zero-length segments (so layouts don't have to store this information)
        // If we encounter a zero-length segment, we can just resolve it now.
        let io_stream = io_stream.filter_map(|request| async move {
            if request.location.length == 0 {
                let alignment = request.location.alignment;
                request.resolve(Ok(ByteBuffer::empty_aligned(alignment)));
                None
            } else {
                Some(request)
            }
        });

        // Check if the segment exists in the cache
        let segment_cache = self.segment_cache.clone();
        let io_stream = io_stream.filter_map(move |request| {
            let segment_cache = segment_cache.clone();
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
        });

        // Grab all available segment requests from the I/O queue so we get maximal visibility into
        // the requests for coalescing.
        // Note that we can provide a somewhat arbitrarily high capacity here since we're going to
        // deduplicate and coalesce. Meaning the resulting stream will at-most cover the entire
        // file and therefore be reasonably bounded.
        let io_stream = io_stream
            .ready_chunks(1024)
            .inspect(|requests| log::debug!("Processing {} segment requests", requests.len()));

        // Coalesce the segment requests to minimize the number of I/O operations.
        let io_stream = io_stream
            .map(move |r| coalesce(r, coalescing_window))
            .flat_map(stream::iter);

        // Submit the coalesced requests to the I/O.
        let read = self.read.clone();
        let segment_map = self.file_layout.segment_map().clone();
        let segment_cache = self.segment_cache.clone();
        let io_stream = io_stream.map(move |request| {
            let read = read.clone();
            let segment_map = segment_map.clone();
            let segment_cache = segment_cache.clone();
            async move {
                evaluate(
                    read.clone(),
                    request,
                    segment_map.clone(),
                    segment_cache.clone(),
                )
                .await
            }
        });

        // Buffer some number of concurrent I/O operations.
        let io_stream = io_stream.buffer_unordered(self.options.io_concurrency);

        // Finally, we unify the stream to drive both the CPU and I/O requests.
        UnifiedDriverStream {
            exec_stream,
            io_stream,
        }
    }
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
    /// The original segment requests, ordered by segment ID.
    pub(crate) requests: Vec<FileSegmentRequest>,
}

async fn evaluate<R: VortexReadAt>(
    read: R,
    request: CoalescedSegmentRequest,
    segment_map: Arc<[Segment]>,
    segment_cache: Arc<dyn SegmentCache>,
) -> VortexResult<()> {
    log::debug!(
        "Reading byte range: {:?} {}",
        request.byte_range,
        request.byte_range.end - request.byte_range.start,
    );
    let buffer: ByteBuffer = read
        .read_byte_range(request.byte_range.clone())
        .await?
        .into();

    // Figure out the segments covered by the read.
    let start = segment_map.partition_point(|s| s.offset < request.byte_range.start);
    let end = segment_map.partition_point(|s| s.offset < request.byte_range.end);

    // Note that we may have multiple requests for the same segment.
    let mut requests_iter = request.requests.into_iter().peekable();

    let cache_futures = FuturesUnordered::new();
    for (i, segment) in segment_map[start..end].iter().enumerate() {
        let segment_id = SegmentId::from(u32::try_from(i + start).vortex_expect("segment id"));
        let offset = usize::try_from(segment.offset - request.byte_range.start)?;
        let buf = buffer
            .slice(offset..offset + segment.length as usize)
            .aligned(segment.alignment);

        // Find any request callbacks and send the buffer
        while let Some(req) = requests_iter.peek() {
            // If the request is before the current segment, we should have already resolved it.
            match req.id.cmp(&segment_id) {
                Ordering::Less => {
                    // This should never happen, it means we missed a segment request.
                    vortex_panic!("Skipped segment request");
                }
                Ordering::Equal => {
                    // Resolve the request
                    requests_iter
                        .next()
                        .vortex_expect("next request")
                        .resolve(Ok(buf.clone()));
                }
                Ordering::Greater => {
                    // No request for this segment, so we continue
                    break;
                }
            }
        }

        cache_futures.push(segment_cache.put(segment_id, buf));
    }

    // Populate the cache
    cache_futures.try_collect::<()>().await?;

    Ok(())
}

/// TODO(ngates): outsource coalescing to a trait
fn coalesce(
    requests: Vec<FileSegmentRequest>,
    coalescing_window: u64,
) -> Vec<CoalescedSegmentRequest> {
    let fetch_ranges = merge_ranges(
        requests
            .iter()
            .map(|r| r.location.offset..r.location.offset + r.location.length as u64),
        coalescing_window,
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
        coalesced[idx].requests.push(req);
    }

    // Ensure we sort the requests by segment ID within the coalesced request.
    for req in coalesced.iter_mut() {
        req.requests.sort_unstable_by_key(|r| r.id);
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
