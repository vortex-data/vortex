use std::cmp::Ordering;
use std::future;
use std::marker::PhantomData;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dashmap::{DashMap, Entry};
use futures::stream::FuturesUnordered;
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, select, stream};
use moka::future::CacheBuilder;
use pin_project_lite::pin_project;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_io::VortexReadAt;
use vortex_layout::instrument;
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::{AsyncSegmentReader, SegmentEvent, SegmentId, SegmentStream};
use vortex_metrics::{Counter, VortexMetrics};

use crate::footer::{Footer, Segment};
use crate::segments::channel::SegmentChannel;
use crate::segments::{InMemorySegmentCache, SegmentCache};
use crate::{FileType, VortexOpenOptions};

/// A type of Vortex file that supports any [`VortexReadAt`] implementation.
///
/// This is a reasonable choice for files backed by a network since it performs I/O coalescing.
pub struct GenericVortexFile<R>(PhantomData<R>);

impl<R: VortexReadAt> VortexOpenOptions<GenericVortexFile<R>> {
    const INITIAL_READ_SIZE: u64 = 1 << 20; // 1 MB

    pub fn file(read: R) -> Self {
        Self::new(read, Default::default())
            .with_segment_cache(Arc::new(InMemorySegmentCache::new(
                // For now, use a fixed 1GB overhead.
                CacheBuilder::new(1 << 30),
            )))
            .with_initial_read_size(Self::INITIAL_READ_SIZE)
    }
}

impl<R: VortexReadAt> FileType for GenericVortexFile<R> {
    type Options = GenericScanOptions;
    type Read = R;
    type ScanDriver = GenericScanDriver<R>;

    fn scan_driver(
        read: Self::Read,
        options: Self::Options,
        footer: Footer,
        segment_cache: Arc<dyn SegmentCache>,
        metrics: VortexMetrics,
    ) -> Self::ScanDriver {
        GenericScanDriver {
            read,
            options,
            footer,
            segment_cache,
            segment_channel: SegmentChannel::new(),
            metrics: metrics.into(),
        }
    }
}

impl<R: VortexReadAt> VortexOpenOptions<GenericVortexFile<R>> {
    pub fn with_io_concurrency(mut self, io_concurrency: usize) -> Self {
        self.options.io_concurrency = io_concurrency;
        self
    }
}

#[derive(Clone)]
pub struct GenericScanOptions {
    /// The number of concurrent I/O requests to spawn.
    /// This should be smaller than execution concurrency for coalescing to occur.
    io_concurrency: usize,
}

impl Default for GenericScanOptions {
    fn default() -> Self {
        Self { io_concurrency: 16 }
    }
}

pub struct GenericScanDriver<R> {
    read: R,
    options: GenericScanOptions,
    footer: Footer,
    segment_cache: Arc<dyn SegmentCache>,
    segment_channel: SegmentChannel,
    metrics: CoalescingMetrics,
}

struct InflightSegment {
    cancel_callback: oneshot::Sender<()>,
    completion_callback: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,
}

#[derive(Default, Clone)]
struct InflightSegments(Arc<DashMap<SegmentId, InflightSegment>>);

impl InflightSegments {
    pub fn insert_or_register_callback(
        &self,
        segment_id: SegmentId,
        completion_callback: Option<oneshot::Sender<VortexResult<ByteBuffer>>>,
    ) -> Option<oneshot::Receiver<()>> {
        match self.0.entry(segment_id) {
            Entry::Occupied(mut entry) => {
                if let Some(cb) = completion_callback {
                    entry.get_mut().completion_callback.replace(cb);
                }
                None
            }
            Entry::Vacant(entry) => {
                let (cancel_tx, cancel_rx) = oneshot::channel();
                entry.insert(InflightSegment {
                    cancel_callback: cancel_tx,
                    completion_callback,
                });
                Some(cancel_rx)
            }
        }
    }

    pub fn cancel(&self, segment_id: SegmentId) {
        if let Some((
            _,
            InflightSegment {
                cancel_callback, ..
            },
        )) = self.0.remove(&segment_id)
        {
            let _ = cancel_callback.send(());
        }
    }

    pub fn complete(&self, segment_id: SegmentId, value: VortexResult<ByteBuffer>) {
        if let Some((
            _,
            InflightSegment {
                completion_callback: Some(cb),
                ..
            },
        )) = self.0.remove(&segment_id)
        {
            cb.send(value)
                .map_err(|_| vortex_err!("send failed"))
                .vortex_expect("send failed");
        }
    }
}

impl<R: VortexReadAt> GenericScanDriver<R> {}

impl<R: VortexReadAt> ScanDriver for GenericScanDriver<R> {
    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        self.segment_channel.reader()
    }

    fn io_stream(self, segments: SegmentStream) -> impl Stream<Item = VortexResult<()>> {
        let segment_requests = self.segment_channel.into_stream();
        let segment_map = self.footer.segment_map().clone();
        let inflight_segments = InflightSegments::default();

        let inflight = inflight_segments.clone();
        let segment_requests = segment_requests.filter_map(move |request| {
            let Some(location) = segment_map.get(*request.id as usize) else {
                request.resolve(Err(vortex_err!("segment not found")));
                return future::ready(None);
            };

            // We support zero-length segments (so layouts don't have to store this information)
            // If we encounter a zero-length segment, we can just resolve it now.
            if location.length == 0 {
                let alignment = location.alignment;
                request.resolve(Ok(ByteBuffer::empty_aligned(alignment)));
                return future::ready(None);
            }

            let cancel_handle =
                inflight.insert_or_register_callback(request.id, Some(request.callback));
            future::ready(
                cancel_handle
                    .map(|handle| SegmentRequest::new(request.id, location.clone(), handle)),
            )
        });

        let inflight = inflight_segments.clone();
        let segment_map = self.footer.segment_map().clone();
        let prefetch_stream = segments.filter_map(move |event| match event {
            SegmentEvent::Cancel(id) => {
                inflight.cancel(id);
                future::ready(None)
            }
            SegmentEvent::Request(id) => {
                future::ready(segment_map.get(*id as usize).and_then(|location| {
                    inflight
                        .insert_or_register_callback(id, None)
                        .map(|cancel_handle| {
                            SegmentRequest::new(id, location.clone(), cancel_handle)
                        })
                }))
            }
        });

        // Check if the segment exists in the cache
        let segment_cache = self.segment_cache.clone();
        let inflight = inflight_segments.clone();
        let segment_requests = segment_requests.filter_map(move |request| {
            filter_with_cache(request, segment_cache.clone(), inflight.clone())
        });
        let segment_cache = self.segment_cache.clone();
        let inflight = inflight_segments.clone();
        let prefetch_stream = prefetch_stream.filter_map(move |request| {
            filter_with_cache(request, segment_cache.clone(), inflight.clone())
        });

        // Grab all available segment requests from the I/O queue so we get maximal visibility into
        // the requests for coalescing.
        // Note that we can provide a somewhat arbitrarily high capacity here since we're going to
        // deduplicate and coalesce. Meaning the resulting stream will at-most cover the entire
        // file and therefore be reasonably bounded.
        // Coalesce the segment requests to minimize the number of I/O operations.
        let perf_hint = self.read.performance_hint();
        let io_stream = SegmentRequestStream {
            requests: segment_requests,
            prefetch: prefetch_stream,
            requests_ready_chunks: 1024,
            prefetch_ready_chunks: self.options.io_concurrency,
        }
        .map(move |r| coalesce(r, perf_hint.coalescing_window(), perf_hint.max_read()))
        .flat_map(stream::iter)
        .inspect(move |(coalesced, _)| self.metrics.record(coalesced));

        // Submit the coalesced requests to the I/O.
        let read = self.read.clone();
        let segment_map = self.footer.segment_map().clone();
        let segment_cache = self.segment_cache.clone();
        let io_stream = io_stream.map(move |(request, cancellation_handle)| {
            let read = read.clone();
            let segment_map = segment_map.clone();
            let segment_cache = segment_cache.clone();
            let inflight = inflight_segments.clone();
            async move {
                select! {
                    _ = cancellation_handle.cancelled().fuse() => Ok(()),
                    evaluated = evaluate(
                        read.clone(),
                        request,
                        segment_map.clone(),
                        segment_cache.clone(),
                        inflight.clone(),

                    ).fuse() => evaluated,
                }
            }
        });

        // Buffer some number of concurrent I/O operations.
        instrument!(
            "io_stream",
            io_stream.buffer_unordered(self.options.io_concurrency)
        )
    }
}

pin_project! {
    struct SegmentRequestStream<Requests, Prefetch> {
        #[pin]
        pub requests: Requests,
        #[pin]
        pub prefetch: Prefetch,
        requests_ready_chunks: usize,
        prefetch_ready_chunks: usize,
    }
}

impl<Requests, Prefetch> Stream for SegmentRequestStream<Requests, Prefetch>
where
    Requests: Stream<Item = SegmentRequest>,
    Prefetch: Stream<Item = SegmentRequest>,
{
    type Item = Vec<SegmentRequest>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        let mut items = Vec::with_capacity(*this.requests_ready_chunks);
        let requests_ended = loop {
            match this.requests.as_mut().poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    items.push(item);
                    if items.len() >= *this.requests_ready_chunks {
                        return Poll::Ready(Some(items));
                    }
                }
                Poll::Pending => break false,
                Poll::Ready(None) => break true,
            }
        };

        let prefetch_limit =
            (items.len() + *this.prefetch_ready_chunks).min(*this.requests_ready_chunks);

        let prefetch_ended = loop {
            match this.prefetch.as_mut().poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    items.push(item);
                    if items.len() >= prefetch_limit {
                        return Poll::Ready(Some(items));
                    }
                }
                Poll::Pending => break false,
                Poll::Ready(None) => break true,
            }
        };

        if requests_ended && prefetch_ended {
            return Poll::Ready(None);
        }
        if items.is_empty() {
            Poll::Pending
        } else {
            Poll::Ready(Some(items))
        }
    }
}

#[derive(Debug)]
struct SegmentRequest {
    id: SegmentId,
    location: Segment,
    cancel_handle: oneshot::Receiver<()>,
}

impl SegmentRequest {
    fn new(id: SegmentId, location: Segment, cancel_handle: oneshot::Receiver<()>) -> Self {
        Self {
            id,
            location,
            cancel_handle,
        }
    }
    fn range(&self) -> Range<u64> {
        self.location.offset..self.location.offset + self.location.length as u64
    }
}

impl From<(SegmentId, Segment, oneshot::Receiver<()>)> for SegmentRequest {
    fn from(value: (SegmentId, Segment, oneshot::Receiver<()>)) -> Self {
        let (id, location, cancel_handle) = value;
        SegmentRequest {
            id,
            location,
            cancel_handle,
        }
    }
}

async fn filter_with_cache(
    request: SegmentRequest,
    cache: Arc<dyn SegmentCache>,
    inflight_segments: InflightSegments,
) -> Option<SegmentRequest> {
    match cache.get(request.id, request.location.alignment).await {
        Ok(None) => {}
        Ok(Some(buffer)) => {
            inflight_segments.complete(request.id, Ok(buffer));
            return None;
        }
        Err(e) => {
            inflight_segments.complete(request.id, Err(e));
            return None;
        }
    };
    // not in cache
    Some(request)
}

#[derive(Debug)]
struct CoalescedSegmentRequest {
    /// The alignment of the first segment.
    // TODO(ngates): is this the best alignment to use?
    pub(crate) alignment: Alignment,
    /// The range of the file to read.
    pub(crate) byte_range: Range<u64>,
    /// The original segment requests, ordered by segment ID.
    pub(crate) requests: Vec<(SegmentId, Segment)>,
}

#[derive(Default)]
struct CoalescedCancellationHandle(Vec<oneshot::Receiver<()>>);

impl CoalescedCancellationHandle {
    fn push(&mut self, handle: oneshot::Receiver<()>) {
        self.0.push(handle);
    }

    async fn cancelled(self) {
        for rx in self.0 {
            // if this segment is completed before cancellation,
            // tx of this will be dropped, so ignore errors.
            let _ = rx.await;
        }
    }
}

async fn evaluate<R: VortexReadAt>(
    read: R,
    request: CoalescedSegmentRequest,
    segment_map: Arc<[Segment]>,
    segment_cache: Arc<dyn SegmentCache>,
    inflight_segments: InflightSegments,
) -> VortexResult<()> {
    log::debug!(
        "Reading byte range for {} requests {:?} size={}",
        request.requests.len(),
        request.byte_range,
        request.byte_range.end - request.byte_range.start,
    );
    let buffer: ByteBuffer = read
        .read_byte_range(request.byte_range.clone(), request.alignment)
        .await?
        .aligned(Alignment::none());

    // Figure out the segments covered by the read.
    let start = segment_map.partition_point(|s| s.offset < request.byte_range.start);
    let end = segment_map.partition_point(|s| s.offset < request.byte_range.end);

    // Note that we may have multiple requests for the same segment.
    let mut requests_iter = request.requests.into_iter().peekable();

    let cache_futures = FuturesUnordered::new();
    let mut results = Vec::new();
    for (i, segment) in segment_map[start..end].iter().enumerate() {
        let segment_id = SegmentId::from(u32::try_from(i + start).vortex_expect("segment id"));
        let offset = usize::try_from(segment.offset - request.byte_range.start)?;
        let buf = buffer
            .slice(offset..offset + segment.length as usize)
            .aligned(segment.alignment);

        // Find any request callbacks and send the buffer
        while let Some((id, _)) = requests_iter.peek() {
            // If the request is before the current segment, we should have already resolved it.
            match id.cmp(&segment_id) {
                Ordering::Less => {
                    // This should never happen, it means we missed a segment request.
                    vortex_panic!("Skipped segment request");
                }
                Ordering::Equal => {
                    // Resolve the request
                    let (id, _) = requests_iter.next().vortex_expect("next request");
                    results.push((id, Ok(buf.clone())));
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
    // resolve requests
    for (id, buf) in results {
        inflight_segments.complete(id, buf);
    }

    Ok(())
}

/// TODO(ngates): outsource coalescing to a trait
fn coalesce(
    requests: Vec<SegmentRequest>,
    coalescing_window: u64,
    max_read: Option<u64>,
) -> Vec<(CoalescedSegmentRequest, CoalescedCancellationHandle)> {
    let fetch_ranges = merge_ranges(
        requests.iter().map(|r| r.range()),
        coalescing_window,
        max_read,
    );
    let mut coalesced = fetch_ranges
        .iter()
        .map(|range| {
            (
                CoalescedSegmentRequest {
                    // We use the alignment of the first segment as the alignment for the entire request.
                    // TODO(ngates): if we had a VortexReadRanges trait, we could use pread where possible
                    //  to ensure correct alignment for all coalesced buffers.
                    alignment: requests
                        .first()
                        .map(|r| r.location.alignment)
                        .unwrap_or(Alignment::none()),
                    byte_range: range.clone(),
                    requests: vec![],
                },
                CoalescedCancellationHandle::default(),
            )
        })
        .collect::<Vec<_>>();

    for req in requests {
        let idx = fetch_ranges.partition_point(|v| v.start <= req.location.offset) - 1;
        let (ref mut request, ref mut cancellation) = coalesced[idx];
        request.requests.push((req.id, req.location));
        cancellation.push(req.cancel_handle);
    }

    // Ensure we sort the requests by segment ID within the coalesced request.
    for (req, _) in coalesced.iter_mut() {
        req.requests.sort_unstable_by_key(|(id, _)| *id);
    }
    coalesced
}

/// Returns a sorted list of ranges that cover `ranges`
///
/// From arrow-rs.
fn merge_ranges<R>(ranges: R, coalesce: u64, max_read: Option<u64>) -> Vec<Range<u64>>
where
    R: IntoIterator<Item = Range<u64>>,
{
    let mut ranges: Vec<Range<u64>> = ranges.into_iter().collect();
    ranges.sort_unstable_by_key(|range| range.start);

    let mut ret = Vec::with_capacity(ranges.len());
    let mut start_idx = 0;
    let mut end_idx = 1;

    while start_idx != ranges.len() {
        let start = ranges[start_idx].start;
        let mut range_end = ranges[start_idx].end;

        while end_idx != ranges.len()
            && ranges[end_idx]
                .start
                .checked_sub(range_end)
                .map(|delta| delta <= coalesce)
                .unwrap_or(true)
        {
            let new_range_end = range_end.max(ranges[end_idx].end);
            if max_read.is_some_and(|max| new_range_end - start > max) {
                break;
            }
            range_end = new_range_end;
            end_idx += 1;
        }

        let end = range_end;
        ret.push(start..end);

        start_idx = end_idx;
        end_idx += 1;
    }

    ret
}

struct CoalescingMetrics {
    bytes_uncoalesced: Arc<Counter>,
    bytes_coalesced: Arc<Counter>,
    request_count_uncoalesced: Arc<Counter>,
    request_count_coalesced: Arc<Counter>,
}

impl From<VortexMetrics> for CoalescingMetrics {
    fn from(metrics: VortexMetrics) -> Self {
        const BYTES: &str = "vortex.scan.requests.bytes";
        const COUNT: &str = "vortex.scan.requests.count";
        Self {
            bytes_uncoalesced: metrics.counter(format!("{BYTES}.uncoalesced")),
            bytes_coalesced: metrics.counter(format!("{BYTES}.coalesced")),
            request_count_uncoalesced: metrics.counter(format!("{COUNT}.uncoalesced")),
            request_count_coalesced: metrics.counter(format!("{COUNT}.coalesced")),
        }
    }
}

impl CoalescingMetrics {
    fn record(&self, req: &CoalescedSegmentRequest) {
        // record request counts
        self.request_count_coalesced.inc();
        if let Ok(len) = req.requests.len().try_into() {
            self.request_count_uncoalesced.add(len);
        }

        // record uncoalesced total byte requests vs coalesced
        if let Ok(bytes) = (req.byte_range.end - req.byte_range.start).try_into() {
            self.bytes_coalesced.add(bytes);
        }
        self.bytes_uncoalesced.add(
            req.requests
                .iter()
                .map(|(_, location)| location.length as i64)
                .sum(),
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_basic_merge() {
        let ranges = vec![0..2, 3..5, 1..4];
        let result = merge_ranges(ranges, 1, None);
        assert_eq!(result, vec![0..5]);
    }

    #[test]
    fn test_coalesce_with_max_read() {
        // Test interaction between coalesce and max_read
        let ranges = vec![0..3, 4..7, 8..11];

        // Should merge all with no max_read
        let result = merge_ranges(ranges.clone(), 2, None);
        assert_eq!(result, vec![0..11]);

        // Should not merge due to max_read limit
        let result = merge_ranges(ranges, 2, Some(5));
        assert_eq!(result, vec![0..3, 4..7, 8..11]);
    }

    #[test]
    fn test_overlapping_ranges_with_max_read() {
        let ranges = vec![0..6, 2..8, 7..10];

        // Should merge all with no max_read
        let result = merge_ranges(ranges.clone(), 1, None);
        assert_eq!(result, vec![0..10]);

        // Should merge partially with max_read
        let result = merge_ranges(ranges, 1, Some(9));
        assert_eq!(result, vec![0..8, 7..10]);
    }
}
