use std::cmp::Ordering;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use dashmap::{DashMap, Entry};
use futures::channel::oneshot;
use futures::{Stream, StreamExt, pin_mut, stream};
use moka::future::CacheBuilder;
use pin_project_lite::pin_project;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_io::{Dispatch, IoDispatcher, VortexReadAt};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};
use vortex_metrics::{Counter, VortexMetrics};

use crate::footer::{Footer, SegmentSpec};
use crate::segments::queue::SegmentQueue;
use crate::segments::{InMemorySegmentCache, SegmentCache};
use crate::{FileType, VortexFile, VortexOpenOptions};

/// A type of Vortex file that supports any [`VortexReadAt`] implementation.
///
/// This is a reasonable choice for files backed by a network since it performs I/O coalescing.
pub struct GenericVortexFile<R> {
    footer: Footer,
    read: R,
    segment_reader: Arc<dyn AsyncSegmentReader>,
    segment_cache: Arc<dyn SegmentCache>,
    metrics: VortexMetrics,
}

impl<R: VortexReadAt + Send + Sync> VortexOpenOptions<GenericVortexFile<R>> {
    const INITIAL_READ_SIZE: u64 = 1 << 20; // 1 MB

    pub fn file(read: R) -> Self {
        Self::new(read, Default::default())
            .with_segment_cache(Arc::new(InMemorySegmentCache::new(
                // For now, use a fixed 1GB overhead.
                CacheBuilder::new(1 << 30),
            )))
            .with_initial_read_size(Self::INITIAL_READ_SIZE)
    }

    pub fn with_io_concurrency(mut self, io_concurrency: usize) -> Self {
        self.options.io_concurrency = io_concurrency;
        self
    }
}

impl<R: VortexReadAt + Send> FileType for GenericVortexFile<R> {
    type Options = GenericScanOptions;
    type Read = R;

    fn open(options: VortexOpenOptions<Self>, footer: Footer) -> VortexResult<VortexFile> {
        let (segment_queue, segment_reader) = SegmentQueue::new();

        // Spawn an I/O driver to serve requests while this file is open.
        let driver = GenericScanDriver {
            read: options.read,
            footer: footer.clone(),
            segment_cache: options.segment_cache,
            segment_queue,
            metrics: CoalescingMetrics::from(options.metrics.clone()),
        };

        options.options.io_dispatcher.dispatch(move || async move {
            let io_stream = driver
                .io_driver()
                .buffer_unordered(options.options.io_concurrency);

            pin_mut!(io_stream);
            while let Some(r) = io_stream.next().await {
                if r.is_err() {
                    log::error!("GenericVortexFile SegmentQueue IO driver failed: {:?}", r)
                }
            }
        })?;

        Ok(VortexFile {
            footer,
            segment_reader,
            metrics: options.metrics,
        })
    }
}

#[derive(Clone)]
pub struct GenericScanOptions {
    /// The number of concurrent I/O requests to spawn.
    /// This should be smaller than execution concurrency for coalescing to occur.
    io_concurrency: usize,
    /// The dispatcher pool used to perform I/O.
    io_dispatcher: IoDispatcher,
}

impl Default for GenericScanOptions {
    fn default() -> Self {
        Self {
            io_concurrency: 16,
            io_dispatcher: IoDispatcher::default(),
        }
    }
}

pub struct GenericScanDriver<R> {
    read: R,
    footer: Footer,
    segment_cache: Arc<dyn SegmentCache>,
    segment_queue: SegmentQueue,
    metrics: CoalescingMetrics,
}

impl<R: VortexReadAt + Send> GenericScanDriver<R> {
    pub fn io_driver(self) -> impl Stream<Item = impl Future<Output = VortexResult<()>>> {
        // Create a stream that yields every time there is more work to do in the segment queue.
        stream::unfold(self, move |mut this| async move {
            // If the segment queue is empty, then wait for the next notification.
            if this.segment_queue.is_empty() {
                let Some(_) = this.segment_queue.next().await else {
                    // The segment queue has completed, meaning no more requests are possible.
                    // We're done!
                    return None;
                };
            }

            // Using the pending segments, construct a single coalesced read.
            let request = this
                .segment_queue
                .with_pending_segments(|pending_segments| {
                    let first = pending_segments
                        .next()
                        .vortex_expect("empty iterator from non-empty queue");
                    let first_spec = this
                        .footer
                        .segment_map()
                        .get(*first.id() as usize)
                        .ok_or_else(|| vortex_err!("SegmentID {} not found", first.id()))?;
                    let first_req = SegmentRequest {
                        id: first.id(),
                        spec: first_spec.clone(),
                        callback: first
                            .take_callback()
                            .vortex_expect("pending segment must have callback"),
                    };

                    // We build up a single coalesced read from the pending segments.
                    // Since pending segments are ordered by priority, we _always_ launch a request
                    // for the highest priority segment.
                    let mut coalesced = CoalescedSegmentRequest {
                        alignment: first_spec.alignment,
                        byte_range: first_spec.offset..first_spec.offset + first_spec.length as u64,
                        requests: vec![first_req],
                    };

                    let perf_hint = this.read.performance_hint();
                    let window = perf_hint.coalescing_window();
                    let max_read = perf_hint.max_read().unwrap_or(2 << 24); // 16MB.

                    for pending in pending_segments {
                        // If the coalesced request has reached the maximum size, ship it.
                        if coalesced.size_bytes() > max_read {
                            break;
                        }

                        // Otherwise, try to include the pending segment in the request.
                        let spec = this
                            .footer
                            .segment_map()
                            .get(*pending.id() as usize)
                            .ok_or_else(|| vortex_err!("SegmentID {} not found", pending.id()))?;

                        let segment_start = spec.offset;
                        let segment_end = spec.offset + spec.length as u64;

                        // Check if the segment should be included in the coalesced request.
                        if coalesced.byte_range.contains(&segment_start)
                            || coalesced.byte_range.contains(&segment_end)
                            || segment_start.abs_diff(coalesced.byte_range.end) < window
                            || segment_end.abs_diff(coalesced.byte_range.start) < window
                        {
                            coalesced.byte_range.start =
                                coalesced.byte_range.start.min(segment_start);
                            coalesced.byte_range.end = coalesced.byte_range.end.max(segment_end);
                            // Take the maximum alignment of all segments in the coalesced request.
                            coalesced.alignment = coalesced.alignment.max(spec.alignment);
                            coalesced.requests.push(SegmentRequest {
                                id: pending.id(),
                                spec: spec.clone(),
                                callback: pending
                                    .take_callback()
                                    .vortex_expect("pending segment must have callback"),
                            });
                        }
                    }

                    // Finally, we ensure the coalesced segments are sorted by ID.
                    coalesced.requests.sort_unstable_by_key(|r| r.id);

                    Ok::<_, VortexError>(coalesced)
                });

            // Launch the coalesced read.
            let read = this.read.clone();
            let segment_map = this.footer.segment_map().clone();
            let fut = async move { evaluate(read, request?, segment_map).await };

            Some((fut, this))
        })
    }
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
//
// impl<R: VortexReadAt> ScanDriver for GenericScanDriver<R> {
//     fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
//         self.segment_channel.reader()
//     }
//
//     fn io_stream(self, segments: SegmentStream) -> impl Stream<Item = VortexResult<()>> {
//         let segment_requests = self.segment_channel.into_stream();
//         let segment_map = self.footer.segment_map().clone();
//         let inflight_segments = InflightSegments::default();
//
//         let inflight = inflight_segments.clone();
//         let segment_requests = segment_requests.filter_map(move |request| {
//             let Some(location) = segment_map.get(*request.id as usize) else {
//                 request.resolve(Err(vortex_err!("segment not found")));
//                 return future::ready(None);
//             };
//
//             // We support zero-length segments (so layouts don't have to store this information)
//             // If we encounter a zero-length segment, we can just resolve it now.
//             if location.length == 0 {
//                 let alignment = location.alignment;
//                 request.resolve(Ok(ByteBuffer::empty_aligned(alignment)));
//                 return future::ready(None);
//             }
//
//             let cancel_handle =
//                 inflight.insert_or_register_callback(request.id, Some(request.callback));
//             future::ready(
//                 cancel_handle
//                     .map(|handle| SegmentRequest::new(request.id, location.clone(), handle)),
//             )
//         });
//
//         let inflight = inflight_segments.clone();
//         let segment_map = self.footer.segment_map().clone();
//         let prefetch_stream = segments.filter_map(move |event| match event {
//             SegmentEvent::Cancel(id) => {
//                 inflight.cancel(id);
//                 future::ready(None)
//             }
//             SegmentEvent::Request(id) => {
//                 future::ready(segment_map.get(*id as usize).and_then(|location| {
//                     inflight
//                         .insert_or_register_callback(id, None)
//                         .map(|cancel_handle| {
//                             SegmentRequest::new(id, location.clone(), cancel_handle)
//                         })
//                 }))
//             }
//         });
//
//         // Check if the segment exists in the cache
//         let segment_cache = self.segment_cache.clone();
//         let inflight = inflight_segments.clone();
//         let segment_requests = segment_requests.filter_map(move |request| {
//             filter_with_cache(request, segment_cache.clone(), inflight.clone())
//         });
//         let segment_cache = self.segment_cache.clone();
//         let inflight = inflight_segments.clone();
//         let prefetch_stream = prefetch_stream.filter_map(move |request| {
//             filter_with_cache(request, segment_cache.clone(), inflight.clone())
//         });
//
//         // Grab all available segment requests from the I/O queue so we get maximal visibility into
//         // the requests for coalescing.
//         // Note that we can provide a somewhat arbitrarily high capacity here since we're going to
//         // deduplicate and coalesce. Meaning the resulting stream will at-most cover the entire
//         // file and therefore be reasonably bounded.
//         // Coalesce the segment requests to minimize the number of I/O operations.
//         let perf_hint = self.read.performance_hint();
//         let io_stream = SegmentRequestStream {
//             requests: segment_requests,
//             prefetch: prefetch_stream,
//             requests_ready_chunks: 1024,
//             prefetch_ready_chunks: self.options.io_concurrency,
//             requested_segments: self.metrics.requested_segments.clone(),
//             prefetched_segments: self.metrics.prefetched_segments.clone(),
//         }
//         .map(move |r| {
//             coalesce(
//                 r,
//                 perf_hint.coalescing_window(),
//                 perf_hint.max_read(),
//                 self.metrics.clone(),
//             )
//         })
//         .flat_map(stream::iter);
//
//         // Submit the coalesced requests to the I/O.
//         let read = self.read.clone();
//         let segment_map = self.footer.segment_map().clone();
//         let segment_cache = self.segment_cache.clone();
//         let io_stream = io_stream.map(move |(request, cancellation_handle)| {
//             let read = read.clone();
//             let segment_map = segment_map.clone();
//             let segment_cache = segment_cache.clone();
//             let inflight = inflight_segments.clone();
//             async move {
//                 select! {
//                     _ = cancellation_handle.cancelled().fuse() => Ok(()),
//                     evaluated = evaluate(
//                         read.clone(),
//                         request,
//                         segment_map.clone(),
//                         segment_cache.clone(),
//                         inflight.clone(),
//
//                     ).fuse() => evaluated,
//                 }
//             }
//         });
//
//         // Buffer some number of concurrent I/O operations.
//         instrument!(
//             "io_stream",
//             io_stream.buffer_unordered(self.options.io_concurrency)
//         )
//     }
// }

pin_project! {
    struct SegmentRequestStream<Requests, Prefetch> {
        #[pin]
        pub requests: Requests,
        #[pin]
        pub prefetch: Prefetch,
        requests_ready_chunks: usize,
        prefetch_ready_chunks: usize,
        requested_segments: Arc<Counter>,
        prefetched_segments: Arc<Counter>,
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
                    this.requested_segments.inc();
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
                    this.prefetched_segments.inc();
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
    spec: SegmentSpec,
    callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl SegmentRequest {
    fn range(&self) -> Range<u64> {
        self.spec.offset..self.spec.offset + self.spec.length as u64
    }
}

async fn filter_with_cache(
    request: SegmentRequest,
    cache: Arc<dyn SegmentCache>,
    inflight_segments: InflightSegments,
) -> Option<SegmentRequest> {
    match cache.get(request.id, request.spec.alignment).await {
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
    pub(crate) requests: Vec<SegmentRequest>,
}

impl CoalescedSegmentRequest {
    fn size_bytes(&self) -> u64 {
        self.byte_range.end - self.byte_range.start
    }
}

#[derive(Default)]
struct CoalescedCancellationHandle {
    handles: Vec<oneshot::Receiver<()>>,
    cancel_received: Arc<Counter>,
    cancelled: Arc<Counter>,
}

impl CoalescedCancellationHandle {
    fn new(
        handles: Vec<oneshot::Receiver<()>>,
        cancel_received: Arc<Counter>,
        cancelled: Arc<Counter>,
    ) -> Self {
        Self {
            handles,
            cancel_received,
            cancelled,
        }
    }

    fn push(&mut self, handle: oneshot::Receiver<()>) {
        self.handles.push(handle);
    }

    async fn cancelled(self) {
        for rx in self.handles {
            // if this segment is completed before cancellation,
            // tx of this will be dropped, so ignore errors.
            let _ = rx.await;
            self.cancel_received.inc();
        }
        self.cancelled.inc();
    }
}

async fn evaluate<R: VortexReadAt + Send>(
    read: R,
    request: CoalescedSegmentRequest,
    segment_map: Arc<[SegmentSpec]>,
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
                    let req = requests_iter.next().vortex_expect("next request");
                    if let Err(_) = req.callback.send(Ok(buf.clone())) {
                        // The receiver was dropped, which means the segment is no longer needed.
                        log::debug!("Segment request was dropped while in-flight: {}", req.id);
                    }
                }
                Ordering::Greater => {
                    // No request for this segment, so we continue
                    break;
                }
            }
        }
    }

    Ok(())
}

#[derive(Clone)]
struct CoalescingMetrics {
    bytes_uncoalesced: Arc<Counter>,
    bytes_coalesced: Arc<Counter>,
    request_count_uncoalesced: Arc<Counter>,
    request_count_coalesced: Arc<Counter>,
    prefetched_segments: Arc<Counter>,
    requested_segments: Arc<Counter>,
    cancel_received: Arc<Counter>,
    cancelled: Arc<Counter>,
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
            prefetched_segments: metrics.counter("vortex.scan.segments.prefetch_count"),
            requested_segments: metrics.counter("vortex.scan.segments.request_count"),
            cancel_received: metrics.counter("vortex.scan.segments.cancel_received"),
            cancelled: metrics.counter("vortex.scan.segments.cancelled"),
        }
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
