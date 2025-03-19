use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

use futures::{Stream, StreamExt, pin_mut, stream};
use moka::future::CacheBuilder;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_io::{Dispatch, IoDispatcher, VortexReadAt};
use vortex_layout::segments::{AsyncSegmentReader, PendingSegmentLease, SegmentId};
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
            io_concurrency: 2,
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
            let Some(_) = this.segment_queue.next().await else {
                // The segment queue has completed, meaning no more requests are possible.
                // We're done!
                log::info!("I/O driver finished");
                return None;
            };

            // Using the pending segments, construct a single coalesced read.
            let request = this
                .segment_queue
                .with_pending_segments(|pending_segments| {
                    let Some(first) = pending_segments.next() else {
                        return Ok(None);
                    };

                    let first_spec = this
                        .footer
                        .segment_map()
                        .get(*first.id() as usize)
                        .ok_or_else(|| vortex_err!("SegmentID {} not found", first.id()))?;
                    let first_req = SegmentRequest {
                        spec: first_spec.clone(),
                        lease: first,
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
                    log::debug!("Using performance hint {:?}", perf_hint);
                    let window = perf_hint.coalescing_window();
                    let max_read = perf_hint.max_read().unwrap_or(2 << 24); // 16MB.

                    for pending in pending_segments {
                        // If the coalesced request has reached the maximum size, ship it.
                        if coalesced.size_bytes() > max_read {
                            log::debug!(
                                "Coalesced read {:?} reached max size {}",
                                coalesced,
                                max_read
                            );
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
                            || segment_start.abs_diff(coalesced.byte_range.end) <= window
                            || segment_end.abs_diff(coalesced.byte_range.start) <= window
                        {
                            coalesced.byte_range.start =
                                coalesced.byte_range.start.min(segment_start);
                            coalesced.byte_range.end = coalesced.byte_range.end.max(segment_end);
                            // Take the maximum alignment of all segments in the coalesced request.
                            // FIXME(ngates): shouldn't this be the _first_ segment?
                            coalesced.alignment = coalesced.alignment.max(spec.alignment);
                            coalesced.requests.push(SegmentRequest {
                                spec: spec.clone(),
                                lease: pending,
                            });
                        }
                    }

                    // Finally, we ensure the coalesced segments are sorted by ID.
                    coalesced.requests.sort_unstable_by_key(|r| r.id());

                    Ok::<_, VortexError>(Some(coalesced))
                });

            // Launch the coalesced read.
            let read = this.read.clone();
            let segment_map = this.footer.segment_map().clone();
            let fut = async move {
                if let Some(request) = request? {
                    evaluate(read, request, segment_map).await
                } else {
                    Ok(())
                }
            };

            Some((fut, this))
        })
    }
}

#[derive(Debug)]
struct SegmentRequest {
    spec: SegmentSpec,
    lease: PendingSegmentLease,
}

impl SegmentRequest {
    fn id(&self) -> SegmentId {
        self.lease.id()
    }

    fn range(&self) -> Range<u64> {
        self.spec.offset..self.spec.offset + self.spec.length as u64
    }
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
            match req.id().cmp(&segment_id) {
                Ordering::Less => {
                    // This should never happen, it means we missed a segment request.
                    vortex_panic!("Skipped segment request");
                }
                Ordering::Equal => {
                    // Resolve the request
                    requests_iter
                        .next()
                        .vortex_expect("next request")
                        .lease
                        .resolve(Ok(buf.clone()));
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
