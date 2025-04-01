use std::sync::Arc;

use futures::{StreamExt, pin_mut};
use moka::sync::CacheBuilder;
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::{Dispatch, IoDispatcher, VortexReadAt};
use vortex_layout::segments::SegmentEvents;

use crate::segments::{CachedSegmentSource, CoalescedDriver, InMemorySegmentCache};
use crate::{FileType, VortexFile, VortexOpenOptions};

/// A type of Vortex file that supports any [`VortexReadAt`] implementation.
///
/// This is a reasonable choice for files backed by a network since it performs I/O coalescing.
// TODO(ngates): rename to TokioVortexFile
pub struct GenericVortexFile;

impl FileType for GenericVortexFile {
    type Options = GenericFileOptions;
}

impl VortexOpenOptions<GenericVortexFile> {
    const INITIAL_READ_SIZE: u64 = 1 << 20; // 1 MB

    /// Open a file using the provided [`VortexReadAt`] implementation.
    pub fn file() -> Self {
        Self::new(Default::default())
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

    pub async fn open<R: VortexReadAt + Send>(self, read: R) -> VortexResult<VortexFile> {
        let footer = self.read_footer(&read).await?;

        // We use segment events for driving I/O.
        let (segment_source, events) = SegmentEvents::create();

        // Wrap the source to resolve segments from the initial read cache.
        let segment_source = Arc::new(CachedSegmentSource::new(
            self.segment_cache.clone(),
            segment_source,
        ));

        let driver = CoalescedDriver::new(
            read.performance_hint(),
            footer.clone(),
            events,
            self.metrics.clone(),
        );

        // Spawn an I/O driver onto the dispatcher.
        let io_concurrency = self.options.io_concurrency;
        self.options
            .io_dispatcher
            .dispatch(move || {
                async move {
                    // Drive the segment event stream.
                    let stream = driver
                        .into_stream()
                        .map(|coalesced_req| coalesced_req.launch(read.clone()))
                        .buffer_unordered(io_concurrency);
                    pin_mut!(stream);

                    // Drive the stream to completion.
                    stream.collect::<()>().await
                }
            })
            .vortex_expect("Failed to spawn I/O driver");

        Ok(VortexFile {
            footer: footer.clone(),
            segment_source,
            metrics: self.metrics,
        })
    }
}

#[cfg(feature = "object_store")]
impl VortexOpenOptions<GenericVortexFile> {
    pub async fn open_object_store(
        self,
        object_store: &Arc<dyn object_store::ObjectStore>,
        path: &str,
    ) -> VortexResult<VortexFile> {
        use std::path::Path;

        use vortex_io::{ObjectStoreReadAt, TokioFile};

        // If the file is local, we much prefer to use TokioFile since object store re-opens the
        // file on every read. This check is a little naive... but we hope that ObjectStore will
        // soon expose the scheme in a way that we can check more thoroughly.
        // See: https://github.com/apache/arrow-rs-object-store/issues/259
        let local_path = Path::new("/").join(path);
        if local_path.exists() {
            // Local disk is too fast to justify prefetching.
            self.open(TokioFile::open(local_path)?).await
        } else {
            self.open(ObjectStoreReadAt::new(
                object_store.clone(),
                path.into(),
                None,
            ))
            .await
        }
    }
}

#[derive(Clone)]
pub struct GenericFileOptions {
    /// The number of concurrent I/O requests to spawn.
    /// This should be smaller than execution concurrency for coalescing to occur.
    io_concurrency: usize,
    /// The dispatcher to use for I/O requests.
    io_dispatcher: IoDispatcher,
}

impl Default for GenericFileOptions {
    fn default() -> Self {
        Self {
            io_concurrency: 10,
            io_dispatcher: IoDispatcher::default(),
        }
    }
}

// pub struct GenericScanDriver<R> {
//     read: R,
//     footer: Footer,
//     segment_cache: Arc<dyn SegmentCache>,
//     segment_queue: (),
//     metrics: VortexMetrics,
// }
//
// impl<R: VortexReadAt + Send> GenericScanDriver<R> {
//     /// Create a stream that is polled every time there is an available slot to perform I/O.
//     pub fn io_driver(self) -> impl Stream<Item = impl Future<Output = VortexResult<()>>> {
//         stream::unfold(self, move |mut this| async move {
//             loop {
//                 // Get the next most important segment to read, or else the stream is complete.
//                 let next = this.segment_queue.next().await?;
//                 let segment_map = this.footer.segment_map().clone();
//
//                 // If the segment is in the cache, we can skip the I/O.
//                 if let Some(cached_buffer) = this.segment_cache.get(next.id()).await.ok().flatten()
//                 {
//                     log::debug!("Resolving segment {} from cache", next.id());
//                     next.resolve(Ok(cached_buffer));
//                     continue;
//                 }
//
//                 // Build up a coalesced read with other segments from the queue.
//                 let coalesced = this.coalesce(next);
//
//                 this.metrics.counter("vortex.scan.generic.request").inc();
//
//                 // Launch the coalesced read.
//                 let read = this.read.clone();
//                 let fut = async move { evaluate(read, coalesced?, segment_map).await };
//
//                 return Some((fut, this));
//             }
//         })
//     }
//
//     fn coalesce(&self, next: PendingSegmentLease) -> VortexResult<CoalescedSegmentRequest> {
//         let segment_map = self.footer.segment_map();
//         let next_spec = segment_map
//             .get(*next.id() as usize)
//             .ok_or_else(|| vortex_err!("SegmentID {} not found", next.id()))?;
//         let first_req = SegmentRequest {
//             spec: next_spec.clone(),
//             lease: next,
//         };
//
//         // We build up a single coalesced read from the pending segments.
//         // Since pending segments are ordered by priority, we _always_ launch a request
//         // for the highest priority segment.
//         let mut coalesced = CoalescedSegmentRequest {
//             alignment: next_spec.alignment,
//             byte_range: next_spec.offset..next_spec.offset + next_spec.length as u64,
//             requests: vec![first_req],
//         };
//
//         let perf_hint = self.read.performance_hint();
//         let window = perf_hint.coalescing_window();
//         let max_read = perf_hint.max_read();
//
//         // We keep expanding our coalesced window until we reach max_read or no more segments
//         // can be coalesced.
//         loop {
//             let lowest_segment: SegmentId = segment_map
//                 .partition_point(|s| {
//                     (s.offset + s.length as u64) < coalesced.byte_range.start.saturating_sub(window)
//                 })
//                 .try_into()?;
//             let highest_segment: SegmentId = segment_map
//                 .partition_point(|s| s.offset < coalesced.byte_range.end.saturating_add(window))
//                 .try_into()?;
//             let segment_range = lowest_segment..highest_segment;
//
//             let matching = self.segment_queue.lease_within_range(&segment_range);
//             if matching.is_empty() {
//                 break;
//             }
//
//             for lease in matching {
//                 let spec = segment_map
//                     .get(*lease.id() as usize)
//                     .ok_or_else(|| vortex_err!("SegmentID {} not found", lease.id()))?;
//
//                 let segment_start = spec.offset;
//                 let segment_end = spec.offset + spec.length as u64;
//
//                 coalesced.byte_range.start = coalesced.byte_range.start.min(segment_start);
//                 coalesced.byte_range.end = coalesced.byte_range.end.max(segment_end);
//                 // Take the maximum alignment of all segments in the coalesced request.
//                 // FIXME(ngates): shouldn't this be the _first_ segment?
//                 coalesced.alignment = coalesced.alignment.max(spec.alignment);
//                 coalesced.requests.push(SegmentRequest {
//                     spec: spec.clone(),
//                     lease,
//                 });
//             }
//
//             if let Some(max_read) = max_read {
//                 if coalesced.byte_range.end - coalesced.byte_range.start > max_read {
//                     break;
//                 }
//             }
//         }
//
//         // Ensure the coalesced requests are sorted
//         coalesced.requests.sort_by_key(|r| r.id());
//
//         Ok(coalesced)
//     }
// }
//
// struct SegmentRequest {
//     spec: SegmentSpec,
//     lease: PendingSegmentLease,
// }
//
// impl Debug for SegmentRequest {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("SegmentRequest")
//             .field("spec", &self.spec)
//             .finish()
//     }
// }
//
// impl SegmentRequest {
//     fn id(&self) -> SegmentId {
//         self.lease.id()
//     }
//
//     fn range(&self) -> Range<u64> {
//         self.spec.offset..self.spec.offset + self.spec.length as u64
//     }
// }
//
// #[derive(Debug)]
// struct CoalescedSegmentRequest {
//     /// The alignment of the first segment.
//     // TODO(ngates): is this the best alignment to use?
//     pub(crate) alignment: Alignment,
//     /// The range of the file to read.
//     pub(crate) byte_range: Range<u64>,
//     /// The original segment requests, ordered by segment ID.
//     pub(crate) requests: Vec<SegmentRequest>,
// }
//
// impl CoalescedSegmentRequest {
//     fn size_bytes(&self) -> u64 {
//         self.byte_range.end - self.byte_range.start
//     }
// }
//
// async fn evaluate<R: VortexReadAt + Send>(
//     read: R,
//     request: CoalescedSegmentRequest,
//     segment_map: Arc<[SegmentSpec]>,
// ) -> VortexResult<()> {
//     log::debug!(
//         "Reading byte range for [{}] requests {:?} size={}",
//         request.requests.iter().map(|r| r.id()).join(", "),
//         request.byte_range,
//         request.byte_range.end - request.byte_range.start,
//     );
//     let buffer: ByteBuffer = read
//         .read_byte_range(request.byte_range.clone(), request.alignment)
//         .await?
//         .aligned(Alignment::none());
//
//     // Figure out the segments covered by the read.
//     let start = segment_map.partition_point(|s| s.offset < request.byte_range.start);
//     let end = segment_map.partition_point(|s| s.offset < request.byte_range.end);
//
//     // Note that we may have multiple requests for the same segment.
//     let mut requests_iter = request.requests.into_iter().peekable();
//
//     for (i, segment) in segment_map[start..end].iter().enumerate() {
//         let segment_id = SegmentId::from(u32::try_from(i + start).vortex_expect("segment id"));
//         let offset = usize::try_from(segment.offset - request.byte_range.start)?;
//         let buf = buffer
//             .slice(offset..offset + segment.length as usize)
//             .aligned(segment.alignment);
//
//         // Find any request callbacks and send the buffer
//         while let Some(req) = requests_iter.peek() {
//             // If the request is before the current segment, we should have already resolved it.
//             match req.id().cmp(&segment_id) {
//                 Ordering::Less => {
//                     // This should never happen, it means we missed a segment request.
//                     vortex_panic!("Skipped segment request");
//                 }
//                 Ordering::Equal => {
//                     // Resolve the request
//                     requests_iter
//                         .next()
//                         .vortex_expect("next request")
//                         .lease
//                         .resolve(Ok(buf.clone()));
//                 }
//                 Ordering::Greater => {
//                     // No request for this segment, so we continue
//                     break;
//                 }
//             }
//         }
//     }
//
//     Ok(())
// }
