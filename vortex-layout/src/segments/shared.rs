// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::WeakShared;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A [`SegmentSource`] that allows multiple requesters to await the same underlying segment
/// request.
pub struct SharedSegmentSource<S> {
    inner: S,
    in_flight: DashMap<RequestKey, WeakShared<SharedSegmentFuture>>,
}

type SharedSegmentFuture = BoxFuture<'static, SharedVortexResult<BufferHandle>>;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum RequestKey {
    Full(SegmentId),
    Ranges(SegmentId, Vec<Range<usize>>),
}

impl<S: SegmentSource> SharedSegmentSource<S> {
    /// Create a new `SharedSegmentSource` wrapping the provided inner source.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            in_flight: DashMap::default(),
        }
    }
}

impl<S: SegmentSource> SegmentSource for SharedSegmentSource<S> {
    fn segment_len(&self, id: SegmentId) -> Option<usize> {
        self.inner.segment_len(id)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        loop {
            let key = RequestKey::Full(id);
            match self.in_flight.entry(key) {
                Entry::Occupied(e) => {
                    if let Some(shared_future) = e.get().upgrade() {
                        return shared_future.map_err(VortexError::from).boxed();
                    } else {
                        // The future has been dropped, remove the entry and try again.
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    let future = self.inner.request(id).map_err(Arc::new).boxed().shared();
                    e.insert(
                        future
                            .downgrade()
                            .vortex_expect("just created, cannot be polled to completion"),
                    );
                    return future.map_err(VortexError::from).boxed();
                }
            }
        }
    }

    fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<usize>>) -> SegmentFuture {
        // For small segments, fetch the full segment via request() (which benefits from
        // WeakShared dedup) then apply ranges locally. Reading extra bytes is negligible
        // for small segments, and this avoids duplicate I/O when different callers request
        // different sub-ranges of the same small segment.
        if self.inner.segment_len(id).is_some_and(|len| len <= 4096) {
            let fut = self.request(id);
            return async move {
                let buf = fut.await?;
                crate::segments::apply_ranges(buf, &ranges)
            }
            .boxed();
        }

        loop {
            let key = RequestKey::Ranges(id, ranges.clone());
            match self.in_flight.entry(key) {
                Entry::Occupied(e) => {
                    if let Some(shared_future) = e.get().upgrade() {
                        return shared_future.map_err(VortexError::from).boxed();
                    } else {
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    let future = self
                        .inner
                        .request_ranges(id, ranges)
                        .map_err(Arc::new)
                        .boxed()
                        .shared();
                    e.insert(
                        future
                            .downgrade()
                            .vortex_expect("just created, cannot be polled to completion"),
                    );
                    return future.map_err(VortexError::from).boxed();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_buffer::ByteBuffer;

    use super::*;
    use crate::segments::SegmentSink;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;

    // Custom source that tracks how many times a segment is requested
    #[derive(Default, Clone)]
    struct CountingSegmentSource {
        segments: TestSegments,
        request_count: Arc<AtomicUsize>,
        range_request_count: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingSegmentSource {
        fn segment_len(&self, id: SegmentId) -> Option<usize> {
            self.segments.segment_len(id)
        }

        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.request_count.fetch_add(1, Ordering::SeqCst);
            self.segments.request(id)
        }

        fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<usize>>) -> SegmentFuture {
            self.range_request_count.fetch_add(1, Ordering::SeqCst);
            let segments = self.segments.clone();
            async move {
                let buffer = segments.request(id).await?;
                crate::segments::apply_ranges(buffer, &ranges)
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn test_shared_source_deduplicates_concurrent_requests() {
        let source = CountingSegmentSource::default();

        // Add a segment to the test source
        let data = ByteBuffer::from(vec![1, 2, 3, 4]);
        let seq_id = SequenceId::root().downgrade();
        source
            .segments
            .write(seq_id, vec![data.clone()])
            .await
            .unwrap();

        let shared_source = SharedSegmentSource::new(source.clone());

        // Request the same segment twice concurrently
        let id = SegmentId::from(0);
        let future1 = shared_source.request(id);
        let future2 = shared_source.request(id);

        // Both futures should resolve to the same data
        let (result1, result2) = futures::join!(future1, future2);
        assert_eq!(result1.unwrap().unwrap_host(), data);
        assert_eq!(result2.unwrap().unwrap_host(), data);

        // The inner source should have been called only once
        assert_eq!(source.request_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_shared_source_deduplicates_concurrent_ranged_requests() {
        let source = CountingSegmentSource::default();

        let data = ByteBuffer::from(vec![1, 2, 3, 4, 5, 6]);
        let seq_id = SequenceId::root().downgrade();
        source.segments.write(seq_id, vec![data]).await.unwrap();

        let shared_source = SharedSegmentSource::new(source.clone());
        let id = SegmentId::from(0);
        let ranges = vec![1..3, 4..6];
        let future1 = shared_source.request_ranges(id, ranges.clone());
        let future2 = shared_source.request_ranges(id, ranges);

        let (result1, result2) = futures::join!(future1, future2);
        assert_eq!(result1.unwrap().unwrap_host().as_slice(), &[2, 3, 5, 6]);
        assert_eq!(result2.unwrap().unwrap_host().as_slice(), &[2, 3, 5, 6]);
        assert_eq!(source.range_request_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_shared_source_handles_dropped_futures() {
        let source = CountingSegmentSource::default();

        // Add a segment
        let data = ByteBuffer::from(vec![5, 6, 7, 8]);
        let seq_id = SequenceId::root().downgrade();
        source
            .segments
            .write(seq_id, vec![data.clone()])
            .await
            .unwrap();

        let shared_source = SharedSegmentSource::new(source.clone());
        let id = SegmentId::from(0);

        // Create and immediately drop a future
        {
            let _future = shared_source.request(id);
            // Future is dropped here
        }

        // A new request should still work correctly
        let result = shared_source.request(id).await;
        assert_eq!(result.unwrap().unwrap_host(), data);

        // Should have made 2 requests since the first was dropped before completion
        assert_eq!(source.request_count.load(Ordering::Relaxed), 2);
    }
}
