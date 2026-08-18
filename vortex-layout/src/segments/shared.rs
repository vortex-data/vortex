// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::future::WeakShared;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A [`SegmentSource`] that allows multiple requesters to await the same underlying segment
/// request.
pub struct SharedSegmentSource<S> {
    inner: S,
    in_flight: Arc<DashMap<SegmentRequest, WeakShared<SharedSegmentFuture>>>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum SegmentRequest {
    Whole(SegmentId),
    Range(SegmentId, Range<u64>),
}

type SharedSegmentFuture = BoxFuture<'static, SharedVortexResult<BufferHandle>>;

struct InFlightGuard {
    in_flight: Arc<DashMap<SegmentRequest, WeakShared<SharedSegmentFuture>>>,
    request: SegmentRequest,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.in_flight.remove(&self.request);
    }
}

impl<S: SegmentSource> SharedSegmentSource<S> {
    /// Create a new `SharedSegmentSource` wrapping the provided inner source.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            in_flight: Arc::default(),
        }
    }
}

impl<S: SegmentSource> SegmentSource for SharedSegmentSource<S> {
    fn preferred_read_size(&self) -> Option<u64> {
        self.inner.preferred_read_size()
    }

    fn segment_len(&self, id: SegmentId) -> Option<u64> {
        self.inner.segment_len(id)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.request_shared(SegmentRequest::Whole(id))
    }

    fn request_range(&self, id: SegmentId, range: Range<u64>) -> SegmentFuture {
        self.request_shared(SegmentRequest::Range(id, range))
    }

    fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<u64>>) -> Vec<SegmentFuture> {
        let mut outputs = (0..ranges.len()).map(|_| None).collect::<Vec<_>>();
        let mut missing = Vec::new();

        for (index, range) in ranges.into_iter().enumerate() {
            let request = SegmentRequest::Range(id, range.clone());
            loop {
                match self.in_flight.entry(request.clone()) {
                    Entry::Occupied(entry) => {
                        if let Some(future) = entry.get().upgrade() {
                            outputs[index] = Some(future.map_err(VortexError::from).boxed());
                            break;
                        }
                        entry.remove();
                    }
                    Entry::Vacant(entry) => {
                        let (send, receive) = oneshot::channel::<SegmentFuture>();
                        let guard = InFlightGuard {
                            in_flight: Arc::clone(&self.in_flight),
                            request: request.clone(),
                        };
                        let future = async move {
                            let _guard = guard;
                            let inner = receive.await.map_err(|_| {
                                Arc::new(vortex_err!("Batched segment request was dropped"))
                            })?;
                            inner.await.map_err(Arc::new)
                        }
                        .boxed()
                        .shared();
                        entry.insert(
                            future
                                .downgrade()
                                .vortex_expect("new shared future cannot be complete"),
                        );
                        outputs[index] = Some(future.map_err(VortexError::from).boxed());
                        missing.push((range, send));
                        break;
                    }
                }
            }
        }

        let inner = self
            .inner
            .request_ranges(id, missing.iter().map(|(range, _)| range.clone()).collect());
        for ((_, send), future) in missing.into_iter().zip(inner) {
            drop(send.send(future));
        }

        outputs
            .into_iter()
            .map(|future| future.vortex_expect("every requested range has a future"))
            .collect()
    }
}

impl<S: SegmentSource> SharedSegmentSource<S> {
    fn request_shared(&self, request: SegmentRequest) -> SegmentFuture {
        loop {
            match self.in_flight.entry(request.clone()) {
                Entry::Occupied(e) => {
                    if let Some(shared_future) = e.get().upgrade() {
                        return shared_future.map_err(VortexError::from).boxed();
                    } else {
                        // The future has been dropped, remove the entry and try again.
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    let inner_future = match &request {
                        SegmentRequest::Whole(id) => self.inner.request(*id),
                        SegmentRequest::Range(id, range) => {
                            self.inner.request_range(*id, range.clone())
                        }
                    };
                    let guard = InFlightGuard {
                        in_flight: Arc::clone(&self.in_flight),
                        request: request.clone(),
                    };
                    let future = async move {
                        let _guard = guard;
                        inner_future.await.map_err(Arc::new)
                    }
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
    use vortex_error::VortexResult;

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
        range_batch_count: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingSegmentSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.request_count.fetch_add(1, Ordering::SeqCst);
            self.segments.request(id)
        }

        fn request_range(&self, id: SegmentId, range: Range<u64>) -> SegmentFuture {
            self.range_request_count.fetch_add(1, Ordering::SeqCst);
            self.segments.request_range(id, range)
        }

        fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<u64>>) -> Vec<SegmentFuture> {
            self.range_batch_count.fetch_add(1, Ordering::SeqCst);
            ranges
                .into_iter()
                .map(|range| self.request_range(id, range))
                .collect()
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
        assert!(shared_source.in_flight.is_empty());
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
        assert!(shared_source.in_flight.is_empty());

        // A new request should still work correctly
        let result = shared_source.request(id).await;
        assert_eq!(result.unwrap().unwrap_host(), data);

        // Should have made 2 requests since the first was dropped before completion
        assert_eq!(source.request_count.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_shared_source_deduplicates_identical_ranges() -> VortexResult<()> {
        let source = CountingSegmentSource::default();
        let data = ByteBuffer::from(vec![1, 2, 3, 4]);
        let seq_id = SequenceId::root().downgrade();
        source.segments.write(seq_id, vec![data]).await?;

        let shared_source = SharedSegmentSource::new(source.clone());
        let id = SegmentId::from(0);
        let (first, second) = futures::join!(
            shared_source.request_range(id, 1..3),
            shared_source.request_range(id, 1..3)
        );
        assert_eq!(first?.unwrap_host(), ByteBuffer::from(vec![2, 3]));
        assert_eq!(second?.unwrap_host(), ByteBuffer::from(vec![2, 3]));
        assert_eq!(source.range_request_count.load(Ordering::Relaxed), 1);
        assert!(shared_source.in_flight.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_shared_source_forwards_missing_ranges_as_one_batch() -> VortexResult<()> {
        let source = CountingSegmentSource::default();
        let data = ByteBuffer::from(vec![1, 2, 3, 4]);
        let seq_id = SequenceId::root().downgrade();
        source.segments.write(seq_id, vec![data]).await?;

        let shared_source = SharedSegmentSource::new(source.clone());
        let reads = shared_source.request_ranges(SegmentId::from(0), vec![0..1, 2..4]);
        let mut results = futures::future::join_all(reads).await.into_iter();
        assert_eq!(results.next().vortex_expect("first range")?.len(), 1);
        assert_eq!(results.next().vortex_expect("second range")?.len(), 2);
        assert_eq!(source.range_batch_count.load(Ordering::Relaxed), 1);
        assert_eq!(source.range_request_count.load(Ordering::Relaxed), 2);
        assert!(shared_source.in_flight.is_empty());
        Ok(())
    }
}
