// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::WeakShared;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_io::ReadAtNowait;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A [`SegmentSource`] that allows multiple requesters to await the same underlying segment
/// request.
pub struct SharedSegmentSource<S> {
    inner: S,
    in_flight: DashMap<SegmentId, WeakShared<SharedSegmentFuture>>,
    request_lock: Mutex<()>,
}

type SharedSegmentFuture = BoxFuture<'static, SharedVortexResult<BufferHandle>>;
type StrongSharedSegmentFuture = Shared<SharedSegmentFuture>;

impl<S: SegmentSource> SharedSegmentSource<S> {
    /// Create a new `SharedSegmentSource` wrapping the provided inner source.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            in_flight: DashMap::default(),
            request_lock: Mutex::new(()),
        }
    }
}

impl<S: SegmentSource> SegmentSource for SharedSegmentSource<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let _guard = self.request_lock.lock();
        self.request_with(id, |source, id| source.request(id))
    }

    fn request_background(&self, id: SegmentId) -> SegmentFuture {
        let _guard = self.request_lock.lock();
        self.request_with(id, |source, id| source.request_background(id))
    }

    fn request_background_batch(&self, ids: &[SegmentId]) -> Vec<SegmentFuture> {
        let _guard = self.request_lock.lock();
        let mut shared = HashMap::<SegmentId, StrongSharedSegmentFuture>::default();
        let mut missing = Vec::new();
        let mut missing_ids = HashSet::<SegmentId>::default();

        for &id in ids {
            if shared.contains_key(&id) {
                continue;
            }
            loop {
                match self.in_flight.entry(id) {
                    Entry::Occupied(entry) => {
                        if let Some(future) = entry.get().upgrade() {
                            shared.insert(id, future);
                            break;
                        }
                        entry.remove();
                    }
                    Entry::Vacant(_) => {
                        if missing_ids.insert(id) {
                            missing.push(id);
                        }
                        break;
                    }
                }
            }
        }

        let delegates = self.inner.request_background_batch(&missing);
        assert_eq!(
            delegates.len(),
            missing.len(),
            "SegmentSource::request_background_batch must return one future per ID"
        );
        for (id, delegate) in missing.into_iter().zip(delegates) {
            let future = delegate.map_err(Arc::new).boxed().shared();
            self.in_flight.insert(
                id,
                future
                    .downgrade()
                    .vortex_expect("just created, cannot be polled to completion"),
            );
            shared.insert(id, future);
        }

        ids.iter()
            .map(|id| shared[id].clone().map_err(VortexError::from).boxed())
            .collect()
    }

    fn request_nowait(&self, id: SegmentId) -> vortex_error::VortexResult<ReadAtNowait> {
        self.inner.request_nowait(id)
    }

    fn prefers_background_reads(&self) -> bool {
        self.inner.prefers_background_reads()
    }
}

impl<S: SegmentSource> SharedSegmentSource<S> {
    fn request_with(
        &self,
        id: SegmentId,
        request: impl Fn(&S, SegmentId) -> SegmentFuture,
    ) -> SegmentFuture {
        loop {
            match self.in_flight.entry(id) {
                Entry::Occupied(e) => {
                    if let Some(shared_future) = e.get().upgrade() {
                        return shared_future.map_err(VortexError::from).boxed();
                    } else {
                        // The future has been dropped, remove the entry and try again.
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    let future = request(&self.inner, id).map_err(Arc::new).boxed().shared();
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
        batch_count: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingSegmentSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.request_count.fetch_add(1, Ordering::SeqCst);
            self.segments.request(id)
        }

        fn request_background_batch(&self, ids: &[SegmentId]) -> Vec<SegmentFuture> {
            self.batch_count.fetch_add(1, Ordering::SeqCst);
            self.request_count.fetch_add(ids.len(), Ordering::SeqCst);
            ids.iter().map(|id| self.segments.request(*id)).collect()
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

    #[tokio::test]
    async fn test_shared_source_preserves_background_batch_and_deduplicates() -> VortexResult<()> {
        let source = CountingSegmentSource::default();
        let first = ByteBuffer::from(vec![1, 2]);
        let second = ByteBuffer::from(vec![3, 4]);
        source
            .segments
            .write(SequenceId::root().downgrade(), vec![first.clone()])
            .await?;
        source
            .segments
            .write(SequenceId::root().downgrade(), vec![second.clone()])
            .await?;

        let shared_source = SharedSegmentSource::new(source.clone());
        let futures = shared_source.request_background_batch(&[
            SegmentId::from(0),
            SegmentId::from(1),
            SegmentId::from(0),
        ]);
        let results = futures::future::try_join_all(futures).await?;

        assert_eq!(results[0].clone().unwrap_host(), first);
        assert_eq!(results[1].clone().unwrap_host(), second);
        assert_eq!(results[2].clone().unwrap_host(), first);
        assert_eq!(source.batch_count.load(Ordering::Relaxed), 1);
        assert_eq!(source.request_count.load(Ordering::Relaxed), 2);
        Ok(())
    }
}
