// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::WeakShared;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use super::SegmentFuture;
use super::SegmentId;
use super::SegmentSource;

/// A [`SegmentSource`] that allows multiple requesters to await the same underlying segment
/// request.
///
/// Many threads may request segments of one source at once, for example the workers of a scan
/// pool sharing one file's splits. Requests therefore look for a request in flight under a shared
/// lock, call the inner source outside any lock, and take an exclusive lock only to record a new
/// request.
pub struct SharedSegmentSource<S> {
    inner: S,
    in_flight: DashMap<SegmentId, WeakShared<SharedSegmentFuture>>,
}

type SharedSegmentFuture = BoxFuture<'static, SharedVortexResult<BufferHandle>>;

impl<S: SegmentSource> SharedSegmentSource<S> {
    /// Create a new `SharedSegmentSource` wrapping the provided inner source.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            // The default shard count scales with the machine's cores, as the requesters do.
            in_flight: DashMap::default(),
        }
    }

    /// The request in flight for `id`, if any.
    fn in_flight(&self, id: SegmentId) -> Option<Shared<SharedSegmentFuture>> {
        self.in_flight.get(&id)?.upgrade()
    }
}

impl<S: SegmentSource> SegmentSource for SharedSegmentSource<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        if let Some(future) = self.in_flight(id) {
            return future.map_err(VortexError::from).boxed();
        }

        let future = self.inner.request(id).map_err(Arc::new).boxed().shared();
        let weak = future
            .downgrade()
            .vortex_expect("just created, cannot be polled to completion");
        match self.in_flight.entry(id) {
            Entry::Occupied(mut entry) => {
                if let Some(existing) = entry.get().upgrade() {
                    // Another requester recorded the same segment first. Its request reads the
                    // same bytes, so dropping this one changes nothing that is read or coalesced.
                    return existing.map_err(VortexError::from).boxed();
                }
                entry.insert(weak);
            }
            Entry::Vacant(entry) => {
                entry.insert(weak);
            }
        }
        future.map_err(VortexError::from).boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use futures::executor::block_on;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use super::*;

    /// Counts its requests, and answers each with the segment's id as one byte.
    #[derive(Default)]
    struct Counting(AtomicUsize);

    impl SegmentSource for Counting {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.0.fetch_add(1, Ordering::SeqCst);
            let byte = u8::try_from(*id).unwrap_or(u8::MAX);
            async move { Ok(BufferHandle::new_host(ByteBuffer::from(vec![byte]))) }.boxed()
        }
    }

    #[test]
    fn concurrent_requests_share_one_inner_request() -> VortexResult<()> {
        let source = SharedSegmentSource::new(Counting::default());
        let first = source.request(SegmentId::from(7));
        let second = source.request(SegmentId::from(7));
        assert_eq!(source.inner.0.load(Ordering::SeqCst), 1);
        assert_eq!(block_on(first)?.to_host_sync().as_slice(), [7]);
        assert_eq!(block_on(second)?.to_host_sync().as_slice(), [7]);
        Ok(())
    }

    #[test]
    fn a_finished_request_is_made_again() -> VortexResult<()> {
        let source = SharedSegmentSource::new(Counting::default());
        drop(block_on(source.request(SegmentId::from(3)))?);
        drop(block_on(source.request(SegmentId::from(3)))?);
        assert_eq!(source.inner.0.load(Ordering::SeqCst), 2);
        Ok(())
    }

    /// Threads requesting the same segments at once each get their segment.
    #[test]
    fn racing_requesters_get_their_segments() -> VortexResult<()> {
        let source = Arc::new(SharedSegmentSource::new(Counting::default()));
        let futures = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let source = Arc::clone(&source);
                    scope.spawn(move || {
                        (0..64u32)
                            .map(|id| source.request(SegmentId::from(id)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("requester panicked"))
                .collect::<Vec<_>>()
        });
        // A requester that loses a race drops its own inner request, so there may be more than
        // one per segment, but never fewer.
        assert!(source.inner.0.load(Ordering::SeqCst) >= 64);
        for (index, future) in futures.into_iter().enumerate() {
            let id = u8::try_from(index % 64).unwrap_or(u8::MAX);
            assert_eq!(block_on(future)?.to_host_sync().as_slice(), [id]);
        }
        Ok(())
    }
}
