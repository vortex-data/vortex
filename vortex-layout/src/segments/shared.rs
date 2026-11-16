// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::WeakShared;
use futures::future::poll_fn;
use parking_lot::Mutex;
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
    in_flight: DashMap<SegmentId, InFlight>,
}

type SharedSegmentFuture = BoxFuture<'static, SharedVortexResult<BufferHandle>>;

struct InFlight {
    future: WeakShared<SharedSegmentFuture>,
    gate: Arc<Mutex<PollState>>,
}

#[derive(Default)]
struct PollState {
    polling: bool,
    waiters: Vec<Waker>,
}

struct PollGuard<'a>(&'a Mutex<PollState>);

impl Drop for PollGuard<'_> {
    fn drop(&mut self) {
        let waiters = {
            let mut state = self.0.lock();
            state.polling = false;
            std::mem::take(&mut state.waiters)
        };
        for waker in waiters {
            waker.wake();
        }
    }
}

fn shared_request(
    mut future: Shared<SharedSegmentFuture>,
    gate: Arc<Mutex<PollState>>,
) -> SegmentFuture {
    poll_fn(move |cx| {
        let mut state = gate.lock();
        if state.polling {
            if !state
                .waiters
                .iter()
                .any(|waker| waker.will_wake(cx.waker()))
            {
                state.waiters.push(cx.waker().clone());
            }
            return Poll::Pending;
        }
        state.polling = true;
        drop(state);

        // A temporary prefetch poll can consume a wake and then stop being driven. A reader
        // racing that poll must be woken after Shared releases its internal polling state.
        // The guard also releases the gate during unwinding, and never wakes under its lock.
        let guard = PollGuard(&gate);
        let result = future.poll_unpin(cx);
        drop(guard);
        result.map(|result| result.map_err(VortexError::from))
    })
    .boxed()
}

impl<S: SegmentSource> SharedSegmentSource<S> {
    /// Create a new `SharedSegmentSource` wrapping the provided inner source.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            in_flight: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
        }
    }
}

impl<S: SegmentSource> SegmentSource for SharedSegmentSource<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        loop {
            match self.in_flight.entry(id) {
                Entry::Occupied(e) => {
                    if let Some(shared_future) = e.get().future.upgrade() {
                        return shared_request(shared_future, Arc::clone(&e.get().gate));
                    } else {
                        // The future has been dropped, remove the entry and try again.
                        e.remove();
                    }
                }
                Entry::Vacant(e) => {
                    let future = self.inner.request(id).map_err(Arc::new).boxed().shared();
                    let gate = Arc::new(Mutex::new(PollState::default()));
                    e.insert(InFlight {
                        future: future
                            .downgrade()
                            .vortex_expect("just created, cannot be polled to completion"),
                        gate: Arc::clone(&gate),
                    });
                    return shared_request(future, gate);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use super::*;
    use crate::segments::CountingSegmentSource;
    use crate::segments::SegmentSink;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;

    #[tokio::test]
    async fn test_shared_source_deduplicates_concurrent_requests() {
        let segments = Arc::new(TestSegments::default());
        let source = CountingSegmentSource::new(Arc::clone(&segments) as _);

        // Add a segment to the test source
        let data = ByteBuffer::from(vec![1, 2, 3, 4]);
        let seq_id = SequenceId::root().downgrade();
        segments.write(seq_id, vec![data.clone()]).await.unwrap();

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
        assert_eq!(source.requests(), 1);
    }

    #[tokio::test]
    async fn test_shared_source_handles_dropped_futures() {
        let segments = Arc::new(TestSegments::default());
        let source = CountingSegmentSource::new(Arc::clone(&segments) as _);

        // Add a segment
        let data = ByteBuffer::from(vec![5, 6, 7, 8]);
        let seq_id = SequenceId::root().downgrade();
        segments.write(seq_id, vec![data.clone()]).await.unwrap();

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
        assert_eq!(source.requests(), 2);
    }
}
