// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

use super::SegmentFuture;
use super::SegmentId;
use super::SegmentSource;

/// A [`SegmentSource`] that allows multiple requesters to await the same underlying segment
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
            in_flight: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
        }
    }
}

impl<S: SegmentSource> SegmentSource for SharedSegmentSource<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
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
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use futures::channel::oneshot;
    use parking_lot::Mutex;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;
    use crate::test_harness::SimulatedExecutor;

    #[derive(Clone, Default)]
    struct GatedSegmentSource {
        inner: Arc<GatedSegmentSourceInner>,
    }

    type PendingSender = oneshot::Sender<VortexResult<BufferHandle>>;

    #[derive(Default)]
    struct GatedSegmentSourceInner {
        next_op: AtomicUsize,
        requests: AtomicUsize,
        dropped_before_completion: AtomicUsize,
        pending: Mutex<BTreeMap<(SegmentId, usize), PendingSender>>,
    }

    struct DropTracker {
        inner: Arc<GatedSegmentSourceInner>,
        key: (SegmentId, usize),
    }

    impl Drop for DropTracker {
        fn drop(&mut self) {
            if self.inner.pending.lock().remove(&self.key).is_some() {
                self.inner
                    .dropped_before_completion
                    .fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl GatedSegmentSource {
        fn request_count(&self) -> usize {
            self.inner.requests.load(Ordering::SeqCst)
        }

        fn cancelled_count(&self) -> usize {
            self.inner.dropped_before_completion.load(Ordering::SeqCst)
        }

        fn pending_count(&self) -> usize {
            self.inner.pending.lock().len()
        }

        fn resolve(&self, id: SegmentId, result: VortexResult<BufferHandle>) -> bool {
            let sender = {
                let mut pending = self.inner.pending.lock();
                let key = pending.keys().find(|(seg_id, _)| *seg_id == id).copied();
                key.and_then(|k| pending.remove(&k))
            };
            sender.is_some_and(|tx| tx.send(result).is_ok())
        }

        fn resolve_bytes(&self, id: SegmentId, bytes: &[u8]) -> bool {
            self.resolve(
                id,
                Ok(BufferHandle::new_host(ByteBuffer::from(bytes.to_vec()))),
            )
        }
    }

    impl SegmentSource for GatedSegmentSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            let op = self.inner.next_op.fetch_add(1, Ordering::SeqCst);
            self.inner.requests.fetch_add(1, Ordering::SeqCst);
            let (tx, rx) = oneshot::channel();
            let key = (id, op);
            self.inner.pending.lock().insert(key, tx);
            let inner = Arc::clone(&self.inner);
            async move {
                let _drop_guard = DropTracker { inner, key };
                rx.await
                    .map_err(|_| vortex_err!("gated segment source dropped while pending"))?
            }
            .boxed()
        }
    }

    #[test]
    fn test_concurrent_coalescing_deduplicates_inner_request() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let inner = GatedSegmentSource::default();
        let shared = SharedSegmentSource::new(inner.clone());

        let f1 = exec.handle().spawn(shared.request(SegmentId::from(0)));
        let f2 = exec.handle().spawn(shared.request(SegmentId::from(0)));
        let f3 = exec.handle().spawn(shared.request(SegmentId::from(0)));

        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 1);
        assert_eq!(inner.pending_count(), 1);

        assert!(inner.resolve_bytes(SegmentId::from(0), b"shared-payload"));

        for fut in [f1, f2, f3] {
            let buf = exec.drive(fut)?;
            assert_eq!(exec.drive(buf.to_host()).as_ref(), b"shared-payload");
        }
        assert_eq!(inner.request_count(), 1);
        Ok(())
    }

    #[test]
    fn test_out_of_order_completion_across_segments() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let inner = GatedSegmentSource::default();
        let shared = SharedSegmentSource::new(inner.clone());

        let f0 = exec.handle().spawn(shared.request(SegmentId::from(0)));
        let f1 = exec.handle().spawn(shared.request(SegmentId::from(1)));
        let f2 = exec.handle().spawn(shared.request(SegmentId::from(2)));

        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 3);

        // Resolve in reverse order: 2, 0, 1.
        assert!(inner.resolve_bytes(SegmentId::from(2), b"seg-2"));
        assert_eq!(exec.drive(exec.drive(f2)?.to_host()).as_ref(), b"seg-2");

        assert!(inner.resolve_bytes(SegmentId::from(0), b"seg-0"));
        assert_eq!(exec.drive(exec.drive(f0)?.to_host()).as_ref(), b"seg-0");

        assert!(inner.resolve_bytes(SegmentId::from(1), b"seg-1"));
        assert_eq!(exec.drive(exec.drive(f1)?.to_host()).as_ref(), b"seg-1");
        Ok(())
    }

    #[test]
    fn test_error_propagates_to_all_waiters_and_allows_retry() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let inner = GatedSegmentSource::default();
        let shared = SharedSegmentSource::new(inner.clone());
        let id = SegmentId::from(5);

        let f1 = exec.handle().spawn(shared.request(id));
        let f2 = exec.handle().spawn(shared.request(id));
        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 1);

        assert!(inner.resolve(id, Err(vortex_err!("transient segment read error"))));

        let err1 = exec
            .drive(f1)
            .expect_err("first waiter must receive shared error");
        let err2 = exec
            .drive(f2)
            .expect_err("second waiter must receive shared error");
        assert!(
            err1.to_string().contains("transient segment read error"),
            "{err1}"
        );
        assert!(
            err2.to_string().contains("transient segment read error"),
            "{err2}"
        );

        // Once all waiters have finished and dropped, a subsequent request retries cleanly.
        let retry = exec.handle().spawn(shared.request(id));
        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 2);
        assert!(inner.resolve_bytes(id, b"recovered"));
        assert_eq!(
            exec.drive(exec.drive(retry)?.to_host()).as_ref(),
            b"recovered"
        );
        Ok(())
    }

    #[test]
    fn test_partial_cancellation_keeps_inner_request_alive() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let inner = GatedSegmentSource::default();
        let shared = SharedSegmentSource::new(inner.clone());
        let id = SegmentId::from(7);

        let f1 = exec.handle().spawn(shared.request(id));
        let f2 = exec.handle().spawn(shared.request(id));
        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 1);

        // Drop the first waiter while the second remains active.
        drop(f1);
        exec.run_until_stalled();
        assert_eq!(inner.cancelled_count(), 0);
        assert_eq!(inner.pending_count(), 1);

        assert!(inner.resolve_bytes(id, b"still-alive"));
        assert_eq!(
            exec.drive(exec.drive(f2)?.to_host()).as_ref(),
            b"still-alive"
        );
        assert_eq!(inner.request_count(), 1);
        Ok(())
    }

    #[test]
    fn test_all_waiters_cancelled_drops_inner_and_reissues_on_next_request() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let inner = GatedSegmentSource::default();
        let shared = SharedSegmentSource::new(inner.clone());
        let id = SegmentId::from(9);

        let f1 = exec.handle().spawn(shared.request(id));
        let f2 = exec.handle().spawn(shared.request(id));
        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 1);

        drop(f1);
        drop(f2);
        exec.run_until_stalled();
        assert_eq!(inner.cancelled_count(), 1);
        assert_eq!(inner.pending_count(), 0);

        let fresh = exec.handle().spawn(shared.request(id));
        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 2);
        assert!(inner.resolve_bytes(id, b"reissued"));
        assert_eq!(
            exec.drive(exec.drive(fresh)?.to_host()).as_ref(),
            b"reissued"
        );
        Ok(())
    }

    #[test]
    fn test_request_after_completion_refetches_from_inner() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let inner = GatedSegmentSource::default();
        let shared = SharedSegmentSource::new(inner.clone());
        let id = SegmentId::from(11);

        let first = exec.handle().spawn(shared.request(id));
        exec.run_until_stalled();
        assert!(inner.resolve_bytes(id, b"v1"));
        assert_eq!(exec.drive(exec.drive(first)?.to_host()).as_ref(), b"v1");

        // All handles to the completed shared future are dropped, so the next call re-requests.
        let second = exec.handle().spawn(shared.request(id));
        exec.run_until_stalled();
        assert_eq!(inner.request_count(), 2);
        assert!(inner.resolve_bytes(id, b"v2"));
        assert_eq!(exec.drive(exec.drive(second)?.to_host()).as_ref(), b"v2");
        Ok(())
    }
}
