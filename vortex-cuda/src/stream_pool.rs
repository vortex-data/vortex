// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA stream pool for managing and reusing streams.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use once_cell::sync::OnceCell;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex_cuda_driver::CudaContext;
use vortex_cuda_driver::CudaStream;

use crate::stream::VortexCudaStream;

/// A pool of CUDA streams that hands out streams in a round-robin fashion.
///
/// Streams are lazily initialized once per slot and remain alive for the lifetime of the pool.
/// Access to initialized slots is lock-free; failed initialization can be retried.
pub struct VortexCudaStreamPool {
    context: Arc<CudaContext>,
    /// Fixed-size array of slots, each holding an optional stream.
    slots: Box<[OnceCell<Arc<CudaStream>>]>,
    /// Round-robin counter for slot selection.
    next_index: AtomicUsize,
}

impl std::fmt::Debug for VortexCudaStreamPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexCudaStreamPool")
            .field("max_capacity", &self.slots.len())
            .field("live_streams", &self.live_stream_count())
            .finish()
    }
}

impl VortexCudaStreamPool {
    /// Creates a new stream pool with the given CUDA context and maximum capacity.
    ///
    /// # Arguments
    ///
    /// * `context` - The CUDA context for creating streams.
    /// * `max_capacity` - Maximum number of streams to maintain in the pool.
    ///
    /// # Panics
    ///
    /// Panics if `max_capacity` is zero, as the pool could never return a stream.
    pub fn new(context: Arc<CudaContext>, max_capacity: usize) -> Self {
        // Fail here rather than on the first `stream`, which would divide by zero.
        assert!(
            max_capacity > 0,
            "CUDA stream pool capacity must be nonzero"
        );
        let slots = (0..max_capacity)
            .map(|_| OnceCell::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            context,
            slots,
            next_index: AtomicUsize::new(0),
        }
    }

    /// Returns a stream from the pool.
    ///
    /// Uses round-robin slot selection. If the selected slot has a stream,
    /// it is reused. Otherwise, a new stream is created for that slot.
    /// Access to initialized slots is lock-free; initialization is serialized per slot.
    pub fn stream(&self) -> VortexResult<VortexCudaStream> {
        let slot_idx = self.next_index.fetch_add(1, Ordering::Relaxed) % self.slots.len();
        let stream = self.slots[slot_idx].get_or_try_init(|| {
            self.context
                .new_stream()
                .map_err(|e| vortex_err!("Failed to create CUDA stream: {}", e))
        })?;

        Ok(VortexCudaStream(Arc::clone(stream)))
    }

    /// Returns the current number of initialized streams in the pool.
    pub fn live_stream_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.get().is_some())
            .count()
    }

    /// Returns the maximum capacity of the pool.
    pub fn max_capacity(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    use vortex::error::VortexResult;
    use vortex::error::vortex_err;
    use vortex::utils::aliases::hash_set::HashSet;
    use vortex_cuda_driver::CudaContext;

    use super::VortexCudaStreamPool;

    #[crate::test]
    fn rejects_zero_capacity() -> VortexResult<()> {
        let context =
            CudaContext::new(0).map_err(|e| vortex_err!("Failed to create CUDA context: {e}"))?;
        let pool =
            std::panic::catch_unwind(AssertUnwindSafe(|| VortexCudaStreamPool::new(context, 0)));
        assert!(pool.is_err());
        Ok(())
    }

    #[crate::test]
    fn lazy_creation() -> VortexResult<()> {
        let context =
            CudaContext::new(0).map_err(|e| vortex_err!("Failed to create CUDA context: {e}"))?;
        let pool = VortexCudaStreamPool::new(context, 3);

        assert_eq!(pool.max_capacity(), 3);
        assert_eq!(pool.live_stream_count(), 0);
        for expected_live in 1..=3 {
            let _stream = pool.stream()?;
            assert_eq!(pool.live_stream_count(), expected_live);
        }
        let _stream = pool.stream()?;
        assert_eq!(pool.live_stream_count(), 3);
        Ok(())
    }

    #[crate::test]
    fn round_robin_reuse() -> VortexResult<()> {
        let context =
            CudaContext::new(0).map_err(|e| vortex_err!("Failed to create CUDA context: {e}"))?;
        let pool = VortexCudaStreamPool::new(context, 3);
        let streams = (0..pool.max_capacity())
            .map(|_| pool.stream())
            .collect::<VortexResult<Vec<_>>>()?;
        let handles = streams
            .iter()
            .map(|stream| stream.cu_stream() as usize)
            .collect::<HashSet<_>>();
        assert_eq!(handles.len(), pool.max_capacity());

        for _ in 0..3 {
            for expected in &streams {
                let stream = pool.stream()?;
                assert!(Arc::ptr_eq(&stream.0, &expected.0));
                assert_eq!(stream.cu_stream(), expected.cu_stream());
            }
        }
        assert_eq!(pool.live_stream_count(), pool.max_capacity());
        Ok(())
    }

    #[crate::test]
    fn concurrent_initialization_respects_capacity() -> VortexResult<()> {
        let context =
            CudaContext::new(0).map_err(|e| vortex_err!("Failed to create CUDA context: {e}"))?;
        for capacity in [1, 4] {
            let pool = VortexCudaStreamPool::new(Arc::clone(&context), capacity);
            let barrier = Barrier::new(32);
            let streams = thread::scope(|scope| {
                let workers = (0..32)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            (0..8)
                                .map(|_| pool.stream())
                                .collect::<VortexResult<Vec<_>>>()
                        })
                    })
                    .collect::<Vec<_>>();
                workers
                    .into_iter()
                    .map(|worker| worker.join().expect("stream worker panicked"))
                    .collect::<VortexResult<Vec<_>>>()
            })?;

            // Keep every returned stream alive so transient handles cannot be recycled.
            let handles = streams
                .iter()
                .flatten()
                .map(|stream| stream.cu_stream() as usize)
                .collect::<HashSet<_>>();
            assert_eq!(handles.len(), capacity);
            assert_eq!(pool.live_stream_count(), capacity);
            let stored = (0..capacity)
                .map(|_| pool.stream())
                .collect::<VortexResult<Vec<_>>>()?;
            for stream in streams.iter().flatten() {
                assert!(stored.iter().any(|slot| Arc::ptr_eq(&stream.0, &slot.0)));
            }
        }
        Ok(())
    }
}
