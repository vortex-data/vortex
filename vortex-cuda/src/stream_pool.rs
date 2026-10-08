// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA stream pool for managing and reusing streams.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use arc_swap::ArcSwapOption;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaStream;
use parking_lot::Mutex;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::stream::VortexCudaStream;

/// A pool of CUDA streams that hands out streams in a round-robin fashion.
///
/// Uses lock-free access to initialized slots via `ArcSwap`.
/// Streams are lazily created under a per-slot lock on first access and remain
/// alive for the lifetime of the pool.
pub struct VortexCudaStreamPool {
    context: Arc<CudaContext>,
    /// Fixed-size array of slots, each holding an optional stream.
    slots: Box<[StreamSlot]>,
    /// Round-robin counter for slot selection.
    next_index: AtomicUsize,
}

struct StreamSlot {
    stream: ArcSwapOption<CudaStream>,
    init_lock: Mutex<()>,
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
    pub fn new(context: Arc<CudaContext>, max_capacity: usize) -> Self {
        let slots = (0..max_capacity)
            .map(|_| StreamSlot {
                stream: ArcSwapOption::empty(),
                init_lock: Mutex::new(()),
            })
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
        let slot = &self.slots[slot_idx];

        // Fast path: stream already exists in slot.
        if let Some(stream) = slot.stream.load_full() {
            return Ok(VortexCudaStream(stream));
        }

        // Recheck under the per-slot lock so racing callers use the same stream.
        let _init_guard = slot.init_lock.lock();
        if let Some(stream) = slot.stream.load_full() {
            return Ok(VortexCudaStream(stream));
        }

        // Slow path: create a new stream.
        // Note: CudaContext::new_stream() already returns Arc<CudaStream>.
        let new_stream = self
            .context
            .new_stream()
            .map_err(|e| vortex_err!("Failed to create CUDA stream: {}", e))?;

        slot.stream.store(Some(Arc::clone(&new_stream)));

        Ok(VortexCudaStream(new_stream))
    }

    /// Returns the current number of initialized streams in the pool.
    pub fn live_stream_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.stream.load().is_some())
            .count()
    }

    /// Returns the maximum capacity of the pool.
    pub fn max_capacity(&self) -> usize {
        self.slots.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    use cudarc::driver::CudaContext;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;

    use super::VortexCudaStreamPool;

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
