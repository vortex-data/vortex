// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA stream pool for managing and reusing streams.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use arc_swap::ArcSwapOption;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaStream;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::stream::VortexCudaStream;

/// A pool of CUDA streams that hands out streams in a round-robin fashion.
///
/// Uses lock-free slot access via `ArcSwap` for high-concurrency scenarios.
/// Streams are lazily created on first access to each slot and remain alive
/// for the lifetime of the pool.
pub struct VortexCudaStreamPool {
    context: Arc<CudaContext>,
    /// Fixed-size array of slots, each holding an optional stream.
    slots: Box<[ArcSwapOption<CudaStream>]>,
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
    pub fn new(context: Arc<CudaContext>, max_capacity: usize) -> Self {
        let slots = (0..max_capacity)
            .map(|_| ArcSwapOption::empty())
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
    /// All operations are lock-free.
    pub fn stream(&self) -> VortexResult<VortexCudaStream> {
        let slot_idx = self.next_index.fetch_add(1, Ordering::Relaxed) % self.slots.len();
        let slot = &self.slots[slot_idx];

        // Fast path: stream already exists in slot.
        if let Some(stream) = slot.load_full() {
            return Ok(VortexCudaStream(stream));
        }

        // Slow path: create a new stream.
        // Note: CudaContext::new_stream() already returns Arc<CudaStream>.
        let new_stream = self
            .context
            .new_stream()
            .map_err(|e| vortex_err!("Failed to create CUDA stream: {}", e))?;

        // Store it in the slot. If another thread raced us, that's fine -
        // we'll just use our newly created stream this time.
        slot.store(Some(Arc::clone(&new_stream)));

        Ok(VortexCudaStream(new_stream))
    }

    /// Returns the current number of initialized streams in the pool.
    pub fn live_stream_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.load().is_some())
            .count()
    }

    /// Returns the maximum capacity of the pool.
    pub fn max_capacity(&self) -> usize {
        self.slots.len()
    }
}
