// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::CudaViewMut;
use cudarc::driver::HostSlice;
use cudarc::driver::SyncOnDrop;
use cudarc::driver::result;
use cudarc::driver::sys::CUevent_flags;
use parking_lot::Mutex;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::utils::aliases::hash_map::HashMap;

use crate::CudaDeviceBuffer;
use crate::stream::VortexCudaStream;

/// A page-locked host buffer allocated by CUDA.
///
/// This is intended as a staging buffer for H2D transfers. Contents are uninitialized after
/// allocation.
pub(crate) struct PinnedByteBuffer {
    ptr: *mut u8,
    capacity: usize,
    logical_len: usize,
    event: CudaEvent,
}

// The allocation is uniquely owned, mutable only through `&mut self`, and retained by the pool
// while CUDA reads it asynchronously.
unsafe impl Send for PinnedByteBuffer {}
unsafe impl Sync for PinnedByteBuffer {}

impl PinnedByteBuffer {
    /// Allocate a pinned host buffer with a given capacity and logical length.
    ///
    /// # Safety
    /// The returned buffer's contents are uninitialized. The caller must initialize before read.
    pub(crate) unsafe fn uninit_with_capacity(
        ctx: &Arc<CudaContext>,
        capacity: usize,
        logical_len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            capacity < isize::MAX as usize,
            "pinned host buffer capacity is too large: {capacity}"
        );
        vortex_ensure!(
            logical_len <= capacity,
            "pinned host buffer length {logical_len} exceeds capacity {capacity}"
        );
        let event = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(|e| vortex_err!("failed to create pinned host buffer event: {e}"))?;
        // Keep file-I/O staging cacheable: cudarc's allocator uses write-combined memory,
        // which makes CPU reads expensive.
        let ptr = unsafe { result::malloc_host(capacity, 0) }
            .map_err(|e| vortex_err!("failed to allocate pinned host buffer: {e}"))?
            .cast::<u8>();
        vortex_ensure!(!ptr.is_null(), "CUDA returned a null pinned host buffer");
        Ok(Self {
            ptr,
            capacity,
            logical_len,
            event,
        })
    }

    /// Returns the buffer as a mutable slice.
    pub(crate) fn as_mut_slice(&mut self) -> VortexResult<&mut [u8]> {
        self.event
            .synchronize()
            .map_err(|e| vortex_err!("failed to access pinned host buffer: {e}"))?;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, self.logical_len) })
    }

    fn set_logical_len(&mut self, len: usize) {
        assert!(len <= self.capacity);
        self.logical_len = len;
    }
}

/// A borrowed range that preserves the pinned buffer's stream synchronization.
struct PinnedByteBufferView<'a> {
    buffer: &'a mut PinnedByteBuffer,
    range: Range<usize>,
}

impl HostSlice<u8> for PinnedByteBufferView<'_> {
    fn len(&self) -> usize {
        self.range.len()
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        stream.context().record_err(stream.wait(&self.buffer.event));
        // SAFETY: The view retains the allocation, and the caller must use the returned slice
        // only with this stream. The guard records completion on the owner's event.
        let slice = unsafe { std::slice::from_raw_parts(self.buffer.ptr, self.buffer.logical_len) };
        let sync = SyncOnDrop::Record(Some((&self.buffer.event, stream)));
        (&slice[self.range.clone()], sync)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        stream.context().record_err(stream.wait(&self.buffer.event));
        // SAFETY: The view exclusively borrows the allocation, and the caller must use the
        // returned slice only with this stream. The guard tracks writes on the owner's event.
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.buffer.ptr, self.buffer.logical_len) };
        let sync = SyncOnDrop::Record(Some((&self.buffer.event, stream)));
        (&mut slice[self.range.clone()], sync)
    }
}

impl Drop for PinnedByteBuffer {
    fn drop(&mut self) {
        let context = self.event.context();
        context.record_err(self.event.synchronize());
        context.record_err(unsafe { result::free_host(self.ptr.cast()) });
    }
}

/// A simple pinned host buffer pool keyed by allocation size.
///
/// Requested sizes are rounded up to the next power of two so that buffers are reusable across
/// similar-but-not-identical request sizes.
pub struct PinnedByteBufferPool {
    ctx: Arc<CudaContext>,
    max_keep_per_size: usize,
    buckets: Mutex<HashMap<usize, Vec<PinnedByteBuffer>>>,
    inflight: Mutex<Vec<InflightPinnedBuffer>>,
    hits: AtomicU64,
    misses: AtomicU64,
    allocs: AtomicU64,
    puts: AtomicU64,
}

impl std::fmt::Debug for PinnedByteBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedByteBufferPool")
            .field("max_keep_per_size", &self.max_keep_per_size)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

struct InflightPinnedBuffer {
    event: CudaEvent,
    buffer: PinnedByteBuffer,
}

impl PinnedByteBufferPool {
    /// Create a new pool with default limits.
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self::with_limits(ctx, 256)
    }

    /// Create a new pool with a maximum number of cached buffers per size class.
    pub fn with_limits(ctx: Arc<CudaContext>, max_keep_per_size: usize) -> Self {
        Self {
            ctx,
            max_keep_per_size: max_keep_per_size.max(1),
            buckets: Mutex::new(HashMap::new()),
            inflight: Mutex::new(Vec::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            allocs: AtomicU64::new(0),
            puts: AtomicU64::new(0),
        }
    }

    /// Try to get a pooled pinned buffer without performing expensive allocation.
    ///
    /// Returns `Ok(None)` if no buffer is available in the pool for the requested size class.
    /// Unlike `get`, this will never call `cuMemAllocHost`.
    pub fn try_get(self: &Arc<Self>, len: usize) -> VortexResult<Option<PooledPinnedBuffer>> {
        self.reclaim_completed()?;
        let key_len = self.size_class_len(len);
        let mut buckets = self.buckets.lock();
        if let Some(bucket) = buckets.get_mut(&key_len)
            && let Some(mut buf) = bucket.pop()
        {
            self.hits.fetch_add(1, Ordering::Relaxed);
            buf.set_logical_len(len);
            Ok(Some(PooledPinnedBuffer::new(buf, Arc::clone(self))))
        } else {
            Ok(None)
        }
    }

    /// Acquire a pooled pinned buffer of the given size in bytes.
    ///
    /// The buffer is returned to the pool when the [`PooledPinnedBuffer`] is dropped.
    pub(crate) fn get(self: &Arc<Self>, len: usize) -> VortexResult<PooledPinnedBuffer> {
        if let Some(buffer) = self.try_get(len)? {
            return Ok(buffer);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.allocs.fetch_add(1, Ordering::Relaxed);
        // SAFETY: Callers initialize the staging buffer before submitting a transfer.
        let inner = unsafe {
            PinnedByteBuffer::uninit_with_capacity(&self.ctx, self.size_class_len(len), len)?
        };
        Ok(PooledPinnedBuffer::new(inner, Arc::clone(self)))
    }

    /// Snapshot pool reuse statistics.
    pub fn stats(&self) -> PinnedPoolStats {
        PinnedPoolStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            allocs: self.allocs.load(Ordering::Relaxed),
            puts: self.puts.load(Ordering::Relaxed),
        }
    }

    fn size_class_len(&self, len: usize) -> usize {
        if len == 0 { 0 } else { len.next_power_of_two() }
    }

    /// Reclaim inflight pinned buffers whose CUDA completion events have fired.
    ///
    /// Completed buffers are moved back into size-class buckets for reuse.
    fn reclaim_completed(&self) -> VortexResult<()> {
        let mut inflight = self.inflight.lock();
        if inflight.is_empty() {
            return Ok(());
        }
        self.ctx
            .bind_to_thread()
            .map_err(|e| vortex_err!("Failed to bind CUDA context: {e}"))?;
        let mut idx = 0usize;
        while idx < inflight.len() {
            if !inflight[idx].event.is_complete() {
                idx += 1;
                continue;
            }
            let completed = inflight.swap_remove(idx);
            self.put(completed.buffer);
        }
        Ok(())
    }

    fn put(&self, buf: PinnedByteBuffer) {
        let len = buf.capacity;
        let overflow = {
            let mut buckets = self.buckets.lock();
            let bucket = buckets.entry(len).or_default();
            if bucket.len() < self.max_keep_per_size {
                bucket.push(buf);
                None
            } else {
                Some(buf)
            }
        };
        // If the pool is full, the buffer (cuMemFreeHost) is dropped outside the lock.
        drop(overflow);
        self.puts.fetch_add(1, Ordering::Relaxed);
    }
}

/// Reuse counters for a pinned buffer pool.
#[derive(Clone, Copy, Debug, Default)]
pub struct PinnedPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub allocs: u64,
    pub puts: u64,
}

/// A pinned buffer that is returned to its pool when dropped.
///
/// This wrapper owns a pinned byte buffer and ensures it gets returned to the
/// [`PinnedByteBufferPool`] when the buffer is no longer needed. This enables efficient
/// buffer reuse for I/O operations.
pub struct PooledPinnedBuffer {
    inner: Option<PinnedByteBuffer>,
    pool: Arc<PinnedByteBufferPool>,
}

impl PooledPinnedBuffer {
    /// Create a new pooled buffer.
    pub(crate) fn new(inner: PinnedByteBuffer, pool: Arc<PinnedByteBufferPool>) -> Self {
        Self {
            inner: Some(inner),
            pool,
        }
    }

    /// Returns the buffer as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the buffer has already been consumed or if the CUDA context is invalid.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner
            .as_mut()
            .vortex_expect("buffer already consumed")
            .as_mut_slice()
            .vortex_expect("failed to access pinned host buffer")
    }

    /// Shortens the logical length without reallocating the pinned buffer.
    ///
    /// This is an O(1) metadata update; the allocation capacity is unchanged.
    #[cfg(target_os = "linux")]
    pub(crate) fn truncate(&mut self, len: usize) {
        let inner = self.inner.as_mut().vortex_expect("buffer already consumed");
        assert!(len <= inner.logical_len);
        inner.set_logical_len(len);
    }

    /// Submits a non-blocking H2D DMA transfer and returns a device buffer.
    ///
    /// The pinned buffer is placed in the pool's inflight queue, gated on a `CudaEvent` marking
    /// the transfer completion. The pool reclaims it once the event fires.
    pub fn transfer_to_device(self, stream: &VortexCudaStream) -> VortexResult<CudaDeviceBuffer> {
        let len = self
            .inner
            .as_ref()
            .vortex_expect("buffer already consumed")
            .logical_len;
        let mut cuda_slice = stream.device_alloc::<u8>(len)?;
        self.copy_to_device(stream, 0..len, &mut cuda_slice.slice_mut(..))?;
        Ok(CudaDeviceBuffer::new(cuda_slice))
    }

    /// Submits a non-blocking H2D copy of `range` into an equally sized destination view.
    ///
    /// Validates the range against the source's logical length and the destination length before
    /// enqueuing. The pool retains the entire pinned allocation until completion, even if the
    /// caller drops the destination or stops waiting.
    pub(crate) fn copy_to_device(
        mut self,
        stream: &VortexCudaStream,
        range: Range<usize>,
        destination: &mut CudaViewMut<'_, u8>,
    ) -> VortexResult<()> {
        let pinned = self.inner.as_mut().vortex_expect("buffer already consumed");
        vortex_ensure!(
            range.start <= range.end && range.end <= pinned.logical_len,
            "invalid pinned host buffer range {:?} for length {}",
            range,
            pinned.logical_len
        );
        vortex_ensure!(
            range.len() == destination.len(),
            "pinned host buffer range length {} does not match destination length {}",
            range.len(),
            destination.len()
        );

        let source = PinnedByteBufferView {
            buffer: pinned,
            range,
        };
        // The page-locked source and its completion event keep this asynchronous.
        stream
            .memcpy_htod(&source, destination)
            .map_err(|e| vortex_err!("Failed to schedule H2D copy: {}", e))?;

        let event = stream
            .record_event(None)
            .map_err(|e| vortex_err!("Failed to record CUDA event: {}", e))?;

        // On earlier errors, Drop returns the buffer to the pool, but the HostSlice event still
        // gates access and freeing. On success, the inflight queue retains it until completion.
        let inner = self.inner.take().vortex_expect("buffer already consumed");
        self.pool.inflight.lock().push(InflightPinnedBuffer {
            event,
            buffer: inner,
        });
        Ok(())
    }
}

impl Drop for PooledPinnedBuffer {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            self.pool.put(inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::CudaContext;
    use rstest::rstest;
    use vortex::array::buffer::DeviceBuffer;
    use vortex::buffer::Alignment;
    use vortex::error::VortexResult;

    use super::*;

    fn setup() -> VortexResult<(Arc<PinnedByteBufferPool>, VortexCudaStream)> {
        let ctx = CudaContext::new(0).map_err(|e| vortex_err!("Failed to initialize CUDA: {e}"))?;
        let pool = Arc::new(PinnedByteBufferPool::new(Arc::clone(&ctx)));
        let stream = VortexCudaStream(
            ctx.new_stream()
                .map_err(|e| vortex_err!("Failed to create stream: {e}"))?,
        );
        Ok((pool, stream))
    }

    #[rstest]
    #[case::length_exceeds_capacity(16, 17)]
    #[case::capacity_too_large(usize::MAX, 1)]
    #[crate::test]
    fn rejects_invalid_allocation_size(
        #[case] capacity: usize,
        #[case] logical_len: usize,
    ) -> VortexResult<()> {
        let (pool, _stream) = setup()?;
        assert!(
            unsafe { PinnedByteBuffer::uninit_with_capacity(&pool.ctx, capacity, logical_len) }
                .is_err()
        );
        Ok(())
    }

    #[crate::test]
    fn transfer_to_device_round_trip() -> VortexResult<()> {
        let (pool, stream) = setup()?;
        let data: Vec<u8> = (0..=255u8).collect();

        let mut pinned = pool.get(data.len())?;
        pinned.as_mut_slice().copy_from_slice(&data);

        let device_buf = pinned.transfer_to_device(&stream)?;

        let host_buf = device_buf.copy_to_host_sync(Alignment::of::<u8>())?;
        assert_eq!(host_buf.as_ref(), &data[..]);
        Ok(())
    }

    #[rstest]
    #[case::nonzero_range(3, 9, 2..8, false)]
    #[case::empty_at_end(10, 10, 4..4, false)]
    #[case::reversed(6, 2, 2..6, true)]
    #[case::end_past_logical_length(4, 11, 2..9, true)]
    #[case::destination_too_short(2, 6, 2..5, true)]
    #[case::destination_too_long(2, 6, 2..7, true)]
    #[crate::test]
    fn copy_to_device_subview(
        #[case] start: usize,
        #[case] end: usize,
        #[case] destination_range: Range<usize>,
        #[case] reject: bool,
    ) -> VortexResult<()> {
        let (pool, stream) = setup()?;
        let data: Vec<u8> = (0..10).collect();
        let mut pinned = pool.get(data.len())?;
        pinned.as_mut_slice().copy_from_slice(&data);
        let mut expected = vec![0xA5u8; 12];
        let mut destination = stream
            .clone_htod(&expected)
            .map_err(|e| vortex_err!("Failed to initialize destination: {e}"))?;
        let result = pinned.copy_to_device(
            &stream,
            start..end,
            &mut destination.slice_mut(destination_range.clone()),
        );
        if reject {
            assert!(result.is_err());
            assert!(pool.inflight.lock().is_empty());
            assert_eq!(pool.stats().puts, 1);
        } else {
            result?;
            expected[destination_range].copy_from_slice(&data[start..end]);
        }
        let host = CudaDeviceBuffer::new(destination).copy_to_host_sync(Alignment::of::<u8>())?;
        assert_eq!(host.as_ref(), &expected[..]);
        Ok(())
    }

    #[rstest]
    #[crate::test]
    fn transfer_retains_and_reuses_source(#[values(false, true)] ranged: bool) -> VortexResult<()> {
        let (pool, stream) = setup()?;
        let mut pinned = pool.get(1024)?;
        pinned.as_mut_slice().fill(0xAB);
        let allocation = pinned.as_mut_slice().as_ptr();
        let destination = if ranged {
            let mut destination = stream.device_alloc::<u8>(256)?;
            pinned.copy_to_device(&stream, 128..384, &mut destination.slice_mut(..))?;
            CudaDeviceBuffer::new(destination)
        } else {
            pinned.transfer_to_device(&stream)?
        };

        assert_eq!(pool.stats().allocs, 1);
        assert_eq!(pool.stats().hits, 0);
        assert_eq!(pool.stats().puts, 0);
        {
            let inflight = pool.inflight.lock();
            assert_eq!(inflight.len(), 1);
            assert_eq!(inflight[0].buffer.ptr.cast_const(), allocation);
        }
        stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to sync stream: {e}"))?;

        let mut reused = pool.get(1024)?;
        assert_eq!(reused.as_mut_slice().as_ptr(), allocation);
        assert!(pool.inflight.lock().is_empty());
        assert_eq!(pool.stats().allocs, 1);
        assert_eq!(pool.stats().hits, 1);
        assert_eq!(pool.stats().puts, 1);
        reused.as_mut_slice().fill(0xCD);
        let host = destination.copy_to_host_sync(Alignment::of::<u8>())?;
        assert_eq!(host.as_ref(), vec![0xAB; if ranged { 256 } else { 1024 }]);
        pool.reclaim_completed()?;
        assert_eq!(pool.stats().puts, 1);
        Ok(())
    }

    #[crate::test]
    fn copy_to_device_pool_drop_waits_for_source() -> VortexResult<()> {
        let (pool, stream) = setup()?;
        let len = 1024 * 1024;
        let mut pinned = pool.get(len)?;
        pinned.as_mut_slice().fill(0xEF);
        let mut destination = stream.device_alloc::<u8>(len - 16)?;
        pinned.copy_to_device(&stream, 16..len, &mut destination.slice_mut(..))?;

        // Dropping the last pool owner must not free the source while DMA still reads it.
        drop(pool);
        let host = CudaDeviceBuffer::new(destination).copy_to_host_sync(Alignment::of::<u8>())?;
        assert_eq!(host.as_ref(), vec![0xEF; len - 16].as_slice());
        Ok(())
    }

    #[rstest]
    #[crate::test]
    fn drop_returns_buffer_to_pool(#[values(false, true)] try_get: bool) -> VortexResult<()> {
        let (pool, _stream) = setup()?;
        assert!(pool.try_get(512)?.is_none());
        assert_eq!(pool.stats().allocs, 0);
        assert_eq!(pool.stats().misses, 0);

        let allocation = {
            let mut pinned = pool.get(512)?;
            pinned.as_mut_slice().fill(0);
            pinned.as_mut_slice().as_ptr()
        };

        let stats = pool.stats();
        assert_eq!(stats.puts, 1);
        assert_eq!(stats.allocs, 1);
        assert_eq!(stats.misses, 1);

        // A shorter request in the same size class reuses storage and updates the logical length.
        let mut reused = if try_get {
            pool.try_get(300)?.expect("expected cached pinned buffer")
        } else {
            pool.get(300)?
        };
        assert_eq!(reused.as_mut_slice().len(), 300);
        assert_eq!(reused.as_mut_slice().as_ptr(), allocation);
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.allocs, 1);
        assert_eq!(stats.misses, 1);

        Ok(())
    }
}
